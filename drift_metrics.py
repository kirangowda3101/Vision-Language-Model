"""
Drift signals for the CLIP model, computed per batch against a reference
window of clean images.

Everything in this file is pure numpy (no model, no dataset), so it can be
unit-tested in isolation and is shared verbatim by the offline experiment
(drift_experiment.py) and the serving-time monitor (drift_monitor.py). That
way the numbers reported in the README come from exactly the code path that
would run in production.

Three families of signals, all oriented so that LARGER = MORE DRIFT:
  - input statistics: brightness, contrast, sharpness (Laplacian variance)
  - embedding drift: cosine distance between mean embeddings, and an MMD
    two-sample test with an RBF kernel and a permutation p-value
  - confidence drift: mean top-class probability and mean entropy of the
    zero-shot class distribution
"""

import numpy as np

SIGNALS = (
    "input_stats",
    "embedding_cosine",
    "embedding_mmd",
    "confidence_top_prob",
    "confidence_entropy",
)

INPUT_STAT_NAMES = ("brightness", "contrast", "sharpness")

# ITU-R BT.601 luma weights -- the standard RGB -> grayscale conversion
# (also what PIL's convert("L") uses).
_LUMA = np.array([0.299, 0.587, 0.114])


# --- Input statistics ------------------------------------------------------

def image_stats(images):
    """
    Per-image input statistics for a batch of RGB uint8 images.

    images: (N, H, W, 3) uint8 array (or a single (H, W, 3) image).
    Returns an (N, 3) float64 array with columns:
      brightness: mean grayscale intensity, in [0, 1]
      contrast:   standard deviation of grayscale intensity (RMS contrast)
      sharpness:  variance of the 4-neighbour Laplacian of the grayscale
                  image. Blur suppresses high frequencies (variance drops),
                  noise adds them (variance rises).

    The Laplacian is only evaluated on interior pixels, so no padding mode
    leaks artificial edges into the sharpness estimate.
    """
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    gray = (images.astype(np.float64) / 255.0) @ _LUMA  # (N, H, W)

    brightness = gray.mean(axis=(1, 2))
    contrast = gray.std(axis=(1, 2))

    laplacian = (
        gray[:, :-2, 1:-1]
        + gray[:, 2:, 1:-1]
        + gray[:, 1:-1, :-2]
        + gray[:, 1:-1, 2:]
        - 4.0 * gray[:, 1:-1, 1:-1]
    )
    sharpness = laplacian.var(axis=(1, 2))

    return np.stack([brightness, contrast, sharpness], axis=1)


def _input_features(stats):
    """
    Transform raw stats into the space the mean-shift test runs in.
    Laplacian variance is heavy-tailed and multiplicative (noise can raise
    it by orders of magnitude, blur shrinks it towards 0), so it's compared
    on a log scale (the epsilon only matters for perfectly flat images);
    brightness and contrast are used as-is.
    """
    stats = np.asarray(stats, dtype=np.float64)
    return np.stack([stats[:, 0], stats[:, 1], np.log(stats[:, 2] + 1e-6)], axis=1)


# --- Zero-shot confidence --------------------------------------------------

def softmax(logits, axis=-1):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def zero_shot_probs(image_embeddings, label_embeddings, logit_scale):
    """
    Class probabilities the same way the model was trained to produce
    them: softmax(logit_scale * cosine_similarity). Both inputs are
    L2-normalized here, so callers can pass raw encoder outputs.
    """
    image_embeddings = l2_normalize(image_embeddings)
    label_embeddings = l2_normalize(label_embeddings)
    return softmax(logit_scale * image_embeddings @ label_embeddings.T, axis=1)


def confidence_stats(probs):
    """
    probs: (N, C) class probabilities. Returns an (N, 2) array with
    columns [top-class probability, entropy in nats].
    """
    probs = np.asarray(probs, dtype=np.float64)
    top_prob = probs.max(axis=1)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    return np.stack([top_prob, entropy], axis=1)


# --- Embedding drift ---------------------------------------------------------

def l2_normalize(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12, None)


def cosine_distance_of_means(reference_embeddings, batch_embeddings):
    """1 - cos(mean reference embedding, mean batch embedding)."""
    mu_ref = reference_embeddings.mean(axis=0)
    mu_batch = batch_embeddings.mean(axis=0)
    cos = mu_ref @ mu_batch / max(np.linalg.norm(mu_ref) * np.linalg.norm(mu_batch), 1e-12)
    return float(1.0 - cos)


def squared_distances(x, y):
    """Pairwise squared Euclidean distances, (len(x), len(y)), clipped at 0."""
    sq = (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * x @ y.T
    return np.clip(sq, 0.0, None)


def rbf_kernel(x, y, bandwidth):
    """k(a, b) = exp(-||a - b||^2 / (2 * bandwidth^2))."""
    return np.exp(-squared_distances(x, y) / (2.0 * bandwidth**2))


def median_heuristic_bandwidth(x, max_points=1000, rng=None):
    """
    Median pairwise distance among (a subsample of) x -- the standard
    default RBF bandwidth for MMD. It's computed ONCE from the reference
    and then frozen: re-estimating it per batch would change the kernel,
    and therefore the meaning of the score, from batch to batch.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    if len(x) > max_points:
        x = x[rng.choice(len(x), max_points, replace=False)]
    d = np.sqrt(squared_distances(x, x))
    return float(np.median(d[np.triu_indices(len(x), k=1)]))


def mmd2_unbiased(x, y, bandwidth):
    """Unbiased estimate of squared MMD between samples x and y (RBF kernel)."""
    m, n = len(x), len(y)
    kxx = rbf_kernel(x, x, bandwidth)
    kyy = rbf_kernel(y, y, bandwidth)
    kxy = rbf_kernel(x, y, bandwidth)
    return float(
        (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
        + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
        - 2.0 * kxy.mean()
    )


def mmd_permutation_test(batch, reference, bandwidth, n_permutations=500, rng=None, reference_kernel=None):
    """
    Unbiased MMD^2 between `batch` and `reference`, with a permutation
    p-value for H0: both samples come from the same distribution.

    Rather than rebuilding the kernel for every permutation, the pooled
    kernel matrix K (n x n) is built once, and each permutation is a 0/1
    column of A (n x P) marking which pooled points play the "batch" role.
    Then for every permutation at once:
        S_xx = sum(A * (K @ A))          within-batch kernel sum
        S_xy = A^T (K 1) - S_xx          batch-vs-rest kernel sum
        S_yy = 1^T K 1 - S_xx - 2 S_xy   within-rest kernel sum
    which turns the whole permutation test into one matrix multiply.

    Returns (mmd2, p_value). With n_permutations=0 only the statistic is
    computed (cheap: no pooled kernel needed) and p_value is None.
    """
    m, r = len(batch), len(reference)
    if reference_kernel is None:
        reference_kernel = rbf_kernel(reference, reference, bandwidth)
    k_bb = rbf_kernel(batch, batch, bandwidth)
    k_br = rbf_kernel(batch, reference, bandwidth)

    # The diagonals of RBF kernels are exactly 1 in theory; using the
    # computed traces keeps the estimator exact even with float round-off.
    observed = (
        (k_bb.sum() - np.trace(k_bb)) / (m * (m - 1))
        + (reference_kernel.sum() - np.trace(reference_kernel)) / (r * (r - 1))
        - 2.0 * k_br.mean()
    )
    if n_permutations <= 0:
        return float(observed), None

    rng = np.random.default_rng() if rng is None else rng
    n = m + r
    pooled = np.empty((n, n))
    pooled[:m, :m] = k_bb
    pooled[:m, m:] = k_br
    pooled[m:, :m] = k_br.T
    pooled[m:, m:] = reference_kernel

    batch_role = rng.random((n_permutations, n)).argsort(axis=1)[:, :m]
    assign = np.zeros((n, n_permutations))
    assign[batch_role, np.arange(n_permutations)[:, None]] = 1.0

    row_sums = pooled.sum(axis=1)
    diag = np.diag(pooled)
    total = row_sums.sum()
    s_xx = (assign * (pooled @ assign)).sum(axis=0)
    s_xy = assign.T @ row_sums - s_xx
    s_yy = total - s_xx - 2.0 * s_xy
    d_x = assign.T @ diag
    d_y = diag.sum() - d_x
    permuted = (s_xx - d_x) / (m * (m - 1)) + (s_yy - d_y) / (r * (r - 1)) - 2.0 * s_xy / (m * r)

    # +1 in numerator and denominator counts the observed split as one of
    # the permutations, which keeps the test valid (p is never exactly 0).
    p_value = (1 + np.sum(permuted >= observed)) / (1 + n_permutations)
    return float(observed), float(p_value)


# --- Mean-shift z-scores -----------------------------------------------------

def mean_shift_z(reference_values, batch_values):
    """
    Absolute z-score of the batch mean against the reference, per column:
        |mean(batch) - mean(ref)| / (std(ref) / sqrt(batch_size))
    i.e. how many standard errors the batch mean sits from where a clean
    batch of this size would be expected to land.
    """
    reference_values = np.atleast_2d(np.asarray(reference_values, dtype=np.float64).T).T
    batch_values = np.atleast_2d(np.asarray(batch_values, dtype=np.float64).T).T
    mu = reference_values.mean(axis=0)
    sd = np.clip(reference_values.std(axis=0, ddof=1), 1e-12, None)
    return np.abs(batch_values.mean(axis=0) - mu) / (sd / np.sqrt(len(batch_values)))


# --- Reference window + per-batch scoring ------------------------------------

class DriftReference:
    """
    Holds the reference window (per-image input stats, embeddings, zero-shot
    probabilities) and scores new batches against it.
    """

    def __init__(self, stats, embeddings, probs, bandwidth=None, seed=0):
        self.input_features = _input_features(stats)
        self.embeddings = l2_normalize(embeddings)
        self.confidence = confidence_stats(probs)
        self.bandwidth = (
            median_heuristic_bandwidth(self.embeddings, rng=np.random.default_rng(seed))
            if bandwidth is None
            else float(bandwidth)
        )
        # Fixed for the life of the reference, so build it once.
        self.kernel = rbf_kernel(self.embeddings, self.embeddings, self.bandwidth)

    def score(self, stats, embeddings, probs, n_permutations=0, rng=None):
        """
        Returns (scores, details):
          scores:  {signal: float} for every name in SIGNALS
          details: per-component values (individual z-scores, raw batch
                   means, MMD p-value) for diagnostics and metrics
        """
        embeddings = l2_normalize(embeddings)
        input_features = _input_features(stats)
        confidence = confidence_stats(probs)

        z_inputs = mean_shift_z(self.input_features, input_features)
        z_conf = mean_shift_z(self.confidence, confidence)
        mmd2, p_value = mmd_permutation_test(
            embeddings, self.embeddings, self.bandwidth,
            n_permutations=n_permutations, rng=rng, reference_kernel=self.kernel,
        )

        scores = {
            # One score for the three input statistics: the largest of their
            # z-scores. The max over three is noisier than any single one,
            # but threshold calibration on clean batches accounts for that.
            "input_stats": float(z_inputs.max()),
            "embedding_cosine": cosine_distance_of_means(self.embeddings, embeddings),
            "embedding_mmd": mmd2,
            "confidence_top_prob": float(z_conf[0]),
            "confidence_entropy": float(z_conf[1]),
        }
        stats = np.asarray(stats, dtype=np.float64)
        details = {
            "z_brightness": float(z_inputs[0]),
            "z_contrast": float(z_inputs[1]),
            "z_sharpness": float(z_inputs[2]),
            "mean_brightness": float(stats[:, 0].mean()),
            "mean_contrast": float(stats[:, 1].mean()),
            "mean_sharpness": float(stats[:, 2].mean()),
            "mean_top_prob": float(confidence[:, 0].mean()),
            "mean_entropy": float(confidence[:, 1].mean()),
            "mmd_pvalue": p_value,
        }
        return scores, details


# --- Thresholds and rates ----------------------------------------------------

def sample_batches(pool_size, batch_size, n_batches, rng):
    """n_batches index arrays, each batch_size distinct indices into a pool."""
    return [rng.choice(pool_size, batch_size, replace=False) for _ in range(n_batches)]


def calibrate_threshold(clean_scores, false_alarm_rate=0.05):
    """
    Alert threshold = the (1 - false_alarm_rate) empirical quantile of
    scores on clean batches; an alert fires when score > threshold.
    method="higher" picks an observed score rather than interpolating,
    which errs on the side of fewer false alarms.
    """
    return float(np.quantile(np.asarray(clean_scores), 1.0 - false_alarm_rate, method="higher"))


def alarm_rate(scores, threshold):
    """Fraction of batches whose score exceeds the threshold."""
    return float(np.mean(np.asarray(scores) > threshold))


def wilson_interval(successes, trials, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (float(centre - half), float(centre + half))


def _rank(x):
    """Ranks starting at 1, ties get their average rank."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for value in np.unique(x):
        tied = x == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x, y):
    """Spearman rank correlation (Pearson correlation of average ranks)."""
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
