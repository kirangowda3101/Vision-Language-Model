import numpy as np
import pytest

import drift_metrics as dm


def naive_mmd2_unbiased(x, y, bandwidth):
    """Direct double-loop definition, to check the vectorized versions against."""
    def k(a, b):
        return np.exp(-np.sum((a - b) ** 2) / (2 * bandwidth**2))

    m, n = len(x), len(y)
    xx = sum(k(x[i], x[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    yy = sum(k(y[i], y[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    xy = sum(k(x[i], y[j]) for i in range(m) for j in range(n)) / (m * n)
    return xx + yy - 2 * xy


# --- input statistics ---------------------------------------------------------

def test_image_stats_constant_image():
    img = np.full((8, 8, 3), 128, dtype=np.uint8)
    brightness, contrast, sharpness = dm.image_stats(img)[0]
    assert brightness == pytest.approx(128 / 255)
    assert contrast == pytest.approx(0.0)
    assert sharpness == pytest.approx(0.0)


def test_image_stats_known_values():
    # Checkerboard of 0/255: grayscale mean 0.5, std 0.5, and every interior
    # Laplacian is +-4, so its variance is 16.
    board = (np.indices((6, 6)).sum(0) % 2 * 255).astype(np.uint8)
    img = np.repeat(board[..., None], 3, axis=2)
    brightness, contrast, sharpness = dm.image_stats(img)[0]
    assert brightness == pytest.approx(0.5)
    assert contrast == pytest.approx(0.5)
    assert sharpness == pytest.approx(16.0)


def test_image_stats_batch_shape_and_luma():
    imgs = np.zeros((4, 5, 5, 3), dtype=np.uint8)
    imgs[1, ..., 0] = 255  # pure red
    stats = dm.image_stats(imgs)
    assert stats.shape == (4, 3)
    assert stats[1, 0] == pytest.approx(0.299)


# --- confidence ------------------------------------------------------------

def test_confidence_stats_uniform_and_onehot():
    uniform = np.full((1, 10), 0.1)
    onehot = np.eye(10)[:1]
    top, ent = dm.confidence_stats(uniform)[0]
    assert top == pytest.approx(0.1)
    assert ent == pytest.approx(np.log(10))
    top, ent = dm.confidence_stats(onehot)[0]
    assert top == pytest.approx(1.0)
    assert ent == pytest.approx(0.0, abs=1e-9)


def test_zero_shot_probs_matches_manual_softmax():
    rng = np.random.default_rng(0)
    img, lab = rng.normal(size=(3, 8)), rng.normal(size=(4, 8))
    probs = dm.zero_shot_probs(img, lab, logit_scale=20.0)
    img_n = img / np.linalg.norm(img, axis=1, keepdims=True)
    lab_n = lab / np.linalg.norm(lab, axis=1, keepdims=True)
    logits = 20.0 * img_n @ lab_n.T
    expected = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    np.testing.assert_allclose(probs, expected, rtol=1e-10)
    np.testing.assert_allclose(probs.sum(1), 1.0)


# --- embedding drift --------------------------------------------------------

def test_cosine_distance_of_means():
    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert dm.cosine_distance_of_means(a, a) == pytest.approx(0.0)
    assert dm.cosine_distance_of_means(a, b) == pytest.approx(1.0)
    assert dm.cosine_distance_of_means(a, -a) == pytest.approx(2.0)


def test_rbf_kernel_values():
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    k = dm.rbf_kernel(x, x, bandwidth=5.0)
    np.testing.assert_allclose(np.diag(k), 1.0)
    assert k[0, 1] == pytest.approx(np.exp(-25 / 50))


def test_median_heuristic_bandwidth():
    # Three collinear points at 0, 1, 3: pairwise distances 1, 2, 3 -> median 2.
    x = np.array([[0.0], [1.0], [3.0]])
    assert dm.median_heuristic_bandwidth(x) == pytest.approx(2.0)


def test_mmd2_matches_naive():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(7, 3)), rng.normal(0.5, 1.0, size=(9, 3))
    assert dm.mmd2_unbiased(x, y, 1.3) == pytest.approx(naive_mmd2_unbiased(x, y, 1.3))
    stat, p = dm.mmd_permutation_test(x, y, 1.3, n_permutations=0)
    assert stat == pytest.approx(naive_mmd2_unbiased(x, y, 1.3))
    assert p is None


def test_mmd_permutation_statistics_match_naive():
    # The matrix-multiply shortcut must give the same statistic as
    # recomputing MMD from scratch on each permuted split.
    rng = np.random.default_rng(2)
    batch, ref = rng.normal(size=(6, 2)), rng.normal(size=(10, 2))
    bw = 1.0
    stat, p = dm.mmd_permutation_test(batch, ref, bw, n_permutations=50, rng=np.random.default_rng(3))

    pooled = np.concatenate([batch, ref])
    perm_rng = np.random.default_rng(3)
    order = perm_rng.random((50, len(pooled))).argsort(axis=1)
    expected = []
    for row in order:
        mask = np.zeros(len(pooled), bool)
        mask[row[: len(batch)]] = True
        expected.append(naive_mmd2_unbiased(pooled[mask], pooled[~mask], bw))
    expected_p = (1 + np.sum(np.array(expected) >= stat)) / 51
    assert p == pytest.approx(expected_p)


def test_mmd_detects_shift_and_is_calibrated_under_null():
    rng = np.random.default_rng(4)
    ref = rng.normal(size=(200, 5))
    bw = dm.median_heuristic_bandwidth(ref)
    ref_kernel = dm.rbf_kernel(ref, ref, bw)

    _, p_shift = dm.mmd_permutation_test(
        rng.normal(0.7, 1.0, size=(50, 5)), ref, bw, n_permutations=200, rng=rng, reference_kernel=ref_kernel
    )
    assert p_shift < 0.01

    # Under H0 the permutation p-value is (super-)uniform: about 5% of
    # same-distribution batches should have p < 0.05. 200 trials -> the
    # count is Binomial(200, 0.05) under exact validity (mean 10); a
    # generous band avoids flakiness while catching a broken test.
    rejections = sum(
        dm.mmd_permutation_test(
            rng.normal(size=(50, 5)), ref, bw, n_permutations=99, rng=rng, reference_kernel=ref_kernel
        )[1] < 0.05
        for _ in range(200)
    )
    assert 2 <= rejections <= 22


def test_mmd_pvalue_bounds():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(5, 2))
    _, p = dm.mmd_permutation_test(x, x + 100.0, 1.0, n_permutations=19, rng=rng)
    assert 1 / 20 <= p <= 1.0


# --- z-scores, thresholds, rates --------------------------------------------

def test_mean_shift_z():
    ref = np.array([[0.0], [2.0], [4.0]])  # mean 2, sample std 2
    batch = np.array([[5.0], [5.0], [5.0], [5.0]])  # mean 5, n=4 -> se = 1
    np.testing.assert_allclose(dm.mean_shift_z(ref, batch), [3.0])
    # 1-D input is treated as one column.
    np.testing.assert_allclose(dm.mean_shift_z(ref[:, 0], batch[:, 0]), [3.0])


def test_calibrate_threshold_and_alarm_rate():
    scores = np.arange(1, 101, dtype=float)  # 1..100
    threshold = dm.calibrate_threshold(scores, 0.05)
    assert threshold == 96.0
    assert dm.alarm_rate(scores, threshold) == pytest.approx(0.04)
    assert dm.alarm_rate([0.0, 1.0], 0.5) == 0.5


def test_wilson_interval():
    lo, hi = dm.wilson_interval(25, 500)
    assert lo < 0.05 < hi
    assert 0.0 <= dm.wilson_interval(0, 50)[0] < 1e-12


def test_spearman():
    assert dm.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert dm.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Ties use average ranks: x ranks [1.5, 1.5, 3], y ranks [1, 2, 3].
    assert dm.spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(np.corrcoef([1.5, 1.5, 3], [1, 2, 3])[0, 1])
    assert np.isnan(dm.spearman([1, 1, 1], [1, 2, 3]))


def test_sample_batches_are_distinct_within_batch():
    batches = dm.sample_batches(50, 20, 10, np.random.default_rng(0))
    assert len(batches) == 10
    for b in batches:
        assert len(set(b.tolist())) == 20 and b.max() < 50


# --- DriftReference end to end ----------------------------------------------

def _fake_window(rng, n, shift=0.0):
    images = rng.integers(0, 256, size=(n, 8, 8, 3), dtype=np.uint8)
    emb = rng.normal(size=(n, 16)) + shift
    probs = dm.softmax(rng.normal(size=(n, 10)) + shift * np.arange(10))
    return dm.image_stats(images), emb, probs


def test_drift_reference_scores_clean_low_and_shifted_high():
    rng = np.random.default_rng(6)
    ref = dm.DriftReference(*_fake_window(rng, 300))
    clean_scores, details = ref.score(*_fake_window(rng, 100), n_permutations=99, rng=rng)
    shifted_scores, _ = ref.score(*_fake_window(rng, 100, shift=1.0), n_permutations=99, rng=rng)

    assert set(clean_scores) == set(dm.SIGNALS)
    assert 0.0 <= details["mmd_pvalue"] <= 1.0
    for s in ("embedding_cosine", "embedding_mmd", "confidence_top_prob", "confidence_entropy"):
        assert shifted_scores[s] > clean_scores[s]


def test_drift_reference_is_deterministic_given_seed():
    rng = np.random.default_rng(7)
    window, batch = _fake_window(rng, 200), _fake_window(rng, 50)
    a = dm.DriftReference(*window).score(*batch, n_permutations=49, rng=np.random.default_rng(0))
    b = dm.DriftReference(*window).score(*batch, n_permutations=49, rng=np.random.default_rng(0))
    assert a == b
