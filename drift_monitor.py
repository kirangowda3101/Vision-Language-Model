"""
Serving-time drift monitor: buffers the images the app classifies, scores
each full window against the reference with exactly the same code as the
offline experiment (drift_metrics.DriftReference), and exports the results
as Prometheus metrics.

Requests arrive one image at a time, but every drift signal is a property
of a *batch*, so the monitor uses tumbling windows: every `window_size`
images it computes one set of scores, updates the gauges, and starts a new
window.

Thresholds are re-calibrated at startup rather than read from the
experiment's JSON, because they depend on (a) the label set the app
classifies against -- the confidence signals are computed over the app's
labels, not the CIFAR-10 classes -- and (b) the window size. The
reference artifact (written by drift_experiment.py) stores the reference
and calibration pools' raw stats and embeddings, which is everything
needed to recompute both.
"""

import os
import threading
import time

import numpy as np
from PIL import Image
from prometheus_client import REGISTRY, Counter, Gauge, start_http_server

import drift_metrics as dm


class DriftMonitor:
    def __init__(
        self,
        reference_path,
        label_features,
        logit_scale,
        window_size=100,
        n_permutations=200,
        false_alarm_rate=0.05,
        n_calibration_batches=500,
        seed=0,
        registry=REGISTRY,
    ):
        artifact = np.load(reference_path)
        self.label_features = dm.l2_normalize(np.asarray(label_features))
        self.logit_scale = float(logit_scale)
        self.window_size = window_size
        self.n_permutations = n_permutations
        self.stats_resolution = tuple(int(v) for v in artifact["stats_resolution"])
        self.rng = np.random.default_rng([seed, 100])

        self.reference = dm.DriftReference(
            artifact["reference_stats"],
            artifact["reference_embeddings"],
            self._probs(artifact["reference_embeddings"]),
            bandwidth=float(artifact["bandwidth"]),
        )

        # Calibrate each signal's threshold on clean calibration windows of
        # the same size the monitor will score. The MMD statistic alone
        # (no permutations) is enough here since thresholds are on the score.
        cal_stats = artifact["calibration_stats"]
        cal_emb = artifact["calibration_embeddings"]
        cal_probs = self._probs(cal_emb)
        cal_scores = {s: [] for s in dm.SIGNALS}
        batches = dm.sample_batches(len(cal_emb), window_size, n_calibration_batches, np.random.default_rng([seed, 101]))
        for idx in batches:
            scores, _ = self.reference.score(cal_stats[idx], cal_emb[idx], cal_probs[idx])
            for s in dm.SIGNALS:
                cal_scores[s].append(scores[s])
        self.thresholds = {s: dm.calibrate_threshold(cal_scores[s], false_alarm_rate) for s in dm.SIGNALS}

        self._lock = threading.Lock()
        self._stats, self._embeddings = [], []

        self.m_score = Gauge("clip_drift_score", "Drift score of the last complete window", ["signal"], registry=registry)
        self.m_threshold = Gauge("clip_drift_threshold", "Calibrated alert threshold", ["signal"], registry=registry)
        self.m_alert = Gauge("clip_drift_alert", "1 if the last window's score exceeded the threshold", ["signal"], registry=registry)
        self.m_component = Gauge("clip_drift_component", "Per-component values of the last window", ["component"], registry=registry)
        self.m_pvalue = Gauge("clip_drift_mmd_pvalue", "Permutation p-value of the last window's MMD test", registry=registry)
        self.m_window_fill = Gauge("clip_drift_window_fill", "Images buffered toward the next window", registry=registry)
        self.m_last_window = Gauge("clip_drift_last_window_timestamp_seconds", "Unix time the last window was scored", registry=registry)
        self.m_windows = Counter("clip_drift_windows", "Windows scored", registry=registry)
        self.m_images = Counter("clip_drift_images", "Images observed by the drift monitor", registry=registry)
        for s, t in self.thresholds.items():
            self.m_threshold.labels(signal=s).set(t)

    def _probs(self, embeddings):
        return dm.zero_shot_probs(embeddings, self.label_features, self.logit_scale)

    def _image_stats(self, image):
        # Reference stats were computed on 32x32 CIFAR images, and sharpness
        # in particular depends heavily on resolution, so uploads are
        # downsampled to the same resolution before computing them.
        image = image.convert("RGB").resize(self.stats_resolution, Image.BILINEAR)
        return dm.image_stats(np.asarray(image))[0]

    def observe(self, image, embedding):
        """
        Record one served request. `image` is the PIL image as received,
        `embedding` the (256,) image embedding the model produced for it.
        Returns the window's (scores, details) when this request completed
        a window, otherwise None.
        """
        stats = self._image_stats(image)
        with self._lock:
            self.m_images.inc()
            self._stats.append(stats)
            self._embeddings.append(np.asarray(embedding, dtype=np.float64))
            if len(self._stats) < self.window_size:
                self.m_window_fill.set(len(self._stats))
                return None
            stats_arr, emb_arr = np.stack(self._stats), np.stack(self._embeddings)
            self._stats, self._embeddings = [], []
            self.m_window_fill.set(0)
            # Scored under the lock: numpy Generators aren't thread-safe, and
            # this only runs once per window.
            return self._score_window(stats_arr, emb_arr)

    def _score_window(self, stats, embeddings):
        scores, details = self.reference.score(
            stats, embeddings, self._probs(embeddings), n_permutations=self.n_permutations, rng=self.rng
        )
        for s, value in scores.items():
            self.m_score.labels(signal=s).set(value)
            self.m_alert.labels(signal=s).set(float(value > self.thresholds[s]))
        for name, value in details.items():
            if name != "mmd_pvalue" and value is not None:
                self.m_component.labels(component=name).set(value)
        if details["mmd_pvalue"] is not None:
            self.m_pvalue.set(details["mmd_pvalue"])
        self.m_last_window.set(time.time())
        self.m_windows.inc()
        return scores, details


def load_drift_monitor(label_features, logit_scale, reference_path=None, metrics_port=None):
    """
    App-facing helper: builds the monitor and starts the /metrics HTTP
    server, or returns None (with a message) if the reference artifact
    hasn't been generated, so the demo still runs without monitoring.
    """
    reference_path = reference_path or os.environ.get("DRIFT_REFERENCE", "drift_reference.npz")
    if not os.path.exists(reference_path):
        print(f"Drift monitoring disabled: {reference_path} not found (run drift_experiment.py to create it).")
        return None
    monitor = DriftMonitor(
        reference_path,
        label_features,
        logit_scale,
        window_size=int(os.environ.get("DRIFT_WINDOW_SIZE", 100)),
    )
    port = int(metrics_port or os.environ.get("METRICS_PORT", 8000))
    start_http_server(port)
    print(f"Drift monitor ready (window={monitor.window_size}); Prometheus metrics on :{port}/metrics")
    return monitor
