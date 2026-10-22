import numpy as np
import pytest
from PIL import Image

pytest.importorskip("prometheus_client")
from prometheus_client import CollectorRegistry  # noqa: E402

import drift_metrics as dm  # noqa: E402
from drift_monitor import DriftMonitor  # noqa: E402


@pytest.fixture
def artifact(tmp_path):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(600, 32, 32, 3), dtype=np.uint8)
    # Real CLIP embeddings share a common direction (their mean is far from
    # zero), which is what makes "cosine distance between means" meaningful;
    # mimic that with a constant offset along the first axis.
    emb = (rng.normal(size=(600, 16)) + 2.0 * np.eye(16)[0]).astype(np.float32)
    path = tmp_path / "ref.npz"
    np.savez(
        path,
        reference_stats=dm.image_stats(images[:300]),
        reference_embeddings=emb[:300],
        calibration_stats=dm.image_stats(images[300:]),
        calibration_embeddings=emb[300:],
        stats_resolution=np.array([32, 32]),
        bandwidth=np.array(dm.median_heuristic_bandwidth(emb[:300].astype(np.float64))),
        seed=np.array(0),
    )
    return path


def make_monitor(artifact, registry):
    labels = np.random.default_rng(1).normal(size=(5, 16))
    return DriftMonitor(
        artifact, labels, logit_scale=20.0, window_size=20,
        n_permutations=49, n_calibration_batches=100, registry=registry,
    )


def test_monitor_scores_each_full_window_and_exports_metrics(artifact):
    registry = CollectorRegistry()
    monitor = make_monitor(artifact, registry)
    rng = np.random.default_rng(2)

    results = []
    for _ in range(40):
        img = Image.fromarray(rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8))
        results.append(monitor.observe(img, rng.normal(size=16) + 2.0 * np.eye(16)[0]))

    # A result exactly at the end of each 20-image window, None otherwise.
    assert [i for i, r in enumerate(results) if r is not None] == [19, 39]
    assert registry.get_sample_value("clip_drift_windows_total") == 2
    assert registry.get_sample_value("clip_drift_images_total") == 40
    assert registry.get_sample_value("clip_drift_window_fill") == 0
    for s in dm.SIGNALS:
        score = registry.get_sample_value("clip_drift_score", {"signal": s})
        threshold = registry.get_sample_value("clip_drift_threshold", {"signal": s})
        alert = registry.get_sample_value("clip_drift_alert", {"signal": s})
        assert score == pytest.approx(results[-1][0][s])
        assert threshold == pytest.approx(monitor.thresholds[s])
        assert alert == float(score > threshold)
    assert 0 < registry.get_sample_value("clip_drift_mmd_pvalue") <= 1


def test_monitor_alerts_on_shifted_embeddings(artifact):
    registry = CollectorRegistry()
    monitor = make_monitor(artifact, registry)
    rng = np.random.default_rng(3)
    for _ in range(20):
        img = Image.fromarray(rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8))
        monitor.observe(img, rng.normal(size=16) + 2.0 * np.eye(16)[1])
    assert registry.get_sample_value("clip_drift_alert", {"signal": "embedding_mmd"}) == 1.0
    assert registry.get_sample_value("clip_drift_alert", {"signal": "embedding_cosine"}) == 1.0
