import numpy as np
import pytest

import drift_metrics as dm
from corruptions import CORRUPTIONS, SEVERITIES, corrupt, corrupt_batch


@pytest.fixture
def images():
    # Smooth-ish random images (structure for blur/JPEG to act on).
    rng = np.random.default_rng(0)
    base = rng.integers(40, 216, size=(16, 8, 8, 3)).astype(np.float64)
    upsampled = np.kron(base, np.ones((1, 4, 4, 1)))  # (16, 32, 32, 3)
    return np.clip(upsampled + rng.normal(0, 8, upsampled.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", list(CORRUPTIONS))
@pytest.mark.parametrize("severity", SEVERITIES)
def test_shape_and_dtype_preserved(images, name, severity):
    out = corrupt(images[0], name, severity, rng=np.random.default_rng(0))
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_brightness_reduction_darkens_monotonically(images):
    means = [dm.image_stats(corrupt_batch(images, "brightness_reduction", s, seed=0))[:, 0].mean() for s in SEVERITIES]
    assert means[0] < dm.image_stats(images)[:, 0].mean()
    assert all(a > b for a, b in zip(means, means[1:]))


def test_blur_reduces_sharpness_monotonically(images):
    sharp = [dm.image_stats(corrupt_batch(images, "gaussian_blur", s, seed=0))[:, 2].mean() for s in SEVERITIES]
    assert sharp[0] < dm.image_stats(images)[:, 2].mean()
    assert all(a > b for a, b in zip(sharp, sharp[1:]))


def test_noise_increases_sharpness(images):
    clean = dm.image_stats(images)[:, 2].mean()
    noisy = dm.image_stats(corrupt_batch(images, "gaussian_noise", 5, seed=0))[:, 2].mean()
    assert noisy > clean


def test_jpeg_changes_image(images):
    out = corrupt_batch(images, "jpeg_compression", 5, seed=0)
    assert np.abs(out.astype(int) - images.astype(int)).mean() > 0


def test_corrupt_batch_is_reproducible_and_order_independent(images):
    idx = np.arange(100, 116)
    a = corrupt_batch(images, "gaussian_noise", 3, seed=0, indices=idx)
    b = corrupt_batch(images, "gaussian_noise", 3, seed=0, indices=idx)
    np.testing.assert_array_equal(a, b)
    # Same image + same index gives the same noise even in a different batch order.
    rev = corrupt_batch(images[::-1], "gaussian_noise", 3, seed=0, indices=idx[::-1])
    np.testing.assert_array_equal(a, rev[::-1])
    # A different seed gives different noise.
    c = corrupt_batch(images, "gaussian_noise", 3, seed=1, indices=idx)
    assert not np.array_equal(a, c)
