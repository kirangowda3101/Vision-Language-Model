"""
CIFAR-10-C-style synthetic corruptions, applied to native 32x32 uint8
images (the resolution CIFAR-10-C itself corrupts at), before the usual
resize-to-224 preprocessing.

Severity parameters for Gaussian noise, Gaussian blur and JPEG follow the
CIFAR-10-C generation code (Hendrycks & Dietterich, 2019). CIFAR-10-C's
"brightness" corruption *increases* brightness; this project wants the
opposite (dark / under-exposed inputs), so brightness_reduction subtracts
the same HSV value offsets that CIFAR-10-C adds.

Every function takes and returns a (32, 32, 3) uint8 array. Gaussian noise
draws from an explicitly passed numpy Generator so results are
reproducible.
"""

import io

import numpy as np
from PIL import Image

SEVERITIES = (1, 2, 3, 4, 5)


def brightness_reduction(image, value_offset, rng=None):
    """Lower the HSV value channel by `value_offset` (in [0, 1] units)."""
    h, s, v = Image.fromarray(image).convert("HSV").split()
    v = np.asarray(v, dtype=np.float64) / 255.0
    v = np.clip(v - value_offset, 0.0, 1.0)
    v = Image.fromarray(np.round(v * 255.0).astype(np.uint8))
    return np.asarray(Image.merge("HSV", (h, s, v)).convert("RGB"))


def _gaussian_kernel_1d(sigma, truncate=4.0):
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(image, sigma, rng=None):
    """Separable Gaussian blur with edge-replicate padding (skimage's default)."""
    kernel = _gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2
    x = image.astype(np.float64)
    padded = np.pad(x, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    x = sum(w * padded[i : i + x.shape[0]] for i, w in enumerate(kernel))
    padded = np.pad(x, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    x = sum(w * padded[:, i : i + x.shape[1]] for i, w in enumerate(kernel))
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def gaussian_noise(image, std, rng):
    """Additive i.i.d. Gaussian noise with standard deviation `std` in [0, 1] units."""
    x = image.astype(np.float64) / 255.0
    x = np.clip(x + rng.normal(0.0, std, size=x.shape), 0.0, 1.0)
    return np.round(x * 255.0).astype(np.uint8)


def jpeg_compression(image, quality, rng=None):
    """Round-trip through PIL's JPEG encoder at the given quality."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))


# name -> (function, parameter for severities 1..5)
CORRUPTIONS = {
    "brightness_reduction": (brightness_reduction, (0.05, 0.1, 0.15, 0.2, 0.3)),
    "gaussian_blur": (gaussian_blur, (0.4, 0.6, 0.7, 0.8, 1.0)),
    "gaussian_noise": (gaussian_noise, (0.04, 0.06, 0.08, 0.09, 0.10)),
    "jpeg_compression": (jpeg_compression, (80, 65, 58, 50, 40)),
}


def corrupt(image, name, severity, rng=None):
    """Apply corruption `name` at `severity` (1-5) to one uint8 image."""
    fn, params = CORRUPTIONS[name]
    return fn(image, params[severity - 1], rng=rng)


def corrupt_batch(images, name, severity, seed, indices=None):
    """
    Corrupt a stack of images. Each image's noise comes from its own
    Generator seeded by (seed, corruption, severity, image index), so a
    given image always receives the same noise regardless of batch order
    or which subset is being processed.
    """
    indices = range(len(images)) if indices is None else indices
    corruption_id = list(CORRUPTIONS).index(name)
    out = np.empty_like(images)
    for i, (image, idx) in enumerate(zip(images, indices)):
        rng = np.random.default_rng([seed, corruption_id, severity, int(idx)])
        out[i] = corrupt(image, name, severity, rng=rng)
    return out
