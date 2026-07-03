"""Robustness evaluation utilities using pure PyTorch and NumPy.

Provides synthetic corruption functions (blur, brightness, noise, fog,
occlusion) and a suite evaluator that measures accuracy degradation under
each perturbation.

All image operations assume input in the normalized range ``[-1, 1]``
(standard ImageNet preprocessing output).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _clip_preprocessed(images: np.ndarray) -> np.ndarray:
    """Clip images to the ``[-1, 1]`` preprocessed range."""
    return np.clip(images, -1.0, 1.0).astype(np.float32)


def _to_unit_range(images: np.ndarray) -> np.ndarray:
    """Map ``[-1, 1]`` preprocessed images to ``[0, 1]`` unit range."""
    return np.clip((images + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)


def _from_unit_range(images: np.ndarray) -> np.ndarray:
    """Map ``[0, 1]`` unit range images back to ``[-1, 1]``."""
    return np.clip(images * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)


def apply_gaussian_blur(images: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to a batch of preprocessed images in ``[-1, 1]``.

    Constructs a separable 2D Gaussian kernel with the given sigma and applies
    it via ``F.conv2d`` with depthwise grouping.

    Args:
        images: Batch of images with shape ``[N, H, W, C]`` in ``[-1, 1]``.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Blurred images with the same shape and range.
    """
    sigma = float(sigma)
    if sigma <= 0.0:
        return images.astype(np.float32, copy=True)

    radius = max(1, int(np.ceil(3.0 * sigma)))
    # Build 1D Gaussian kernel
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-0.5 * torch.square(coords / sigma))
    kernel_1d = kernel_1d / torch.sum(kernel_1d)

    # Outer product → 2D kernel
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]

    channels = int(images.shape[-1])
    # Repeat for each channel (depthwise convolution)
    kernel = kernel_2d.repeat(channels, 1, 1, 1)  # [C, 1, K, K]

    # Convert NHWC → NCHW for PyTorch conv2d
    x = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    blurred = F.conv2d(x, kernel, padding=radius, groups=channels)

    # Convert back to NHWC numpy
    result = blurred.permute(0, 2, 3, 1).numpy()
    return _clip_preprocessed(result)


def adjust_brightness(images: np.ndarray, factor: float) -> np.ndarray:
    """Scale brightness in unit space, then map back to ``[-1, 1]``."""
    unit = _to_unit_range(images)
    adjusted = np.clip(unit * float(factor), 0.0, 1.0)
    return _from_unit_range(adjusted)


def add_gaussian_noise(
    images: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Add Gaussian noise in unit space, then map back to ``[-1, 1]``."""
    sigma = float(sigma)
    if sigma <= 0.0:
        return images.astype(np.float32, copy=True)
    unit = _to_unit_range(images)
    noisy = np.clip(unit + rng.normal(0.0, sigma, size=unit.shape), 0.0, 1.0)
    return _from_unit_range(noisy)


def add_fog(images: np.ndarray, level: float) -> np.ndarray:
    """Simulate haze/fog by blending toward white in unit space."""
    level = float(np.clip(level, 0.0, 1.0))
    if level <= 0.0:
        return images.astype(np.float32, copy=True)
    unit = _to_unit_range(images)
    fogged = np.clip(unit * (1.0 - level) + level, 0.0, 1.0)
    return _from_unit_range(fogged)


def add_occlusion(
    images: np.ndarray, frac: float, rng: np.random.Generator
) -> np.ndarray:
    """Add random square occlusion patches in unit space."""
    frac = float(np.clip(frac, 0.0, 0.9))
    if frac <= 0.0:
        return images.astype(np.float32, copy=True)

    unit = _to_unit_range(images)
    n, h, w, _ = unit.shape
    patch_area = int(max(1, round(frac * h * w)))
    patch_size = int(max(1, round(np.sqrt(patch_area))))

    occluded = unit.copy()
    for idx in range(n):
        ph = min(h, patch_size)
        pw = min(w, patch_size)
        top = int(rng.integers(0, max(1, h - ph + 1)))
        left = int(rng.integers(0, max(1, w - pw + 1)))
        occluded[idx, top: top + ph, left: left + pw, :] = 1.0

    return _from_unit_range(occluded)


def _metrics_from_probs(
    labels: np.ndarray, probs: np.ndarray
) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1 from probability predictions."""
    preds = np.argmax(probs, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def _with_drop(
    base: dict[str, float], metrics: dict[str, float]
) -> dict[str, float]:
    """Augment metrics dict with accuracy and F1 drop relative to baseline."""
    return {
        **metrics,
        "accuracy_drop": float(base["accuracy"] - metrics["accuracy"]),
        "macro_f1_drop": float(base["macro_f1"] - metrics["macro_f1"]),
    }


def evaluate_robustness_suite(
    predictor: Callable[[np.ndarray], np.ndarray],
    images: np.ndarray,
    labels: np.ndarray,
    blur_sigmas: Sequence[float],
    brightness_factors: Sequence[float],
    noise_sigmas: Sequence[float],
    fog_levels: Sequence[float],
    occlusion_fracs: Sequence[float],
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate degradation under synthetic adverse conditions.

    Args:
        predictor: Callable that accepts ``[N, H, W, C]`` images and returns
            ``[N, C]`` probability predictions.
        images: Clean images with shape ``[N, H, W, C]`` in ``[-1, 1]``.
        labels: Integer class labels of shape ``[N]``.
        blur_sigmas: Gaussian blur sigma values to test.
        brightness_factors: Brightness scaling factors to test.
        noise_sigmas: Gaussian noise sigma values to test.
        fog_levels: Fog intensity levels to test.
        occlusion_fracs: Occlusion fraction values to test.
        seed: Random seed for reproducibility.

    Returns:
        Comprehensive robustness report dictionary.
    """
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if images.ndim != 4:
        raise ValueError("images must have shape [N, H, W, C].")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("labels must be shape [N] and match image count.")

    rng = np.random.default_rng(int(seed))

    base_probs = np.asarray(predictor(images), dtype=np.float64)
    base_metrics = _metrics_from_probs(labels, base_probs)

    report: dict[str, Any] = {
        "status": "ok",
        "samples": int(images.shape[0]),
        "base": base_metrics,
        "gaussian_blur": [],
        "brightness_shift": [],
        "gaussian_noise": [],
        "fog": [],
        "occlusion": [],
    }

    for sigma in blur_sigmas:
        corrupted = apply_gaussian_blur(images, sigma=sigma)
        metrics = _metrics_from_probs(
            labels, np.asarray(predictor(corrupted), dtype=np.float64)
        )
        report["gaussian_blur"].append(
            {"sigma": float(sigma), **_with_drop(base_metrics, metrics)}
        )

    for factor in brightness_factors:
        corrupted = adjust_brightness(images, factor=factor)
        metrics = _metrics_from_probs(
            labels, np.asarray(predictor(corrupted), dtype=np.float64)
        )
        report["brightness_shift"].append(
            {"factor": float(factor), **_with_drop(base_metrics, metrics)}
        )

    for sigma in noise_sigmas:
        corrupted = add_gaussian_noise(images, sigma=sigma, rng=rng)
        metrics = _metrics_from_probs(
            labels, np.asarray(predictor(corrupted), dtype=np.float64)
        )
        report["gaussian_noise"].append(
            {"sigma": float(sigma), **_with_drop(base_metrics, metrics)}
        )

    for level in fog_levels:
        corrupted = add_fog(images, level=level)
        metrics = _metrics_from_probs(
            labels, np.asarray(predictor(corrupted), dtype=np.float64)
        )
        report["fog"].append(
            {"level": float(level), **_with_drop(base_metrics, metrics)}
        )

    for frac in occlusion_fracs:
        corrupted = add_occlusion(images, frac=frac, rng=rng)
        metrics = _metrics_from_probs(
            labels, np.asarray(predictor(corrupted), dtype=np.float64)
        )
        report["occlusion"].append(
            {"fraction": float(frac), **_with_drop(base_metrics, metrics)}
        )

    all_rows: list[dict] = []
    for key in (
        "gaussian_blur",
        "brightness_shift",
        "gaussian_noise",
        "fog",
        "occlusion",
    ):
        all_rows.extend(report[key])
    if all_rows:
        worst = max(all_rows, key=lambda item: item["accuracy_drop"])
        report["worst_case"] = {
            "accuracy_drop": float(worst["accuracy_drop"]),
            "macro_f1_drop": float(worst["macro_f1_drop"]),
        }

    return report
