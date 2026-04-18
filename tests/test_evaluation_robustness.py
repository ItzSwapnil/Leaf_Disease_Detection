import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.robustness import (
    add_fog,
    add_gaussian_noise,
    add_occlusion,
    adjust_brightness,
    apply_gaussian_blur,
    evaluate_robustness_suite,
)


def _dummy_predict(images: np.ndarray) -> np.ndarray:
    means = images.mean(axis=(1, 2, 3))
    logits = np.stack([means, -means, np.zeros_like(means)], axis=1)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def test_corruption_transforms_preserve_shape_and_bounds():
    images = (
        np.random.default_rng(7)
        .uniform(-1.0, 1.0, size=(6, 32, 32, 3))
        .astype(np.float32)
    )
    rng = np.random.default_rng(11)

    outputs = [
        apply_gaussian_blur(images, sigma=1.0),
        adjust_brightness(images, factor=0.8),
        add_gaussian_noise(images, sigma=0.05, rng=rng),
        add_fog(images, level=0.2),
        add_occlusion(images, frac=0.2, rng=rng),
    ]

    for out in outputs:
        assert out.shape == images.shape
        assert np.max(out) <= 1.0 + 1e-6
        assert np.min(out) >= -1.0 - 1e-6


def test_evaluate_robustness_suite_returns_expected_sections():
    pos = np.full((5, 24, 24, 3), 0.5, dtype=np.float32)
    neg = np.full((5, 24, 24, 3), -0.5, dtype=np.float32)
    images = np.concatenate([pos, neg], axis=0)
    labels = np.array([0] * 5 + [1] * 5, dtype=np.int64)

    report = evaluate_robustness_suite(
        predictor=_dummy_predict,
        images=images,
        labels=labels,
        blur_sigmas=(0.5,),
        brightness_factors=(0.8, 1.2),
        noise_sigmas=(0.03,),
        fog_levels=(0.1,),
        occlusion_fracs=(0.1,),
        seed=17,
    )

    assert report["status"] == "ok"
    assert report["samples"] == 10
    assert "base" in report
    assert len(report["gaussian_blur"]) == 1
    assert len(report["brightness_shift"]) == 2
    assert len(report["gaussian_noise"]) == 1
    assert len(report["fog"]) == 1
    assert len(report["occlusion"]) == 1
    assert "worst_case" in report
