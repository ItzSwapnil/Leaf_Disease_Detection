from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image


def assess_leaf_likelihood(img_path: str, img_size: int) -> dict[str, Any]:
    """Heuristic leaf plausibility check to reject obvious non-leaf uploads."""
    try:
        with Image.open(img_path) as img:
            arr = np.asarray(
                img.convert("RGB").resize((img_size, img_size)), dtype=np.float32
            )

        if arr.size == 0:
            return {
                "leaf_score": 0.0,
                "vegetation_ratio": 0.0,
                "reason": "Image content could not be analyzed.",
            }

        arr_norm = arr / 255.0
        maxc = np.max(arr_norm, axis=2)
        minc = np.min(arr_norm, axis=2)
        delta = np.maximum(maxc - minc, 1e-8)

        hue = np.zeros_like(maxc)
        r = arr_norm[:, :, 0]
        g = arr_norm[:, :, 1]
        b = arr_norm[:, :, 2]

        r_mask = maxc == r
        g_mask = maxc == g
        b_mask = maxc == b

        hue[r_mask] = (60.0 * ((g[r_mask] - b[r_mask]) / delta[r_mask]) + 360.0) % 360.0
        hue[g_mask] = (60.0 * ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 120.0) % 360.0
        hue[b_mask] = (60.0 * ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 240.0) % 360.0

        sat = np.where(maxc <= 0.0, 0.0, delta / maxc)
        val = maxc

        vegetation_mask = (hue >= 20.0) & (hue <= 140.0) & (sat >= 0.15) & (val >= 0.15)
        vegetation_ratio = float(np.mean(vegetation_mask))
        contrast = float(np.std(arr_norm))

        leaf_score = min(1.0, vegetation_ratio * 1.7 + contrast * 0.6)

        reason = ""
        if vegetation_ratio < 0.08:
            reason = "Very little leaf-like color/texture was detected."
        elif leaf_score < 0.35:
            reason = "Leaf signal is weak in this image."

        return {
            "leaf_score": round(leaf_score, 3),
            "vegetation_ratio": round(vegetation_ratio, 3),
            "reason": reason,
        }
    except Exception as exc:
        return {
            "leaf_score": 0.0,
            "vegetation_ratio": 0.0,
            "reason": f"Image validation failed: {exc}",
        }


def compute_prediction_diagnostics(
    probs: np.ndarray,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Compute top-class confidence, margin, and entropy diagnostics."""
    scores = np.asarray(probs, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Prediction probabilities cannot be empty.")

    if not np.all(np.isfinite(scores)):
        raise ValueError("Prediction probabilities must be finite.")

    total = float(np.sum(scores))
    if total <= 0.0:
        raise ValueError("Prediction probabilities must sum to a positive value.")

    scores = scores / total

    order = np.argsort(scores)
    top1_idx = int(order[-1])
    top2_idx = int(order[-2]) if scores.size > 1 else top1_idx

    top1_prob = float(scores[top1_idx])
    top2_prob = float(scores[top2_idx]) if scores.size > 1 else 0.0
    confidence_margin = max(0.0, top1_prob - top2_prob)

    clipped = np.clip(scores, eps, 1.0)
    entropy_bits = float(-np.sum(clipped * np.log2(clipped)))
    entropy_max_bits = float(math.log2(max(2, scores.size)))
    entropy_ratio = float(entropy_bits / entropy_max_bits)

    return {
        "top1_index": top1_idx,
        "top2_index": top2_idx,
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "confidence_margin": confidence_margin,
        "entropy_bits": entropy_bits,
        "entropy_ratio": entropy_ratio,
    }


def evaluate_inference_safety(
    diagnostics: dict[str, Any],
    leaf_validation: dict[str, Any],
    confidence_threshold: float,
    entropy_threshold_bits: float,
    msp_threshold: float,
    min_margin: float = 0.12,
    non_leaf_leaf_score: float = 0.23,
    weak_leaf_score: float = 0.35,
    min_vegetation_ratio: float = 0.08,
) -> dict[str, Any]:
    """Decide whether to reject an inference result as low-trust/OOD."""
    top1_prob = float(diagnostics.get("top1_prob", 0.0))
    margin = float(diagnostics.get("confidence_margin", 0.0))
    entropy_bits = float(diagnostics.get("entropy_bits", 0.0))
    entropy_ratio = float(diagnostics.get("entropy_ratio", 0.0))

    leaf_score = float(leaf_validation.get("leaf_score", 0.0))
    vegetation_ratio = float(leaf_validation.get("vegetation_ratio", 0.0))

    appears_non_leaf = leaf_score < float(
        non_leaf_leaf_score
    ) and vegetation_ratio < float(min_vegetation_ratio)
    weak_leaf_signal = leaf_score < float(weak_leaf_score)

    low_confidence = top1_prob < float(confidence_threshold)
    low_msp = top1_prob < float(msp_threshold)
    entropy_threshold_value = float(entropy_threshold_bits)
    # Backward-compatible threshold mode:
    # <= 1.0 means normalized entropy ratio, otherwise entropy in bits.
    if entropy_threshold_value <= 1.0:
        high_entropy = entropy_ratio > entropy_threshold_value
    else:
        high_entropy = entropy_bits > entropy_threshold_value
    low_margin = margin < float(min_margin)

    uncertainty_flags = {
        "low_confidence": low_confidence,
        "low_msp": low_msp,
        "high_entropy": high_entropy,
        "low_margin": low_margin,
    }
    uncertainty_score = int(sum(1 for flag in uncertainty_flags.values() if flag))

    reject = bool(
        appears_non_leaf
        or (weak_leaf_signal and uncertainty_score >= 1)
        or uncertainty_score >= 2
    )

    reasons: list[str] = []
    if appears_non_leaf:
        reasons.append("image_appears_non_leaf")
    if weak_leaf_signal:
        reasons.append("weak_leaf_signal")
    if low_confidence:
        reasons.append("low_confidence")
    if low_msp:
        reasons.append("low_msp")
    if high_entropy:
        reasons.append("high_entropy")
    if low_margin:
        reasons.append("low_margin")

    return {
        "reject": reject,
        "reasons": reasons,
        "appears_non_leaf": appears_non_leaf,
        "weak_leaf_signal": weak_leaf_signal,
        "uncertainty_score": uncertainty_score,
        "flags": uncertainty_flags,
    }
