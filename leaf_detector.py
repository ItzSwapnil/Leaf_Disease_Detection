"""Binary leaf detector: is there a valid leaf in the image?"""

from __future__ import annotations

import numpy as np
from PIL import Image


def detect_leaf_presence(
    img_path: str,
    img_size: int = 224,
    vegetation_ratio_threshold: float = 0.05,
    contrast_threshold: float = 0.05,
) -> dict:
    """
    Detect if an image contains a meaningful leaf structure.

    Returns:
        dict with keys:
        - 'is_leaf': bool, whether a leaf is detected
        - 'vegetation_ratio': float 0..1, fraction of leaf-like colors
        - 'contrast': float, standard deviation of normalized image
        - 'leaf_score': float 0..1, combined confidence score
        - 'reason': str, explanation if not a leaf
    """
    try:
        with Image.open(img_path) as img:
            arr = np.asarray(
                img.convert("RGB").resize((img_size, img_size)),
                dtype=np.float32,
            )

        if arr.size == 0:
            return {
                "is_leaf": False,
                "vegetation_ratio": 0.0,
                "contrast": 0.0,
                "leaf_score": 0.0,
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

        hue[r_mask] = (
            60.0 * ((g[r_mask] - b[r_mask]) / delta[r_mask]) + 360.0
        ) % 360.0
        hue[g_mask] = (
            60.0 * ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 120.0
        ) % 360.0
        hue[b_mask] = (
            60.0 * ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 240.0
        ) % 360.0

        sat = np.divide(
            delta,
            maxc,
            out=np.zeros_like(delta),
            where=maxc > 0.0,
        )
        val = maxc

        # Expanded hue range: green (20-150) covers most leaf colors
        vegetation_mask = (
            (hue >= 15.0) & (hue <= 150.0) & (sat >= 0.12) & (val >= 0.10)
        )
        vegetation_ratio = float(np.mean(vegetation_mask))
        contrast = float(np.std(arr_norm))

        leaf_score = min(1.0, vegetation_ratio * 1.7 + contrast * 0.6)
        is_leaf = (
            vegetation_ratio >= vegetation_ratio_threshold
            and contrast >= contrast_threshold
        )

        reason = ""
        if not is_leaf:
            if vegetation_ratio < vegetation_ratio_threshold:
                reason = "Insufficient leaf-like color detected."
            elif contrast < contrast_threshold:
                reason = "Image lacks sufficient texture/detail."
            else:
                reason = "Leaf signal is weak."

        return {
            "is_leaf": is_leaf,
            "vegetation_ratio": round(vegetation_ratio, 3),
            "contrast": round(contrast, 3),
            "leaf_score": round(leaf_score, 3),
            "reason": reason,
        }

    except Exception as exc:
        return {
            "is_leaf": False,
            "vegetation_ratio": 0.0,
            "contrast": 0.0,
            "leaf_score": 0.0,
            "reason": f"Leaf detection failed: {exc}",
        }
