"""
Inference wrapper for the trained binary leaf detector model.

This replaces the heuristic leaf detection with a learned model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from config import IMG_SIZE, MODELS_DIR

LEAF_DETECTOR_MODEL_PATH = Path(MODELS_DIR) / "leaf_detector_final.keras"
LEAF_DETECTOR_CHECKPOINT_PATH = (
    Path(MODELS_DIR) / "leaf_detector_checkpoint.keras"
)


class LeafDetectorModel:
    """Binary leaf/non-leaf classifier using trained model."""

    def __init__(self, model_path: str | Path | None = None):
        """Initialize with trained leaf detector model."""
        if model_path is None:
            if LEAF_DETECTOR_MODEL_PATH.exists():
                model_path = LEAF_DETECTOR_MODEL_PATH
            elif LEAF_DETECTOR_CHECKPOINT_PATH.exists():
                model_path = LEAF_DETECTOR_CHECKPOINT_PATH
            else:
                raise FileNotFoundError(
                    f"Leaf detector model not found. "
                    f"Expected: {LEAF_DETECTOR_MODEL_PATH} or {LEAF_DETECTOR_CHECKPOINT_PATH}. "
                    f"Run: python train_leaf_detector.py"
                )

        self.model = tf.keras.models.load_model(str(model_path), compile=False)
        self.model_path = model_path

    def predict(self, img_path: str) -> dict:
        """
        Predict if an image contains a leaf.

        Returns:
            dict with:
            - 'is_leaf': bool, whether a leaf is detected
            - 'leaf_score': float 0..1, model confidence that image contains a leaf
            - 'non_leaf_score': float 0..1, model confidence that image does NOT contain leaf
            - 'reason': str, explanation if rejected
        """
        try:
            with Image.open(img_path) as img:
                img_array = np.asarray(
                    img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
                    dtype=np.float32,
                )

            if img_array.size == 0:
                return {
                    "is_leaf": False,
                    "leaf_score": 0.0,
                    "non_leaf_score": 1.0,
                    "reason": "Image could not be loaded.",
                }

            # Normalize
            img_array = img_array / 255.0

            # Predict
            leaf_prob = float(
                self.model.predict(img_array[None, ...], verbose=0)[0, 0]
            )
            non_leaf_prob = 1.0 - leaf_prob

            # Decision threshold: 0.5
            is_leaf = leaf_prob >= 0.5

            reason = ""
            if not is_leaf:
                if non_leaf_prob > 0.8:
                    reason = "Model is confident this is not a leaf image."
                else:
                    reason = "Model is uncertain whether this is a leaf (borderline case)."

            return {
                "is_leaf": is_leaf,
                "leaf_score": round(leaf_prob, 3),
                "non_leaf_score": round(non_leaf_prob, 3),
                "reason": reason,
            }

        except Exception as exc:
            return {
                "is_leaf": False,
                "leaf_score": 0.0,
                "non_leaf_score": 1.0,
                "reason": f"Leaf detection failed: {exc}",
            }


def create_leaf_detector(
    model_path: str | Path | None = None,
) -> LeafDetectorModel:
    """Create a leaf detector instance."""
    return LeafDetectorModel(model_path)
