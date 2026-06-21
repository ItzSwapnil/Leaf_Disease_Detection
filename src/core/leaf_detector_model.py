"""
Inference wrapper for the trained binary leaf detector model.

This replaces the heuristic leaf detection with a learned model.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from src.utils.config import IMG_SIZE, MODELS_DIR
from src.utils.hardware import get_device

LEAF_DETECTOR_MODEL_PATH = Path(MODELS_DIR) / "leaf_detector_final.pt"
LEAF_DETECTOR_CHECKPOINT_PATH = (
    Path(MODELS_DIR) / "leaf_detector_checkpoint.pt"
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

        self.device = get_device()
        self.model = torch.load(str(model_path), map_location=self.device)
        self.model.eval()
        self.model_path = model_path

        self.transform = v2.Compose([
            v2.Resize((IMG_SIZE, IMG_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # Standard ImageNet normalization commonly used with Torchvision models
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
                img_rgb = img.convert("RGB")

            # Apply transforms
            input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                output = self.model(input_tensor)
                # Assuming model outputs logits for binary classification (1 output node)
                # or 2 output nodes. If 1 node:
                if output.shape[-1] == 1:
                    leaf_prob = torch.sigmoid(output).item()
                else:
                    # If 2 nodes (softmax)
                    leaf_prob = torch.softmax(output, dim=-1)[0, 1].item()

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
