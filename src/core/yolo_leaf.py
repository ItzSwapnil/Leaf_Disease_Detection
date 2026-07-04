"""YOLOv26 leaf focus detection utilities.

The classifier must always receive the original RGB image. YOLO detections are
used only to derive focus masks for saliency alignment or visual review
overlays.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from src.utils.config import MODELS_DIR

YOLO_MODEL_PATH = Path(MODELS_DIR) / "yolo26_leaf_detector.pt"


def _create_yolo_model(model_path: str | Path):
    """Create an Ultralytics YOLO model lazily to avoid import side effects."""
    from ultralytics import YOLO

    return YOLO(str(model_path))


class LeafDetection(TypedDict):
    """Detection result from YOLOv26 leaf detector."""
    found: bool
    bbox: tuple[int, int, int, int]
    confidence: float


class YOLOLeafDetector:
    """Detect leaves via YOLOv26 and expose focus targets."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            model_path = (
                YOLO_MODEL_PATH
                if YOLO_MODEL_PATH.exists()
                else Path(MODELS_DIR) / "yolo26m.pt"
            )
        self.model = _create_yolo_model(model_path)
        self.model_path = model_path
        print(f"[*] YOLOLeafDetector initialized: {model_path}")

    def detect(self, image: np.ndarray | str | Path) -> LeafDetection:
        """Run YOLOv26 inference, return best leaf bbox and confidence."""
        if isinstance(image, (str, Path)):
            img_arr = cv2.imread(str(image))
            if img_arr is None:
                return {
                    "found": False,
                    "bbox": (0, 0, 0, 0),
                    "confidence": 0.0,
                }
        else:
            img_arr = image

        results = self.model.predict(img_arr, verbose=False)

        if (
            len(results) > 0
            and results[0].boxes is not None
            and len(results[0].boxes) > 0
        ):
            confs = results[0].boxes.conf.cpu().numpy()
            best = int(np.argmax(confs))
            xyxy = results[0].boxes.xyxy[best].cpu().numpy()
            h, w = img_arr.shape[:2]
            x1 = max(0, int(xyxy[0]))
            y1 = max(0, int(xyxy[1]))
            x2 = min(w, int(xyxy[2]))
            y2 = min(h, int(xyxy[3]))
            return {
                "found": True,
                "bbox": (x1, y1, x2, y2),
                "confidence": float(confs[best]),
            }

        return {"found": False, "bbox": (0, 0, 0, 0), "confidence": 0.0}

    def get_masked(self, image_path: str | Path) -> np.ndarray:
        """Return the untouched RGB image for legacy callers.

        This method is intentionally non-masking. Older call sites used the
        name while experimenting with pixel-level masking, but the active
        training and inference contract keeps all input pixels unchanged.
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def get_focus_mask(self, image: np.ndarray) -> np.ndarray:
        """Return a binary focus mask with 1.0 inside the detected leaf box."""
        h, w = image.shape[:2]
        detection = self.detect(image)
        mask = np.zeros((h, w, 1), dtype=np.float32)
        if detection["found"]:
            x1, y1, x2, y2 = detection["bbox"]
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                mask[y1:y2, x1:x2] = 1.0
                return mask
        # Fallback: focus on the entire image
        mask.fill(1.0)
        return mask
