"""
Multi-stage leaf disease detection pipeline.

Orchestrates leaf validation, YOLO focus detection, classification, and
analysis. Stage 2 preserves the original image pixels; any YOLO bounding box is
metadata for review and saliency guidance only.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.yolo_leaf import YOLOLeafDetector

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from src.core.inference_guard import (
    compute_prediction_diagnostics,
    evaluate_inference_safety,
)
from src.core.leaf_detector import detect_leaf_presence
from src.core.leaf_detector_model import create_leaf_detector
from src.utils.config import (
    CHECKPOINT_PATH,
    CLASS_INDICES_PATH,
    CONFIDENCE_REJECT_THRESHOLD,
    ENTROPY_REJECT_THRESHOLD,
    IMG_SIZE,
    OOD_MSP_THRESHOLD,
    USE_YOLO_LEAF_DETECTION,
)
from src.utils.model_paths import resolve_pytorch_model_path


def _extract_disease_output(outputs):
    """Return disease output tensor from single-output or multi-output models."""
    if isinstance(outputs, dict):
        if "disease_output" in outputs:
            return outputs["disease_output"]
        return next(iter(outputs.values()))
    if isinstance(outputs, (list, tuple)):
        return outputs[-1]
    return outputs


class LeafDiseasePipeline:
    """Multi-stage pipeline for leaf disease detection."""

    def __init__(self, model_paths: list[str] | str | None = None):
        from src.pipeline.predict import _load_model_robust
        from src.utils.config import ENSEMBLE_MODEL_PATHS
        from src.utils.hardware import get_device

        if not model_paths and ENSEMBLE_MODEL_PATHS:
            model_paths = ENSEMBLE_MODEL_PATHS
        elif isinstance(model_paths, str):
            model_paths = [model_paths]
        elif not model_paths:
            model_paths = [CHECKPOINT_PATH]

        self.device = get_device()
        self.models = []
        for path in model_paths:
            resolved_path = resolve_pytorch_model_path([path] if path else None)
            model, b_name = _load_model_robust(resolved_path)
            self.models.append(model)

        self._load_class_indices()

        self.transform = v2.Compose([
            v2.Resize((IMG_SIZE, IMG_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.leaf_detector = None
        try:
            self.leaf_detector = create_leaf_detector()
            print("[*] Loaded trained leaf detector model")
        except FileNotFoundError:
            print(
                "[!] Trained leaf detector not found. "
                "Using heuristic instead. Run: python train_leaf_detector.py"
            )

        self.use_yolo = USE_YOLO_LEAF_DETECTION and (
            os.getenv("LEAF_PIPELINE_YOLO_FOCUS", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.yolo_leaf_detector: YOLOLeafDetector | None = None

    def _get_yolo_leaf_detector(self):
        """Return a lazily initialized YOLO focus detector, if available."""
        if not self.use_yolo:
            return None
        if self.yolo_leaf_detector is not None:
            return self.yolo_leaf_detector
        try:
            from src.core.yolo_leaf import YOLOLeafDetector

            self.yolo_leaf_detector = YOLOLeafDetector()
        except Exception as exc:
            print(f"[!] Failed to initialize YOLOLeafDetector: {exc}")
            self.yolo_leaf_detector = None
        return self.yolo_leaf_detector

    def _load_class_indices(self) -> None:
        from pathlib import Path

        if Path(CLASS_INDICES_PATH).exists():
            with open(CLASS_INDICES_PATH, encoding="utf-8") as f:
                indices_dict = json.load(f)
                first_key = next(iter(indices_dict), None)
                if first_key is not None and str(first_key).isdigit():
                    self.class_names = {
                        int(k): str(v) for k, v in indices_dict.items()
                    }
                else:
                    self.class_names = {
                        int(v): str(k) for k, v in indices_dict.items()
                    }
        else:
            self.class_names = {i: f"Class_{i}" for i in range(46)}

    def stage_1_detect_leaf(self, img_path: str) -> dict[str, Any]:
        if self.leaf_detector is not None:
            return self.leaf_detector.predict(img_path)

        return detect_leaf_presence(img_path, img_size=IMG_SIZE)

    def stage_2_remove_background(self, img_path: str) -> dict[str, Any]:
        try:
            with Image.open(img_path) as img:
                img_resized = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                img_array = np.asarray(img_resized, dtype=np.float32)

            crop_bbox = (0, 0, img_array.shape[1], img_array.shape[0])
            reason = "Bypassed masking (kept original image)"

            yolo_leaf_detector = self._get_yolo_leaf_detector()
            if yolo_leaf_detector is not None:
                img_bgr = cv2.imread(img_path)
                if img_bgr is not None:
                    detection = yolo_leaf_detector.detect(img_bgr)
                    if detection["found"]:
                        crop_bbox = detection["bbox"]
                        reason = f"YOLOv26 leaf focus detected bbox: {crop_bbox}"

            return {
                "preprocessed_image": img_path, # Changed to return path for transform
                "crop_bbox": crop_bbox,
                "mask": None,
                "segmentation_ratio": 1.0,
                "reason": reason,
            }
        except Exception as exc:
            return {
                "preprocessed_image": None,
                "crop_bbox": None,
                "mask": None,
                "segmentation_ratio": 0.0,
                "reason": f"Failed to load image: {exc}",
            }

    @torch.no_grad()
    def stage_3_classify_leaf(
        self, img_path: str
    ) -> tuple[str, float, dict[str, Any]]:
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        if len(self.models) > 1:
            all_probs = []
            for m in self.models:
                logits = _extract_disease_output(m(tensor))
                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy()[0])
            probs = np.mean(all_probs, axis=0)
        else:
            logits = _extract_disease_output(self.models[0](tensor))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        predicted_idx = int(np.argmax(probs))
        predicted_class = self.class_names.get(
            predicted_idx, f"Class_{predicted_idx}"
        )
        confidence = float(probs[predicted_idx])

        diagnostics = compute_prediction_diagnostics(probs)

        return predicted_class, confidence, diagnostics

    def stage_4_disease_analysis(self, class_name: str) -> dict[str, Any]:
        parts = class_name.split("___")
        plant = parts[0] if parts else "Unknown"
        disease = parts[1] if len(parts) > 1 else "Unknown"

        return {
            "plant": plant,
            "disease": disease,
            "class_name": class_name,
        }

    def predict(
        self,
        img_path: str,
        skip_safety_check: bool = False,
    ) -> dict[str, Any]:
        pipeline_stages: list[tuple[str, Any]] = []

        # Stage 1: Leaf Detection
        leaf_detection = self.stage_1_detect_leaf(img_path)
        pipeline_stages.append(("leaf_detection", leaf_detection))

        if not leaf_detection["is_leaf"]:
            return {
                "success": False,
                "pipeline_stages": pipeline_stages,
                "final_prediction": None,
                "rejection_reason": leaf_detection.get("reason", "Not a leaf"),
            }

        # Stage 2: Background Removal
        bg_removal = self.stage_2_remove_background(img_path)
        pipeline_stages.append(("background_removal", bg_removal))

        if bg_removal["preprocessed_image"] is None:
            return {
                "success": False,
                "pipeline_stages": pipeline_stages,
                "final_prediction": None,
                "rejection_reason": bg_removal.get(
                    "reason", "Background removal failed"
                ),
            }

        # Stage 3: Classify Leaf
        class_name, confidence, diagnostics = self.stage_3_classify_leaf(
            bg_removal["preprocessed_image"]
        )
        classification_result = {
            "class_name": class_name,
            "confidence": confidence,
            "diagnostics": diagnostics,
        }
        pipeline_stages.append(("leaf_classification", classification_result))

        # Stage 4: Disease Analysis
        disease_info = self.stage_4_disease_analysis(class_name)

        if not skip_safety_check:
            safety = evaluate_inference_safety(
                diagnostics=diagnostics,
                leaf_validation=leaf_detection,
                confidence_threshold=CONFIDENCE_REJECT_THRESHOLD,
                entropy_threshold_bits=ENTROPY_REJECT_THRESHOLD,
                msp_threshold=OOD_MSP_THRESHOLD,
            )
            pipeline_stages.append(("safety_check", safety))

            if safety["reject"]:
                return {
                    "success": False,
                    "pipeline_stages": pipeline_stages,
                    "final_prediction": {
                        "class_name": class_name,
                        "confidence": confidence,
                        "disease_info": disease_info,
                    },
                    "rejected": True,
                    "rejection_reason": safety.get(
                        "reason", "Inference safety check failed"
                    ),
                }

        return {
            "success": True,
            "pipeline_stages": pipeline_stages,
            "final_prediction": {
                "class_name": class_name,
                "confidence": confidence,
                "disease_info": disease_info,
                "diagnostics": diagnostics,
            },
            "rejected": False,
        }

def create_pipeline(model_path: str | None = None) -> LeafDiseasePipeline:
    return LeafDiseasePipeline(model_path)
