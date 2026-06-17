"""
Multi-stage leaf disease detection pipeline.

Orchestrates leaf validation, YOLO focus detection, classification, and
analysis. Stage 2 preserves the original image pixels; any YOLO bounding box is
metadata for review and saliency guidance only.
"""

from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from src.core.inference_guard import (
    compute_prediction_diagnostics,
    evaluate_inference_safety,
)
from src.core.leaf_detector import detect_leaf_presence
from src.core.leaf_detector_model import create_leaf_detector
from src.core.preprocessing import get_preprocessing_fn
from src.utils.config import (
    CHECKPOINT_PATH,
    CLASS_INDICES_PATH,
    CONFIDENCE_REJECT_THRESHOLD,
    ENTROPY_REJECT_THRESHOLD,
    IMG_SIZE,
    OOD_MSP_THRESHOLD,
    USE_YOLO_LEAF_DETECTION,
)
from src.utils.model_paths import resolve_keras_model_path


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
        """Initialize pipeline with disease classifiers and leaf detector.

        Supports ensembling if multiple model paths are provided.
        """
        from src.utils.config import ENSEMBLE_MODEL_PATHS

        if not model_paths and ENSEMBLE_MODEL_PATHS:
            model_paths = ENSEMBLE_MODEL_PATHS
        elif isinstance(model_paths, str):
            model_paths = [model_paths]
        elif not model_paths:
            model_paths = [CHECKPOINT_PATH]

        from tensorflow.keras.models import load_model

        self.models = []
        for path in model_paths:
            resolved_path = resolve_keras_model_path([path] if path else None)
            self.models.append(load_model(resolved_path, compile=False))

        self._load_class_indices()
        # Auto-detect backbone of first model to select correct preprocessing
        detected_backbone = "EfficientNetV2B0"
        if self.models:
            first_model = self.models[0]
            for layer in getattr(first_model, "layers", []):
                layer_name = (getattr(layer, "name", "") or "").lower()
                if any(
                    tok in layer_name for tok in ["vit", "dino", "transformer"]
                ):
                    detected_backbone = "DINOv3"
                    break
        self.preprocessing_fn = get_preprocessing_fn(detected_backbone)

        # Try to load trained leaf detector; fallback to heuristic
        self.leaf_detector = None
        try:
            self.leaf_detector = create_leaf_detector()
            print("[*] Loaded trained leaf detector model")
        except FileNotFoundError:
            print(
                "[!] Trained leaf detector not found. "
                "Using heuristic instead. Run: python train_leaf_detector.py"
            )

        # YOLO focus detection is intentionally lazy. Importing Ultralytics can
        # initialize Torch/Triton, which is unnecessary for classifier startup.
        self.use_yolo = USE_YOLO_LEAF_DETECTION and (
            os.getenv("LEAF_PIPELINE_YOLO_FOCUS", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.yolo_leaf_detector = None

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
        """Load class name mapping."""
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
            # Fallback: numeric class names
            self.class_names = {i: f"Class_{i}" for i in range(46)}

    def stage_1_detect_leaf(self, img_path: str) -> dict[str, Any]:
        """
        Stage 1: Detect if the image contains a valid leaf.

        Uses trained model if available, otherwise falls back to heuristic.

        Returns:
            dict with is_leaf (bool), leaf_score, reason
        """
        if self.leaf_detector is not None:
            return self.leaf_detector.predict(img_path)

        return detect_leaf_presence(img_path, img_size=IMG_SIZE)

    def stage_2_remove_background(self, img_path: str) -> dict[str, Any]:
        """
        Stage 2: Load original image and compute the YOLO focus bounding box.

        The method name and return keys remain for compatibility with older
        callers. ``preprocessed_image`` is the untouched resized RGB image, and
        ``crop_bbox`` is only a focus/review bounding box.
        """
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
                "preprocessed_image": img_array,
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

    def stage_3_classify_leaf(
        self, img_array: np.ndarray
    ) -> tuple[str, float, dict[str, Any]]:
        """
        Stage 3: Classify leaf into one of 46 disease classes.

        Returns:
            tuple: (predicted_class_name, confidence, diagnostics_dict)
        """
        # Preprocess input
        if img_array.dtype != np.float32:
            img_array = img_array.astype(np.float32)
        if img_array.max() > 1.0:
            img_array = img_array / 255.0

        # Ensure correct shape
        if img_array.shape != (IMG_SIZE, IMG_SIZE, 3):
            if len(img_array.shape) == 2:
                img_array = np.repeat(img_array[:, :, None], 3, axis=2)
            if (
                img_array.shape[0] != IMG_SIZE
                or img_array.shape[1] != IMG_SIZE
            ):
                img_array = (
                    np.asarray(
                        Image.fromarray(
                            (img_array * 255).astype(np.uint8)
                        ).resize((IMG_SIZE, IMG_SIZE)),
                        dtype=np.float32,
                    )
                    / 255.0
                )

        # Apply preprocessing
        img_processed = self.preprocessing_fn(img_array[None, ...])

        # Predict
        if len(self.models) > 1:
            all_probs = []
            for m in self.models:
                logits = m(img_processed, training=False)
                logits = _extract_disease_output(logits)
                all_probs.append(tf.nn.softmax(logits, axis=-1).numpy()[0])
            probs = np.mean(all_probs, axis=0)
        else:
            logits = self.models[0](img_processed, training=False)
            logits = _extract_disease_output(logits)
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

        # Diagnostics
        predicted_idx = int(np.argmax(probs))
        predicted_class = self.class_names.get(
            predicted_idx, f"Class_{predicted_idx}"
        )
        confidence = float(probs[predicted_idx])

        diagnostics = compute_prediction_diagnostics(probs)

        return predicted_class, confidence, diagnostics

    def stage_4_disease_analysis(self, class_name: str) -> dict[str, Any]:
        """
        Stage 4: Provide disease-specific insights.

        Returns:
            dict with disease_info, treatment, prevention, etc.
        """
        # Parse class name to extract plant and disease
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
        """
        Execute full multi-stage pipeline on an image.

        Returns:
            dict with pipeline_stages (list), final_prediction,
            rejection details, etc.
        """
        pipeline_stages = []

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

        # Final safety check
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
    """Create a new pipeline instance."""
    return LeafDiseasePipeline(model_path)
