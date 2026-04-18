from __future__ import annotations

import os

import numpy as np
from tensorflow.keras.applications import (
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2M,
    EfficientNetV2L,
)
from tensorflow.keras.applications.efficientnet_v2 import (
    preprocess_input as effnetv2_preprocess_input,
)


def _preprocess_dinov3(images: np.ndarray) -> np.ndarray:
    """ImageNet mean/std normalization used by DINO-style ViT backbones."""
    arr = np.asarray(images, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (arr - mean) / std


def _build_dinov3_backbone(
    input_shape=(224, 224, 3), include_top: bool = False, weights: str = "imagenet"
):
    """Best-effort DINOv3 loader via KerasHub presets (experimental)."""
    del include_top, weights  # Not used by preset-based backbones.

    try:
        import keras_hub as kh
    except Exception as exc:
        raise ImportError(
            "DINOv3 backbone requires keras-hub. Install with: uv add keras-hub"
        ) from exc

    preset_override = (os.getenv("LEAF_DINOV3_PRESET") or "").strip()
    preset_candidates = [
        preset_override,
        # Use built-in KerasHub presets first to avoid HF auth issues.
        "vit_base_patch16_224_imagenet",
        "vit_large_patch16_224_imagenet",
        # Keep HF URIs as optional fallbacks if user provides working access.
        "hf://facebook/dinov3-vit-base-patch16-224",
        "hf://facebook/dinov3-vit-small-patch16-224",
    ]
    preset_candidates = [p for p in preset_candidates if p]

    constructors = []
    if hasattr(kh, "models") and hasattr(kh.models, "Backbone"):
        constructors.append(kh.models.Backbone.from_preset)
    if hasattr(kh, "models") and hasattr(kh.models, "ImageClassifier"):
        constructors.append(kh.models.ImageClassifier.from_preset)

    last_error = None
    tried_presets: list[str] = []
    for constructor in constructors:
        for preset in preset_candidates:
            tried_presets.append(preset)
            for _ in range(2):
                try:
                    model_or_classifier = constructor(preset)
                    backbone = getattr(
                        model_or_classifier, "backbone", model_or_classifier
                    )
                    if getattr(backbone, "input_shape", None) is None:
                        continue
                    return backbone
                except Exception as exc:
                    last_error = exc

    if isinstance(last_error, ImportError):
        error_text = str(last_error).lower()
        if "huggingface_hub" in error_text:
            raise RuntimeError(
                "DINOv3 preset loading requires Hugging Face support. "
                "Install it with: uv add huggingface-hub, then run: uv sync"
            ) from last_error

    if last_error is not None:
        error_text = str(last_error).lower()
        if "repository not found" in error_text or "unauthorized" in error_text:
            raise RuntimeError(
                "The selected Hugging Face preset is unavailable or requires auth. "
                "Use a built-in preset such as LEAF_DINOV3_PRESET=vit_base_patch16_224_imagenet."
            ) from last_error

    raise RuntimeError(
        "Failed to load DINOv3 preset via keras-hub. "
        "Set LEAF_DINOV3_PRESET to a valid preset name/URI if needed. "
        f"Tried presets: {', '.join(tried_presets)}"
    ) from last_error


BACKBONE_REGISTRY = {
    "EfficientNetV2B0": EfficientNetV2B0,
    "EfficientNetV2B1": EfficientNetV2B1,
    "EfficientNetV2B2": EfficientNetV2B2,
    "EfficientNetV2B3": EfficientNetV2B3,
    "EfficientNetV2M": EfficientNetV2M,
    "EfficientNetV2L": EfficientNetV2L,
    "DINOv3": _build_dinov3_backbone,
}


PREPROCESS_FUNCTIONS = {
    "DINOv3": _preprocess_dinov3,
}


def list_backbone_names() -> list[str]:
    return list(BACKBONE_REGISTRY.keys())


def resolve_backbone_factory(name: str):
    if name in BACKBONE_REGISTRY:
        return BACKBONE_REGISTRY[name]
    supported = ", ".join(sorted(BACKBONE_REGISTRY.keys()))
    raise ValueError(f"Supported backbones: {supported}.")


def resolve_preprocess_function(name: str):
    return PREPROCESS_FUNCTIONS.get(name, effnetv2_preprocess_input)


def resolve_backbone_name(requested: str | None, default: str) -> str:
    candidate = (requested or "").strip()
    if not candidate:
        candidate = default
    if candidate not in BACKBONE_REGISTRY:
        supported = ", ".join(sorted(BACKBONE_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported backbone '{candidate}'. Supported backbones: {supported}."
        )
    return candidate
