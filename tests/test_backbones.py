import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.core.backbones import (
    list_backbone_names,
    resolve_backbone_name,
    resolve_preprocess_function,
)


def test_backbone_registry_contains_efficientnetv2_variants():
    names = list_backbone_names()

    assert "EfficientNetV2S" in names
    assert "EfficientNetV2B0" in names
    assert "DINOv3" in names
    assert len(names) >= 6


def test_resolve_backbone_name_uses_default_when_empty():
    assert (
        resolve_backbone_name("", default="EfficientNetV2S")
        == "EfficientNetV2S"
    )


def test_resolve_backbone_name_allows_dinov3_backbone():
    assert (
        resolve_backbone_name("DINOv3", default="EfficientNetV2S") == "DINOv3"
    )


def test_resolve_backbone_name_rejects_unknown_backbone():
    with pytest.raises(ValueError):
        resolve_backbone_name("UnknownBackbone", default="EfficientNetV2S")


def test_resolve_preprocess_function_for_dinov3_normalizes_input():
    preprocess = resolve_preprocess_function("DINOv3")
    x = np.full((1, 2, 2, 3), 255.0, dtype=np.float32)
    out = preprocess(x)

    assert out.shape == x.shape
    assert out.dtype == np.float32
