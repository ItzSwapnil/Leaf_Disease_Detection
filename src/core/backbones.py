from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_V2_L_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
    ViT_B_16_Weights,
    ViT_L_16_Weights,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
    vit_b_16,
    vit_l_16,
)

# EfficientNetV2 B0-B3 are not natively in torchvision under those names,
# but torchvision provides V2_S, V2_M, V2_L.
# We will alias the requested names to the closest available if needed,
# or we can use timm (but we didn't install timm). We will alias them to V2_S for now.

def _build_efficientnet_v2_s(pretrained=True):
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    return efficientnet_v2_s(weights=weights)

def _build_efficientnet_v2_m(pretrained=True):
    weights = EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
    return efficientnet_v2_m(weights=weights)

def _build_efficientnet_v2_l(pretrained=True):
    weights = EfficientNet_V2_L_Weights.DEFAULT if pretrained else None
    return efficientnet_v2_l(weights=weights)

def _build_vit_b_16(pretrained=True):
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    return vit_b_16(weights=weights)

def _build_vit_l_16(pretrained=True):
    weights = ViT_L_16_Weights.DEFAULT if pretrained else None
    return vit_l_16(weights=weights)

BACKBONE_REGISTRY = {
    "EfficientNetV2B0": _build_efficientnet_v2_s,
    "EfficientNetV2B1": _build_efficientnet_v2_s,
    "EfficientNetV2B2": _build_efficientnet_v2_s,
    "EfficientNetV2B3": _build_efficientnet_v2_s,
    "EfficientNetV2S": _build_efficientnet_v2_s,
    "EfficientNetV2M": _build_efficientnet_v2_m,
    "EfficientNetV2L": _build_efficientnet_v2_l,
    "ViT_B_16": _build_vit_b_16,
    "ViT_L_16": _build_vit_l_16,
    "DINOv3": _build_vit_b_16, # Fallback to standard ViT since no DINOv3 natively in torchvision without torch.hub
}

def list_backbone_names() -> list[str]:
    return list(BACKBONE_REGISTRY.keys())

def resolve_backbone_factory(name: str):
    if name in BACKBONE_REGISTRY:
        return BACKBONE_REGISTRY[name]
    supported = ", ".join(sorted(BACKBONE_REGISTRY.keys()))
    raise ValueError(f"Supported backbones: {supported}.")

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

class BackboneWrapper(nn.Module):
    """
    Wraps a torchvision backbone to behave somewhat like a feature extractor,
    replacing the final classification head with an Identity layer, so we can
    attach our own classification head.
    """
    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        self.name = resolve_backbone_name(name, default="EfficientNetV2S")
        factory = resolve_backbone_factory(self.name)
        self.backbone = factory(pretrained=pretrained)

        # Remove the classification head
        if hasattr(self.backbone, "classifier"):
            if isinstance(self.backbone.classifier, nn.Sequential):
                # Usually out_features is the in_features of the last layer
                self.out_features = self.backbone.classifier[-1].in_features
            else:
                self.out_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "heads"):
            # For ViT
            if isinstance(self.backbone.heads, nn.Sequential):
                self.out_features = self.backbone.heads[-1].in_features
            else:
                self.out_features = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()
        elif hasattr(self.backbone, "fc"):
            self.out_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError(f"Could not find classifier head to replace in {self.name}")

    def forward(self, x):
        return self.backbone(x)
