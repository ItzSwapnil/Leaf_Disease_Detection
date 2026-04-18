from __future__ import annotations

import os
from typing import Iterable, Optional

from config import EFFNET_MODEL_PATH, FINAL_MODEL_PATH


def resolve_keras_model_path(preferred_paths: Optional[Iterable[str]] = None) -> str:
    """Resolve one of the canonical model paths strictly.

    Allowed model files:
    - models/leaf_disease_refined.keras (DINOv3)
    - models/leaf_disease_EfficientNetV2-S.keras (EfficientNetV2-S)
    - models/EfficientNetv2B0/* (EfficientNetV2-B0 variants)
    - models/EfficientNetv2S/* (EfficientNetV2-S variants)
    """

    allowed = {
        os.path.abspath(str(FINAL_MODEL_PATH)),
        os.path.abspath(str(EFFNET_MODEL_PATH)),
        # EfficientNetV2-B0 variants
        os.path.abspath(
            os.path.join(
                os.path.dirname(str(FINAL_MODEL_PATH)), "EfficientNetv2B0", "leaf_disease_classifier.keras"
            )
        ),
        os.path.abspath(
            os.path.join(
                os.path.dirname(str(FINAL_MODEL_PATH)), "EfficientNetv2B0", "leaf_disease_checkpoint.keras"
            )
        ),
        # EfficientNetV2-S variants
        os.path.abspath(
            os.path.join(
                os.path.dirname(str(FINAL_MODEL_PATH)), "EfficientNetv2S", "leaf_disease_classifier.keras"
            )
        ),
        os.path.abspath(
            os.path.join(
                os.path.dirname(str(FINAL_MODEL_PATH)), "EfficientNetv2S", "leaf_disease_checkpoint.keras"
            )
        ),
        # Legacy patterns (backward compat)
        os.path.abspath(
            os.path.join(
                os.path.dirname(str(EFFNET_MODEL_PATH)), "leaf_disease_classifier.keras"
            )
        ),
        os.path.abspath(
            os.path.join(
                os.path.dirname(str(EFFNET_MODEL_PATH)), "leaf_disease_checkpoint.keras"
            )
        ),
    }

    if preferred_paths:
        for path in preferred_paths:
            if not path:
                continue
            normalized = os.path.abspath(str(path))
            if normalized not in allowed:
                raise ValueError(
                    "Only canonical model paths are allowed: "
                    f"{sorted(allowed)}. Received: {normalized}"
                )

            if os.path.exists(normalized):
                return normalized

    for candidate in [
        os.path.abspath(str(FINAL_MODEL_PATH)),
        os.path.abspath(str(EFFNET_MODEL_PATH)),
    ]:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"No canonical model file found. Expected one of: {sorted(allowed)}"
    )
