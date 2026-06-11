from __future__ import annotations

import os
from typing import Iterable, Optional

from config import (
    CHECKPOINT_PATH,
    CLASSIFIER_PATH,
    EFFNET_MODEL_PATH,
    FINAL_MODEL_PATH,
    MODELS_DIR,
)


def resolve_keras_model_path(
    preferred_paths: Optional[Iterable[str]] = None,
) -> str:
    if preferred_paths:
        for path in preferred_paths:
            if not path:
                continue
            normalized = os.path.abspath(str(path))
            if os.path.exists(normalized):
                return normalized

    canonical_candidates = [
        os.path.abspath(str(FINAL_MODEL_PATH)),
        os.path.abspath(str(CLASSIFIER_PATH)),
        os.path.abspath(str(CHECKPOINT_PATH)),
        os.path.abspath(str(EFFNET_MODEL_PATH)),
    ]

    for candidate in canonical_candidates:
        if os.path.exists(candidate):
            return candidate

    models_root = os.path.abspath(str(MODELS_DIR))
    discovered: list[str] = []
    if os.path.isdir(models_root):
        for root, _, files in os.walk(models_root):
            discovered.extend(
                os.path.abspath(os.path.join(root, filename))
                for filename in files
                if filename.lower().endswith((".keras", ".h5"))
            )

    for candidate in sorted(discovered):
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "No model file found. Expected one of: "
        f"{canonical_candidates} or any .keras/.h5 under {models_root}"
    )


resolve_model_path = resolve_keras_model_path


# Globally register custom layers so load_model doesn't fail
try:
    import training_utils  # noqa: F401
except ImportError:
    pass
