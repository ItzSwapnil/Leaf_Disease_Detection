"""Model path resolution utilities for PyTorch ``.pt`` / ``.pth`` checkpoints.

Searches canonical model locations defined in ``src.utils.config`` and
falls back to a recursive discovery scan of the ``models/`` directory for
``.pt``, ``.pth``, or ``.keras`` files.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

from src.utils.config import (
    CHECKPOINT_PATH,
    CLASSIFIER_PATH,
    EFFNET_MODEL_PATH,
    FINAL_MODEL_PATH,
    MODELS_DIR,
)


def resolve_model_path(
    preferred_paths: Optional[Iterable[str]] = None,
) -> str:
    """Resolve a PyTorch model file path from candidates.

    Args:
        preferred_paths: Optional iterable of paths to check first.

    Returns:
        Absolute path to the first existing ``.pt`` / ``.pth`` / ``.keras`` model file.

    Raises:
        FileNotFoundError: If no model file is found in any candidate location.
    """
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
                if filename.lower().endswith((".pt", ".pth", ".keras"))
            )

    for candidate in sorted(discovered):
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "No model file found. Expected one of: "
        f"{canonical_candidates} or any .pt/.pth/.keras under {models_root}"
    )


# Backward-compatible alias
resolve_pytorch_model_path = resolve_model_path
resolve_keras_model_path = resolve_model_path
