from __future__ import annotations

import os
from typing import Iterable, List, Optional

from config import CHECKPOINT_PATH, FINAL_MODEL_PATH, MODELS_DIR

def resolve_keras_model_path(preferred_paths: Optional[Iterable[str]] = None) -> str:
    """Pick the highest-priority existing model path.

    - preferred_paths first
    - FINAL_MODEL_PATH then CHECKPOINT_PATH
    - auto-discover .keras/.h5/.hdf5 in MODELS_DIR
    """

    candidates: List[str] = []
    if preferred_paths:
        for path in preferred_paths:
            if not path:
                continue
            normalized = os.path.abspath(str(path))
            if normalized not in candidates:
                candidates.append(normalized)

    for default_path in [FINAL_MODEL_PATH, CHECKPOINT_PATH]:
        if default_path:
            normalized = os.path.abspath(str(default_path))
            if normalized not in candidates:
                candidates.append(normalized)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    discovered: List[str] = []
    if os.path.isdir(MODELS_DIR):
        for name in sorted(os.listdir(MODELS_DIR)):
            if name.lower().endswith((".keras", ".h5", ".hdf5")):
                discovered.append(os.path.abspath(os.path.join(MODELS_DIR, name)))

    if discovered:
        return discovered[0]

    formatted = {
        "preferred": list(preferred_paths or []),
        "default": [FINAL_MODEL_PATH, CHECKPOINT_PATH],
        "discovered": discovered,
    }

    raise FileNotFoundError(
        "No model file found. "
        f"Preferred paths: {formatted['preferred']}. "
        f"Default paths: {formatted['default']}. "
        f"Discovered: {formatted['discovered']}."
    )
