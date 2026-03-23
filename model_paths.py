from __future__ import annotations

import os
from typing import Iterable, List, Optional

from config import CHECKPOINT_PATH, FINAL_MODEL_PATH, MODELS_DIR

def resolve_keras_model_path(preferred_paths: Optional[Iterable[str]] = None) -> str:
    
    candidates: List[str] = []
    if preferred_paths:
        for path in preferred_paths:
            if path and path not in candidates:
                candidates.append(path)

    for default_path in [FINAL_MODEL_PATH, CHECKPOINT_PATH]:
        if default_path not in candidates:
            candidates.append(default_path)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    discovered: List[str] = []
    if os.path.isdir(MODELS_DIR):
        for name in sorted(os.listdir(MODELS_DIR)):
            if name.endswith(".keras"):
                discovered.append(os.path.join(MODELS_DIR, name))

    if discovered:
        return discovered[0]

    raise FileNotFoundError(
        "No model file found. "
        f"Expected one of: {candidates}. "
        f"Discovered in models/: {discovered if discovered else 'none'}"
    )
