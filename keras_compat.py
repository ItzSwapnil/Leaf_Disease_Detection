"""Compatibility shim: expose TensorFlow's `tf.keras` as `keras_compat`.

This module avoids importing TensorFlow at import-time; attribute access
is forwarded lazily to `tf.keras` on first use.
"""
from __future__ import annotations

import importlib
from typing import Any


def _ensure_tf_keras():
    """Import TensorFlow on first real use and return `tf.keras`."""
    tf = importlib.import_module("tensorflow")
    return getattr(tf, "keras")


def __getattr__(name: str) -> Any:
    """Module-level lazy attribute resolver that proxies to `tf.keras`."""
    return getattr(_ensure_tf_keras(), name)


def __dir__() -> list[str]:
    try:
        return dir(_ensure_tf_keras())
    except Exception:
        return []
