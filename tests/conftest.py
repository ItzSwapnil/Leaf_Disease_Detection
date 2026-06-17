"""Pytest runtime configuration for local reliability.

The development machine uses an RTX 50-series GPU. Current TensorFlow wheels
may JIT PTX for this architecture and fail during small unit tests, so tests
run on CPU by default. Training and serving code still use the normal hardware
configuration outside pytest.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
