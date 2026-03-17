# How to Run (GPU-first)

Follow these steps in order on WSL (bash). For Windows IDEs, point the interpreter to `.venv\Scripts\python.exe` instead.

## 1) Activate the existing venv
```bash
cd /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection
source .venv/bin/activate
```

## 2) Sync dependencies (Python 3.13, TF 2.21 CUDA)
```bash
uv lock --refresh
uv sync
```

## 3) Verify TensorFlow + GPU
```bash
uv run python - <<'PY'
import tensorflow as tf
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
PY
```

## 4) Optional warm-up (pays PTX JIT cost once for SM 12.0a)
```bash
uv run python - <<'PY'
import tensorflow as tf, numpy as np
x = tf.constant(np.random.rand(2,224,224,3), dtype=tf.float32)
m = tf.keras.applications.EfficientNetV2B0(include_top=False, weights=None)
_ = m(x)
print("Warm-up done; GPUs:", tf.config.list_physical_devices('GPU'))
PY
```

## 5) Train
```bash
uv run python train_99pct.py
```

## 6) Resume fine-tuning (memory-optimized)
```bash
uv run python resume_training.py
```

## 7) Validate saved model
```bash
uv run python validation.py
```

## 8) Web app (Flask)
```bash
uv run python app.py
# Open http://127.0.0.1:5000
```

## 9) CLI prediction
```bash
uv run python predict.py --image dataset/test/<Class>/<image>.jpg --top_k 3
```

Notes:
- Image size is 224x224 for training/inference; batch size set to 16 to balance GPU memory.
- GPU is preferred everywhere via `hardware.configure_tensorflow()`; memory growth is enabled.
- PTX JIT warnings for compute capability 12.0a are expected on first heavy run; the warm-up step mitigates latency.

