# Reproducible Execution Protocol

This document provides step-by-step instructions for running every pipeline
in the Leaf Disease Detection project. The default configuration uses
EfficientNetV2-S transfer learning.

## Execution Flow

```mermaid
flowchart LR
    A["Activate Env"] --> B["Sync Deps"]
    B --> C["Verify TF+GPU"]
    C --> D["Train"]
    D --> E["Fine-tune"]
    E --> F["Evaluate"]
    F --> G["Generate Figures"]
    G --> H["Run Web App / CLI"]
```

## 1. Activate Environment

```bash
cd /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection
source .venv/bin/activate
```

## 2. Install/Sync Dependencies

```bash
uv lock --refresh
uv sync
```

## 3. Runtime Check

```bash
uv run python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUs', tf.config.list_physical_devices('GPU'))"
```

## 4. Base Training

The project command dispatcher is now handled automatically:

- `BASE_MODEL=EfficientNetV2S` (default) uses `model_training.py`.

```bash
uv run python model_training.py
```

Or via the unified runner:

```bash
uv run python main.py train
```

## 5. Fine-Tuning

```bash
uv run python model_fine_tuning.py
```

- Model is loaded from `models/leaf_disease_checkpoint.keras` or `models/leaf_disease_classifier.keras` via config fallback.
- Unfreezes all layers and uses AdamW + warmup + cosine annealing schedule.

## 6. Evaluation

```bash
uv run python model_evaluation.py
```

Output in `docs/reports/`:

- Latest/archived JSON + Markdown reports
- Confusion matrix + per-class precision/recall/F1
- Top confusion pairs for error analysis

## 7. Generate Figures

```bash
uv run python visualization_pipeline.py
```

Expected outputs in `plots/`:

- `class_distribution.png`
- `learning_curves.png`
- `confusion_matrix.png`
- `model_architecture.png`
- `sample_predictions.png`

## 8. Web App

```bash
uv run python app.py
# Visit http://127.0.0.1:5000
```

The web app auto-selects the training script based on `BASE_MODEL` and uses the shared preprocessing module.

## 9. CLI Prediction

```bash
uv run python predict.py --image dataset/test/<class>/<image>.jpg
uv run python predict.py --image path/to/folder    # batch mode
```

## Key Directories

- `dataset/train`, `dataset/val`, `dataset/test` — images by class
- `models/` — checkpoints and final `.keras` files
- `models/logs/` — CSV training progress logs
- `plots/` — generated figures
- `docs/reports/` — evaluation report snapshots

## Notes

- `IMG_SIZE=224` matches EfficientNetV2-S native resolution.
- `preprocessing.py` centralises when to use inside-model scaling (ViT) vs EfficientNet `preprocess_input`.
- `USE_MIXUP=True`, `USE_CUTMIX=True`, `BATCH_SIZE=64` (fp16 on 8 GB VRAM).
- `ACCUMULATION_STEPS=1` (no gradient accumulation needed at BS=64).
- Target accuracy in config is 99.0%.
- `main.py` dispatches automatically based on `BASE_MODEL`; no silent model type mismatch.

## Quick Validation

```bash
uv run python -m py_compile config.py model_training.py model_fine_tuning.py model_evaluation.py predict.py app.py visualization_pipeline.py preprocessing.py
uv run pytest -q
```

Set `LEAF_SAVE_LOG_ARCHIVE=1` for timestamped archive log mode.
