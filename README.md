# Leaf Disease Detection System

Deep learning-based plant leaf disease classification using EfficientNetV2 transfer learning with SOTA augmentation strategies. Supports web inference, CLI inference, and reproducible train/fine-tune/evaluate pipelines.

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21%2B%20CUDA-orange.svg)
![Keras](https://img.shields.io/badge/Keras-3.13%2B-red.svg)
![Classes](https://img.shields.io/badge/Classes-46-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Table of Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model and Training](#model-and-training)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Overview

This project provides an end-to-end workflow for plant disease detection:

1. Train and fine-tune an EfficientNetV2-S classifier on 46 disease classes.
2. Evaluate with strict top-1 accuracy and macro-averaged metrics.
3. Serve predictions through a Flask web app or CLI.
4. Generate publication-ready visualisations.

```mermaid
flowchart LR
  A["Leaf Image"] --> B["Preprocessing"]
  B --> C["EfficientNetV2-S Inference"]
  C --> D["Class + Confidence"]
  D --> E["Actionable Disease Guidance"]
```

## Highlights

- **SOTA Training Pipeline**: Two-phase transfer learning with cosine annealing, MixUp, CutMix, label smoothing, and AdamW with EMA.
- **Mixed Precision**: Automatic `mixed_float16` on NVIDIA GPUs for 2x memory efficiency.
- **Multiple Interfaces**: Flask web app, CLI, and Python API.
- **Live Progress Monitoring**: Machine-readable JSON progress events for the web control panel.
- **Unified Model Resolver**: Deterministic fallback order across all scripts.
- **Publication-Ready Figures**: Confusion matrix, learning curves, class distribution, and sample predictions.

## Quick Start

### Prerequisites

- Python 3.13+
- NVIDIA GPU + CUDA runtime compatible with TensorFlow (recommended)
- RAM: 8 GB minimum, 16 GB recommended for training

### Setup (Linux / WSL)

```bash
git clone https://github.com/ItzSwapnil/Leaf_Disease_Detection.git
cd Leaf_Disease_Detection
uv venv --python 3.13
source .venv/bin/activate
uv sync
uv run python -c "import tensorflow as tf; print(tf.__version__)"
```

### Setup (Windows PowerShell)

```powershell
git clone https://github.com/ItzSwapnil/Leaf_Disease_Detection.git
cd Leaf_Disease_Detection
uv venv --python 3.13
./.venv/Scripts/Activate.ps1
uv sync
uv run python -c "import tensorflow as tf; print(tf.__version__)"
```

## Usage

### A) Web App (Recommended)

```bash
uv run python app.py
```

Open <http://127.0.0.1:5000>

Features:

- Leaf image upload and prediction
- Disease details (description, symptoms, treatment, prevention)
- Workflow control panel: Train, Fine-Tune, Evaluate, Generate Figures
- Job status and live progress logs

### B) Command Runner

```bash
uv run python main.py serve
uv run python main.py train
uv run python main.py fine_tune
uv run python main.py evaluate
uv run python main.py visualize

# Keep timestamped archive logs for a run
uv run python main.py train --archive-logs
```

### C) Dedicated Scripts

```bash
uv run python model_training.py
uv run python model_fine_tuning.py  # optional full fine-tuning
uv run python model_evaluation.py
uv run python visualization_pipeline.py
```

### D) CLI Prediction

```bash
uv run python predict.py --image path/to/leaf.jpg
uv run python predict.py --image path/to/folder
```

### E) Python API

```python
from predict import LeafDiseasePredictor

predictor = LeafDiseasePredictor()
result = predictor.predict("path/to/leaf_image.jpg")
print(result["disease"], result["confidence"])
```

### Web Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Main UI |
| `/predict` | POST | Image upload + prediction |
| `/health` | GET | App/model health check |
| `/control/actions` | GET | List workflow actions |
| `/control/run/<action_key>` | POST | Start workflow job |
| `/control/jobs` | GET | List job history/status |
| `/control/system` | GET | Compute backend info |
| `/control/stop/<job_id>` | POST | Stop running job |

## Model and Training

### Architecture

```text
Input (224x224x3)
  -> EfficientNetV2-S (ImageNet pretrained, include_top=False)
  -> GlobalAveragePooling2D
  -> BatchNormalization
  -> Dense(512, Swish) + Dropout(0.4)
  -> Dense(256, Swish) + Dropout(0.2)
  -> Dense(46, Softmax)
```

### Training Strategy

1. **Phase 1**: Train classification head with frozen backbone (5 epochs).
2. **Phase 2**: Unfreeze backbone and fine-tune with cosine LR schedule (10 epochs).
3. **Optional**: Extended fine-tuning via `fine_tune_model.py`.

### SOTA Techniques

| Technique | Reference |
| --- | --- |
| EfficientNetV2-S backbone | Tan & Le, ICML 2021 |
| MixUp augmentation | Zhang et al., ICLR 2018 |
| CutMix augmentation | Yun et al., ICCV 2019 |
| Label smoothing (0.1) | Szegedy et al., CVPR 2016 |
| AdamW + EMA | Loshchilov & Hutter, ICLR 2019 |
| Cosine annealing with warmup | Loshchilov & Hutter, ICLR 2017 |
| Mixed precision (float16) | Micikevicius et al., ICLR 2018 |

### Model Artefact Resolution Order

1. `models/leaf_disease_classifier.keras`
2. `models/leaf_disease_checkpoint.keras`
3. First discovered `.keras` in `models/`

## Dataset

| Split | Images |
| --- | --- |
| Training | 220,498 |
| Validation | 19,419 |
| Test | 19,218 |
| **Total** | **259,135** |
| Classes | 46 |

```text
dataset/
├── train/    # 46 class folders
├── val/      # 46 class folders
└── test/     # 46 class folders
```

Supported crops: Apple, Tomato, Corn, Grape, Potato, Rice, Pepper, Cherry, Peach, Strawberry, Orange, Wheat, Squash, Blueberry, Raspberry, and Soybean.

## Results

| Metric | Value |
| --- | --- |
| Validation Accuracy | 99.46% |
| Macro F1-Score | 0.9901 |
| Classes | 46 |

### Visual Outputs

![Learning Curves](plots/learning_curves.png)
![Confusion Matrix](plots/confusion_matrix.png)
![Class Distribution](plots/class_distribution.png)
![Sample Predictions](plots/sample_predictions.png)

## Project Structure

```text
Leaf_Disease_Detection/
├── pyproject.toml              # uv dependencies and metadata
├── uv.lock                     # uv strict dependency lockfile
├── app.py                      # Flask web application
├── main.py                     # Unified task runner
├── config.py                   # Central configuration
├── model_training.py           # EfficientNetV2 training pipeline
├── model_fine_tuning.py        # Fine-tuning pipeline
├── model_evaluation.py         # Evaluation and reporting
├── predict.py                  # Inference API and CLI
├── visualization_pipeline.py   # Figure generation
├── training_utils.py           # Shared training components
├── training_progress.py        # Progress emission callbacks
├── learning_curve_utils.py     # Metric timeline helpers
├── preprocessing.py            # Input preprocessing
├── model_paths.py              # Model path resolution
├── hardware.py                 # GPU/CPU configuration
├── dataset/                    # Train/val/test image data
├── models/                     # Saved model artefacts
├── plots/                      # Generated figures
├── templates/                  # Flask HTML templates
├── tests/                      # Unit tests
└── docs/                       # Architecture and run documentation
```

## Documentation

- Reproducible run protocol: [docs/run.md](docs/run.md)
- System architecture: [docs/architecture.md](docs/architecture.md)
- DFD context view: [docs/DFD_Level0.md](docs/DFD_Level0.md)
- DFD detailed view: [docs/DFD_Level1.md](docs/DFD_Level1.md)

## Configuration

Key settings in `config.py`:

```python
IMG_SIZE = 224                   # EfficientNetV2-S native resolution
BATCH_SIZE = 64                  # Fits 8 GB VRAM at fp16
BASE_MODEL = "EfficientNetV2S"   # ImageNet-pretrained backbone
EPOCHS_PHASE1 = 5                # Head-only warm-up
EPOCHS_PHASE2 = 10               # Full fine-tuning
USE_MIXUP = True                 # MixUp regularisation
USE_CUTMIX = True                # CutMix regularisation
USE_OPTIMIZER_EMA = True         # Exponential Moving Average
USE_FOCAL_LOSS = False           # CrossEntropy preferred
LABEL_SMOOTHING = 0.1
```

Runtime overrides:

- `LEAF_SAVE_LOG_ARCHIVE=1` — enable timestamped archive logs
- `LEAF_SAVE_RUN_MANIFESTS=1` — enable per-run manifest JSON files

## Testing

```bash
pytest
```

## Troubleshooting

- **Model not found**: Ensure at least one `.keras` model exists in `models/`.
- **GPU not detected**: Verify TensorFlow/CUDA compatibility. Note: AMD integrated GPUs (e.g., Radeon 860M) are not supported by TensorFlow — only NVIDIA GPUs with CUDA are usable.
- **Upload errors**: `/predict` accepts `.jpg`, `.jpeg`, `.png`, `.webp` (max 16 MB).
- **OOM during training**: Reduce `BATCH_SIZE` in `config.py` or enable gradient accumulation via `ACCUMULATION_STEPS`.

## License

Licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- PlantVillage and related agricultural datasets
- EfficientNetV2 (Tan & Le, 2021)
- TensorFlow and Keras communities

## Contact

Made by Swapnil — [@ItzSwapnil](https://github.com/ItzSwapnil)
