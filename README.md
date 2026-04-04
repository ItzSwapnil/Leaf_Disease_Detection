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
  B --> C["EfficientNetV2 Inference"]
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

- Python 3.13 (pyproject requires >=3.13,<3.14)
- NVIDIA GPU + CUDA runtime compatible with TensorFlow (recommended for training)
- RAM: 8 GB minimum, 16 GB recommended for training

### Setup

There are two supported ways to prepare the environment: using the `uv` tool (recommended for reproducible installs with `pyproject.toml` and `uv.lock`) or a standard `venv` + `pip` workflow.

Recommended (using `uv`):

```bash
git clone https://github.com/ItzSwapnil/Leaf_Disease_Detection.git
cd Leaf_Disease_Detection
# create virtual env (uv manages pyproject-based envs)
uv venv --python 3.13
source .venv/bin/activate
uv sync
# quick TF probe
uv run python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Alternative (standard `venv` + `pip`):

```bash
git clone https://github.com/ItzSwapnil/Leaf_Disease_Detection.git
cd Leaf_Disease_Detection
python3.13 -m venv .venv
source .venv/bin/activate          # PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# verify TensorFlow and GPU available
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

## Usage

### A) Web App (Recommended)

Run the Flask app directly or via the project runner. With `uv`:

```bash
uv run python app.py
```

Or with an activated `venv`:

```bash
python app.py
```

Open <http://127.0.0.1:5000>

Features:

- Leaf image upload and prediction
- Disease details (description, symptoms, treatment, prevention)
- Workflow control panel: Train, Fine-Tune, Evaluate, Generate Figures
- Job status and live progress logs

### Quick Architecture Map

| What you want to do | Entry file | Run command |
| --- | --- | --- |
| Launch web UI + control panel | `app.py` | `uv run python app.py` |
| Run task dispatcher (serve/train/evaluate/visualize) | `main.py` | `uv run python main.py <task>` |
| Train model | `train_model.py` | `uv run python train_model.py` |
| Fine-tune model | `fine_tune_model.py` | `uv run python fine_tune_model.py` |
| Evaluate model | `evaluate_model.py` | `uv run python evaluate_model.py` |
| Run CLI prediction | `predict.py` | `uv run python predict.py --image <path>` |
| Generate core figures | `scripts/generate_figures.py` | `uv run python scripts/generate_figures.py` |
| Generate publication figures | `scripts/generate_publication_figures.py` | `uv run python scripts/generate_publication_figures.py` |
| Build report tables | `tools/reporting/generate_report_tables.py` | `uv run python tools/reporting/generate_report_tables.py` |
| Count dataset per split/class | `tools/dataset/count_dataset.py` | `uv run python tools/dataset/count_dataset.py` |

### B) Command Runner

The unified CLI dispatcher is `main.py` and provides the same tasks used by the web UI:

```bash
# Start the web app via runner
python main.py serve

# Training / evaluation / visualization
python main.py train
python main.py fine_tune
python main.py evaluate
python main.py visualize

# Keep timestamped archive logs for a run
python main.py train --archive-logs
```

If you install the package via `pyproject`/`uv`, the following console entrypoints are available as well: `leaf-disease`, `leaf-disease-train`, `leaf-disease-fine-tune`, `leaf-disease-evaluate`, `leaf-disease-predict`, `leaf-disease-figures` (see `[project.scripts]` in `pyproject.toml`).

### C) Dedicated Scripts

You can run each pipeline script directly (activated `venv`), for example:

```bash
python train_model.py
python fine_tune_model.py  # optional full fine-tuning
python evaluate_model.py
python scripts/generate_figures.py
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

The project resolves model files with the following priority (see `model_paths.resolve_keras_model_path`):

1. `models/leaf_disease_classifier.keras` (final trained model, if present)
2. `models/leaf_disease_checkpoint.keras` (latest checkpoint)
3. First discovered `.keras`, `.h5`, or `.hdf5` file in `models/`

Note: This repository currently includes `models/leaf_disease_checkpoint.keras` (a checkpoint saved during training). The final classifier file may be created by the training pipeline if you run the full workflow.

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

Representative figures are shown in the Gallery below. Full-size files are available in the `plots/` directory.

### Plots Catalog — categorized, captioned, and linked

There are many analysis artifacts in `plots/`. Below is a compact, readable catalog grouped by purpose with short context notes and links to the files. Thumbnails show representative items; use the links to inspect full-size images.

#### Representative thumbnails

- Learning curves
  ![Learning curves](plots/learning_curves.png)
- Confusion matrix
  ![Confusion matrix](plots/confusion_matrix.png)
- Case gallery
  ![Case gallery](plots/case_gallery.png)

#### Catalog (grouped)

- Training dynamics & diagnostics
  - [learning_curves.png](plots/learning_curves.png): epoch-level training & validation accuracy/loss, phase boundaries and best-epoch markers.
  - [training_dynamics.png](plots/training_dynamics.png): extended interval metrics and annotations (analysis export).
  - [calibration_curves.png](plots/calibration_curves.png): predicted-confidence calibration per crop/class.

- Confusion & error analysis
  - [confusion_matrix.png](plots/confusion_matrix.png): overall normalized confusion matrix on the test set.
  - [rice_confusion_matrix.png](plots/rice_confusion_matrix.png): crop-specific confusion focus (rice).

- ROC / PR / calibration
  - `plots/roc_<crop>.png`: ROC curves grouped by crop, with one colored line per disease and AUC values in the legend.

- Class / crop distributions & rankings
  - [class_distribution.png](plots/class_distribution.png): number of training images per class.
  - [crop_distribution.png](plots/crop_distribution.png): crop-level distribution visualisation.
  - [per_class_f1_ranked.png](plots/per_class_f1_ranked.png): per-class F1 ranking and short-list of hard classes.
  - [class_imbalance.png](plots/class_imbalance.png): class imbalance heatmap and summary statistics.

- Misclassifications & galleries
  - [case_gallery.png](plots/case_gallery.png): curated case gallery used for error analysis.
  - [misclassification_gallery.png](plots/misclassification_gallery.png): full misclassification gallery for visual review.
  - [hard_class_gallery.png](plots/hard_class_gallery.png): gallery focused on frequently-misclassified classes.
  - [crop_gallery.png](plots/crop_gallery.png): crop-specific gallery used for per-crop visual inspection.

- Per-class / crop analysis (tables & ranked outputs)
  - [precision_recall_support.png](plots/precision_recall_support.png): per-class precision/recall/support heatmap/table.
  - [top_confusions.png](plots/top_confusions.png): highest-frequency confusion pairs.
  - [top_bottom_classes.png](plots/top_bottom_classes.png): top and bottom performing classes summary.
  - [crop_level_f1.png](plots/crop_level_f1.png): crop-level F1 breakdown.
  - [error_share_by_crop.png](plots/error_share_by_crop.png): error-share analysis across crops.

- Model & system visuals
  - [model_architecture.png](plots/model_architecture.png): schematic of backbone + head.
  - [system_workflow.png](plots/system_workflow.png): system/data/workflow diagram (exported image).
  - [artifact_lineage.png](plots/artifact_lineage.png): file lineage / artifact provenance diagram.

- Misc / example outputs
  - [sample_predictions.png](plots/sample_predictions.png): examples with predicted label + confidence and correctness cue.

> Note: filenames with a common prefix are additional analysis exports — they are produced alongside the main `scripts/generate_figures.py` outputs.

#### How to regenerate and explore

- Regenerate all plots with:

```bash
python scripts/generate_figures.py
```

- The script reads dataset splits in `dataset/` and the selected model via `model_paths.resolve_keras_model_path()`.
- Output files are placed in `plots/`; CSV logs and run manifests are stored in `reports/` and `models/logs/`.

### Diagrams (Mermaid)

Training and system diagrams are rendered with Mermaid for easy editing. The project also includes pre-rendered PNG versions under `plots/`.

```mermaid
flowchart TD
  A["dataset/train, val, test"] --> B["Preprocessing & Augmentations"]
  B --> C["Backbone: EfficientNetV2-S"]
  C --> D["Head training (Phase 1)"]
  D --> E["Unfreeze & Fine-tune (Phase 2)"]
  E --> F["Evaluation & Reports"]
  F --> G["Plots / docs / web UI"]
```

```mermaid
graph LR
  Input[224x224x3] --> EN(EfficientNetV2-S)
  EN --> GAP(GlobalAvgPool2D)
  GAP --> BN(BatchNorm)
  BN --> D1(Dense 512, Swish)
  D1 --> DO1(Dropout 0.4)
  DO1 --> D2(Dense 256, Swish)
  D2 --> DO2(Dropout 0.2)
  DO2 --> OUT(Dense 46, Softmax)
```

```mermaid
sequenceDiagram
  participant U as User
  participant W as Web UI
  participant S as Server (Flask)
  participant M as Model
  U->>W: Upload image
  W->>S: POST /predict
  S->>M: Preprocess & predict
  M-->>S: JSON prediction
  S-->>W: Render results
  W-->>U: Show prediction + advice
```

```mermaid
stateDiagram-v2
  [*] --> Idle
  Idle --> Queued: Run action
  Queued --> Running: Worker started
  Running --> Completed: Exit code 0
  Running --> Failed: Non-zero exit
  Running --> Stopped: Stop requested
  Completed --> [*]
  Failed --> [*]
  Stopped --> [*]
```

```mermaid
flowchart LR
  A["train_model.py"] --> B["models/leaf_disease_checkpoint.keras"]
  B --> C["evaluate_model.py"]
  C --> D["reports/evaluation_report.json"]
  D --> E["scripts/generate_publication_figures.py"]
  E --> F["plots/*.png"]
```

### Math & Equations

Key formulas used in training and reporting:

Cross-entropy with label smoothing (smoothing parameter $\varepsilon$):

$$
\ell(y, \hat{y}) = -\sum_{i=1}^{K} \tilde{y}_i \log \hat{y}_i, \quad
  ilde{y}_i = (1 - \varepsilon) y_i + \frac{\varepsilon}{K}
$$

Cosine annealing learning rate (with period $T$ and step $t$):

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

Warmup (linear) for the first $T_{\text{warmup}}$ steps:

$$
\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}, \quad t \le T_{\text{warmup}}
$$

These formulas are implemented across the `training_utils.py` callbacks and schedule helpers.

## Project Structure

```text
Leaf_Disease_Detection/
├── pyproject.toml              # uv dependencies and metadata
├── uv.lock                     # uv strict dependency lockfile
├── app.py                      # Flask web application
├── main.py                     # Unified task runner
├── config.py                   # Central configuration
├── train_model.py              # Canonical training entrypoint
├── fine_tune_model.py          # Canonical fine-tuning entrypoint
├── evaluate_model.py           # Canonical evaluation entrypoint
├── predict.py                  # Canonical inference CLI/API entrypoint
├── scripts/                    # Plot and image generation scripts only
├── training_utils.py           # Shared training components
├── training_progress.py        # Progress emission callbacks
├── learning_curve_utils.py     # Metric timeline helpers
├── preprocessing.py            # Input preprocessing
├── model_paths.py              # Model path resolution
├── hardware.py                 # GPU/CPU configuration
├── tools/                      # Non-plot utility scripts
│   ├── dataset/                # Dataset utilities
│   └── reporting/              # Reporting/table generation utilities
├── dataset/                    # Train/val/test image data
├── models/                     # Saved model artefacts
├── plots/                      # Generated figures
├── templates/                  # Flask HTML templates
├── tests/                      # Unit tests
└── reports/                    # Evaluation reports and dataset summaries
```

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

## Contributing

- Run the unit tests: `pytest`
- Follow existing code style and add tests for new features.
- If you modify the training pipeline, include run manifests by setting `LEAF_SAVE_RUN_MANIFESTS=1`.

## Known Limitations & Safety

- The model is trained on curated dataset splits and may not generalise to field images without further fine-tuning.
- Do not use predictions as a substitute for expert agronomic or medical advice.
- The dataset and generated models may contain biases; always validate on local samples before deployment.

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
