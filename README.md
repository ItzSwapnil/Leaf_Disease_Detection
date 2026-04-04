# Leaf Disease Detection System

> Plant Leaf Disease classification with EfficientNetV2, strong augmentations, and a full train to deploy workflow.

Deep learning-based plant leaf disease classification using EfficientNetV2 transfer learning with SOTA augmentation strategies. Supports web inference, CLI inference, and reproducible train/fine-tune/evaluate pipelines.

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21%2B%20CUDA-orange.svg)
![Keras](https://img.shields.io/badge/Keras-3.13%2B-red.svg)
![Classes](https://img.shields.io/badge/Classes-46-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## At a Glance

| Feature | What you get |
| --- | --- |
| End-to-end workflow | Train, fine-tune, evaluate, visualize, and serve from one codebase |
| Robust training recipe | MixUp, CutMix, label smoothing, cosine schedule, AdamW + EMA |
| Interfaces | Flask web app, CLI, and Python API |
| Dataset scale | 259k+ images, 46 classes, 16 crops |
| Figure outputs | Publication-ready plots + report artifacts in `plots/` and `reports/` |

Quick links: [Quick Start](#quick-start) | [Usage](#usage) | [Results](#results) | [Visual Gallery](#visual-gallery) | [Project Structure](#project-structure)

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

### Visual Gallery

This project generates a large set of analysis plots. Start with these high-signal visuals, then use the catalog to drill down.

<table>
  <tr>
    <td align="center"><b>System Workflow</b></td>
    <td align="center"><b>Training Dynamics</b></td>
    <td align="center"><b>Confusion Matrix</b></td>
  </tr>
  <tr>
    <tr>
    <td><a href="plots/system_workflow.png"><img src="plots/system_workflow.png" alt="system workflow" width="360" height="220"></a></td>
    <td><a href="plots/training_dynamics.png"><img src="plots/training_dynamics.png" alt="training dynamics" width="360" height="220"></a></td>
    <td><a href="plots/confusion_matrix.png"><img src="plots/confusion_matrix.png" alt="confusion matrix" width="360" height="220"></a></td>
  </tr>
  <tr>
    <td align="center"><b>ROC All Crops</b></td>
    <td align="center"><b>Label Smoothing</b></td>
    <td align="center"><b>Class Imbalance</b></td>
  </tr>
  <tr>
    <td><a href="plots/roc_all_crops_compiled.png"><img src="plots/roc_all_crops_compiled.png" alt="roc all crops" width="360" height="220"></a></td>
    <td><a href="plots/label_smoothing.png"><img src="plots/label_smoothing.png" alt="label smoothing" width="360" height="220"></a></td>
    <td><a href="plots/class_imbalance.png"><img src="plots/class_imbalance.png" alt="class imbalance" width="360" height="220"></a></td>
  </tr>
  <tr>
    <td align="center"><b>Sample Predictions</b></td>
    <td align="center"><b>Case Gallery</b></td>
    <td align="center"><b>Crop Distribution</b></td>
  </tr>
  <tr>
    <td><a href="plots/sample_predictions.png"><img src="plots/sample_predictions.png" alt="sample predictions" width="360" height="220"></a></td>
    <td><a href="plots/case_gallery.png"><img src="plots/case_gallery.png" alt="case gallery" width="360" height="220"></a></td>
    <td><a href="plots/crop_distribution.png"><img src="plots/crop_distribution.png" alt="crop distribution" width="360" height="220"></a></td>
  </tr>
</table>

### Plots Catalog

#### Training and optimization

| Plot | Purpose |
| --- | --- |
| [learning_curves.png](plots/learning_curves.png) | Training and validation trends with phase transitions and best-epoch markers |
| [training_dynamics.png](plots/training_dynamics.png) | Extended dynamics view with additional timeline annotations |
| [calibration_curves.png](plots/calibration_curves.png) | Confidence calibration behavior |

#### Confusion and error analysis

| Plot | Purpose |
| --- | --- |
| [confusion_matrix.png](plots/confusion_matrix.png) | Global normalized confusion matrix |
| [rice_confusion_matrix.png](plots/rice_confusion_matrix.png) | Crop-focused confusion matrix for rice classes |
| [top_confusions.png](plots/top_confusions.png) | Most frequent confusion pairs |
| [error_share_by_crop.png](plots/error_share_by_crop.png) | Relative error contribution by crop |

#### ROC and per-class behavior

| Plot | Purpose |
| --- | --- |
| [roc_all_crops_compiled.png](plots/roc_all_crops_compiled.png) | Multi-panel ROC dashboard across all crops |
| [precision_recall_support.png](plots/precision_recall_support.png) | Per-class precision, recall, and support overview |
| [per_class_f1_ranked.png](plots/per_class_f1_ranked.png) | Ranked per-class F1 performance |
| [crop_level_f1.png](plots/crop_level_f1.png) | F1 breakdown at crop level |
| [top_bottom_classes.png](plots/top_bottom_classes.png) | Best and hardest classes summary |

#### Dataset balance and visual inspections

| Plot | Purpose |
| --- | --- |
| [class_distribution.png](plots/class_distribution.png) | Number of samples per class |
| [crop_distribution.png](plots/crop_distribution.png) | Number of samples per crop |
| [class_imbalance.png](plots/class_imbalance.png) | Imbalance profile and summary statistics |
| [crop_gallery.png](plots/crop_gallery.png) | Representative crop-level visual gallery |
| [case_gallery.png](plots/case_gallery.png) | Curated qualitative analysis cases |
| [misclassification_gallery.png](plots/misclassification_gallery.png) | Misclassified sample gallery |
| [hard_class_gallery.png](plots/hard_class_gallery.png) | Gallery focused on difficult classes |
| [sample_predictions.png](plots/sample_predictions.png) | Model predictions with confidence overlays |

#### System and architecture visuals

| Plot | Purpose |
| --- | --- |
| [model_architecture.png](plots/model_architecture.png) | Backbone and classification head schematic |
| [system_workflow.png](plots/system_workflow.png) | End-to-end data and inference workflow |

### Regenerate Figures

```bash
uv run python scripts/generate_figures.py
uv run python scripts/generate_additional_figures.py
```

Outputs are written to `plots/`. Reports and run metadata are written to `reports/` and `models/logs/`.

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
\tilde{y}_i = (1 - \varepsilon) y_i + \frac{\varepsilon}{K}
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
