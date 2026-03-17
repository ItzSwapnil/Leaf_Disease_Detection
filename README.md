# Leaf Disease Detection System

Deep learning-based plant leaf disease classification with EfficientNetV2B0, supporting web inference, CLI inference, and reproducible train/fine-tune/evaluate pipelines.

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21%2B%20CUDA-orange.svg)
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
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Overview

This project provides an end-to-end workflow for plant disease detection:

1. Train and fine-tune an EfficientNetV2B0-based classifier on 46 classes.
2. Evaluate with top-1, top-3, and macro metrics.
3. Serve predictions through a Flask web app and CLI.
4. Generate report-ready visualizations.

```mermaid
flowchart LR
  A[Leaf Image] --> B[Preprocessing]
  B --> C[EfficientNetV2B0 Inference]
  C --> D[Class + Confidence]
  D --> E[Actionable Disease Guidance]
```

## Highlights

- Unified model fallback resolver across app, prediction, evaluation, and visualization.
- Web control panel can trigger training/fine-tuning/evaluation/figure generation jobs.
- Live progress emission with machine-readable progress events.
- CPU and GPU compatible runtime with TensorFlow device detection and strategy selection.
- Multiple interfaces: Flask app, CLI, and Python API.

## Quick Start

### Prerequisites

- Python 3.13+
- Optional but recommended: NVIDIA GPU + CUDA runtime compatible with TensorFlow
- RAM: 8 GB minimum, 16 GB recommended for training

### Setup (Linux / WSL)

```bash
git clone https://github.com/ItzSwapnil/Leaf_Disease_Detection.git
cd Leaf_Disease_Detection
uv venv --python 3.13
source .venv/bin/activate
uv sync
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Setup (Windows PowerShell)

```powershell
git clone https://github.com/ItzSwapnil/Leaf_Disease_Detection.git
cd Leaf_Disease_Detection
uv venv --python 3.13
./.venv/Scripts/Activate.ps1
uv sync
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Usage

### A) Web App (Recommended)

```bash
python app.py
```

Open http://127.0.0.1:5000

Features in the web UI:

- Leaf image upload and prediction
- Disease details (description, symptoms, treatment, prevention)
- Workflow control actions: Train, Fine Tune, Evaluate, Generate Figures
- Job status and logs

### B) Command Runner (Single Entry)

```bash
python main.py serve
python main.py train
python main.py fine_tune
python main.py evaluate
python main.py visualize
```

### C) Dedicated Scripts

```bash
python train_model.py
python fine_tune_model.py
python evaluate_model.py
python generate_figures.py
```

### D) CLI Prediction

```bash
python predict_cli.py --image dataset/test/<Class>/<image>.jpg --top_k 3
python predict_cli.py --image path/to/folder
```

### E) Python API

```python
from predict import LeafDiseasePredictor

predictor = LeafDiseasePredictor()
result = predictor.predict("path/to/leaf_image.jpg", top_k=3)
print(result["disease"], result["confidence"])
```

### Web Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Main UI |
| `/predict` | POST | Image upload + prediction |
| `/health` | GET | App/model health |
| `/control/actions` | GET | List workflow actions |
| `/control/run/<action_key>` | POST | Start workflow job |
| `/control/jobs` | GET | List job history/status |
| `/control/system` | GET | Compute backend info |
| `/control/stop/<job_id>` | POST | Stop running job |

## Model and Training

### Architecture

```text
Input (224x224x3)
  -> EfficientNetV2B0 (ImageNet pretrained, include_top=False)
  -> GlobalAveragePooling2D
  -> BatchNormalization
  -> Dense(1024, relu)
  -> Dropout(0.4)
  -> Dense(46, softmax)
```

### Training Strategy

1. Phase 1: train classification head with frozen backbone.
2. Phase 2: unfreeze top layers and continue fine-tuning.
3. Optional extended fine-tuning via `fine_tune_model.py`.

### Model Artifact Resolution Order

Shared resolver order used by app and scripts:

1. `models/leaf_disease_classifier.keras`
2. `models/leaf_disease_checkpoint.keras`
3. First discovered `.keras` in `models/`

## Dataset

Current local dataset stats (from this repository copy):

- Classes: 46
- Training images: 220,498
- Validation images: 19,419
- Test images: 19,218
- Total images: 259,135

```text
dataset/
├── train/    # 46 class folders
├── val/      # 46 class folders
└── test/     # 46 class folders
```

Supported crop groups include Apple, Tomato, Corn, Grape, Potato, Rice, Pepper, Cherry, Peach, Strawberry, Orange, Wheat, Squash, Blueberry, Raspberry, and Soybean categories.

## Results

| Metric | Value |
| --- | --- |
| Validation Accuracy (recent run) | 97.40% |
| Validation Top-3 Accuracy (recent run) | 99.73% |
| Macro F1 (recent run) | 0.9657 |
| Number of Classes | 46 |

### Visual Outputs

![Learning Curves](plots/learning_curves.png)
![Confusion Matrix](plots/confusion_matrix.png)
![Class Distribution](plots/class_distribution.png)
![Sample Predictions](plots/sample_predictions.png)

## Project Structure

```text
Leaf_Disease_Detection/
├── app.py
├── main.py
├── config.py
├── model_training.py
├── model_fine_tuning.py
├── model_evaluation.py
├── predict.py
├── visualization_pipeline.py
├── train_model.py
├── fine_tune_model.py
├── evaluate_model.py
├── generate_figures.py
├── model_paths.py
├── hardware.py
├── training_progress.py
├── dataset/
├── docs/
│   ├── architecture.md
│   ├── DFD_Level0.md
│   ├── DFD_Level1.md
│   ├── run.md
│   └── reports/
│       ├── report.md
│       ├── report.html
│       └── training-result.md
├── models/
├── plots/
├── templates/
└── tests/
```

## Documentation

- Reproducible run protocol: [docs/run.md](docs/run.md)
- System architecture: [docs/architecture.md](docs/architecture.md)
- DFD context view: [docs/DFD_Level0.md](docs/DFD_Level0.md)
- DFD detailed view: [docs/DFD_Level1.md](docs/DFD_Level1.md)
- Reports: [docs/reports/report.md](docs/reports/report.md), [docs/reports/training-result.md](docs/reports/training-result.md)

## Configuration

Key settings are in `config.py`:

```python
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 46
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 15
LEARNING_RATE_PHASE1 = 0.002
LEARNING_RATE_PHASE2 = 0.0001
UNFREEZE_LAYERS = 50
```

## Testing

Run tests with:

```bash
pytest
```

## Troubleshooting

- Model not found: ensure at least one `.keras` model exists in `models/`.
- GPU not detected: verify TensorFlow/CUDA compatibility and driver installation.
- Upload errors: `/predict` accepts `.jpg`, `.jpeg`, `.png`, `.webp` and max upload size is 16 MB.
- Slow first run on GPU: initial kernel compilation/warm-up can take longer.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a pull request.

## License

Licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- PlantVillage and related agricultural datasets
- EfficientNetV2 research by Google
- TensorFlow and Keras communities

## Contact
Made by Swapnil: [@ItzSwapnil](https://github.com/ItzSwapnil)