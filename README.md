# Leaf Disease Detection System

A deep learning-based plant leaf disease detection system using EfficientNetV2 that identifies 46 plant disease classes across 14 crop types with strong validation performance.

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B%20CUDA-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-97.40%25-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training](#training)
- [Results](#results)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Overview

This project implements an automated plant disease detection workflow with transfer learning. It supports web app inference, script-based prediction, and modular training/fine-tuning/evaluation pipelines.

### System Workflow

```mermaid
flowchart LR
  A[Leaf Image Input] --> B[Preprocessing]
  B --> C[EfficientNetV2 Inference]
  C --> D[Disease Class + Confidence]
  D --> E[Treatment and Prevention Guidance]
```

### Supported Crops and Diseases

| Crop | Diseases Detected |
| ---- | ----------------- |
| Apple | Apple Scab, Black Rot, Brown Spot, Cedar Apple Rust, Grey Spot, Mosaic, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl Virus, Healthy |
| Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Rice | Brown Spot, Leaf Blast, Neck Blast, Healthy |
| Pepper | Bacterial Spot, Healthy |
| Cherry | Powdery Mildew, Healthy |
| Peach | Bacterial Spot, Healthy |
| Strawberry | Leaf Scorch, Healthy |
| Orange | Huanglongbing (Citrus Greening) |
| Wheat | Brown Spot Disease |
| Squash | Powdery Mildew |
| Blueberry, Raspberry, Soybean | Healthy |

## Features

- High accuracy: recent validation run reached 97.40% top-1 and 99.73% top-3.
- Transfer learning: EfficientNetV2B0 pretrained on ImageNet.
- CPU friendly inference path for non-GPU environments.
- Multiple interfaces: Flask web app, CLI, and Python API.
- Visualization pipeline for learning curves and confusion matrix outputs.
- Unified model loading fallback across app/CLI/evaluation/visualization.

## Dataset

The dataset contains approximately 240,000 images split into training, validation, and test sets.

```text
dataset/
├── train/          # Training images (46 classes)
├── val/            # Validation images (46 classes)
└── test/           # Test images (46 classes)
```

## Model Architecture

```text
Input (224x224x3)
  -> EfficientNetV2B0 backbone
  -> GlobalAveragePooling2D
  -> BatchNormalization
  -> Dense(1024, relu)
  -> Dropout(0.4)
  -> Dense(46, softmax)
```

### Training Strategy

1. Phase 1: Freeze base model, train classification head.
2. Phase 2: Unfreeze top layers for in-script fine-tuning.

Optional: run `fine_tune_model.py` for extended fine-tuning from saved weights.

## Installation

### Prerequisites

- Python 3.13+
- NVIDIA GPU and CUDA stack compatible with your TensorFlow build (optional but recommended)
- 8 GB RAM minimum (16 GB recommended)

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

### Graphical Control Panel

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) and use the Workflow Control Panel to run:

- Train Model
- Fine Tune Model
- Evaluate Model
- Generate Figures

The panel shows live status, recent logs, runtime, and lets you stop running jobs.

Model selection for inference/evaluation is automatic and shared across scripts.
Load order is:

1. `models/leaf_disease_classifier.keras`
2. `models/leaf_disease_checkpoint.keras`
3. First discovered `.keras` file under `models/`

### Quick Prediction (Python)

```python
from predict import LeafDiseasePredictor

predictor = LeafDiseasePredictor()
result = predictor.predict("path/to/leaf_image.jpg")
print(result)
```

### Command Line

```bash
python predict_cli.py --image path/to/image.jpg
python evaluate_model.py
```

### Batch Prediction

```python
predictor = LeafDiseasePredictor()
results = predictor.predict_batch("path/to/folder/with/images")
```

## Project Structure

```text
Leaf_Disease_Detection/
├── app.py
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
├── dataset/
├── docs/
├── models/
├── plots/
├── templates/
└── tests/
```

## Training

```bash
python train_model.py
python fine_tune_model.py
python evaluate_model.py
```

## Results

| Metric | Value |
| ------ | ----- |
| Validation Accuracy (recent run) | 97.40% |
| Validation Top-3 Accuracy (recent run) | 99.73% |
| Macro F1 (recent run) | 0.9657 |
| Model Size | ~15 MB |
| Inference Time | ~50ms/image |
| Number of Classes | 46 |

### Result Visualizations

![Learning Curves](plots/learning_curves.png)
![Confusion Matrix](plots/confusion_matrix.png)
![Class Distribution](plots/class_distribution.png)

## Configuration

Configure hyperparameters in `config.py`.

```python
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 46
LEARNING_RATE_PHASE1 = 0.002
LEARNING_RATE_PHASE2 = 0.0001
```

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a pull request.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- PlantVillage and related agricultural datasets.
- EfficientNetV2 research by Google.
- TensorFlow and Keras communities.

## Contact

Swapnil: [@ItzSwapnil](https://github.com/ItzSwapnil)
