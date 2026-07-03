# Leaf Disease Detection

Production-focused plant leaf disease classification powered entirely by PyTorch.
This repository includes end-to-end scripts for training, evaluation, 
safety-guarded inference, and a Flask web UI.

![Python 3.14+](https://img.shields.io/badge/Python-3.14%2B-blue.svg)
![PyTorch Nightly](https://img.shields.io/badge/PyTorch-Nightly-orange.svg)
![Keras 3 (Torch Backend)](https://img.shields.io/badge/Keras--3-PyTorch--Backend-red.svg)
![Classes 29](https://img.shields.io/badge/Classes-29-brightgreen.svg)
![Dataset 8.8K+](https://img.shields.io/badge/Dataset-8.8K%2B-blue.svg)
![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Abstract

This repository presents a production-oriented plant leaf disease recognition
system that combines modern deep backbones, calibrated confidence estimates,
and safety-aware inference controls. The workflow spans dataset preparation,
training, robustness evaluation, and deployment via CLI and web endpoints. 
The design objective is not only high classification performance but also 
trustworthy predictions through uncertainty-aware gating and auditable artifacts, 
while strictly adhering to an 8GB VRAM compute ceiling.

## Scientific Highlights

| Dimension | Design Choice | Scientific Rationale |
|---|---|---|
| Representation learning | EfficientNetV2 + DINOv3 backbones | Strong transfer performance across plant-pathology textures |
| Hardware Constraints | `torch.amp.autocast` Mixed Precision | Aggressively optimizes throughput under 8GB VRAM limits |
| Reliability | Temperature scaling + calibration metrics | Aligns confidence with empirical correctness |
| Safety | Confidence, entropy, and OOD-style rejection | Mitigates high-risk low-trust predictions |
| Type Safety | Strict Mypy + Ruff | Ensures runtime stability and memory safety across pipelines |

## Overview

This project supports three backbone families and one unified workflow:

- EfficientNetV2 variants (lightweight to larger CNN backbones)
- DINOv3 (ViT-based backbone)
- A shared PyTorch `nn.Module` classification head, calibration pipeline, and safety guard layer

Main workflows included:

- Train from dataset split folders using PyTorch DataLoaders
- Evaluate calibration, robustness, and uncertainty
- Run predictions from CLI or web app

## Dataset Used

This repository expects the **PlantDoc** dataset (*"PlantDoc: A Dataset for Visual Plant Disease Detection"*, Singh et al., 2019) structured into local split folders.

Primary dataset used for this project:

- PlantDoc (Visual Plant Disease Detection): [Singh et al., 2019](https://github.com/daved01/PlantDoc-Dataset)

Important notes:

- The training setup uses **29 classes** (covering 13 unique crop families).
- Class index mapping is dynamically generated and tracked in [class_indices.json](file:///mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/class_indices.json).
- The dataset is partitioned into proper, stratified splits: **Train (80%)**, **Validation (10%)**, and **Test (10%)**.

Expected dataset layout:

```text
dataset/
  train/
    <class_name>/
      *.jpg|*.jpeg|*.png
  val/
    <class_name>/
      *.jpg|*.jpeg|*.png
  test/
    <class_name>/
      *.jpg|*.jpeg|*.png
```

## Environment Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/ItzSwapnil/Leaf_Disease_Detection.git
cd Leaf_Disease_Detection
uv sync --prerelease allow
```

### 2. Verify installation

```bash
uv run python src/main.py --help
uv run pytest -v
uv run mypy .
```

### 3. Reproducibility checklist

| Check | Action |
|---|---|
| Dependency lock | Use `uv sync` |
| Split leakage audit | Run `uv run python tools/dataset/create_leakage_free_split.py` |
| Determinism control | Set `RUN_SEED` when comparing experiments |
| Artifact traceability | Keep `reports/` outputs and `models/logs/` histories |

## How Everything Works (End-to-End)

### 1. Data loading and preprocessing

- Dataset paths come from `config.py`: `dataset/train`, `dataset/val`, `dataset/test`.
- Images are loaded natively using pure PyTorch `DataLoaders`.
- Preprocessing is routed through `src/core/preprocessing.py`.

### 2. Model construction

- Backbone registry and factories are defined in `src/core/backbones.py`.
- `src/training/train_model.py` builds the classifier head as an `nn.Module` and attaches it to the backbone.
- Backbones can be selected from CLI (`--base-model`).

### 3. Training stage

- End-to-end training utilizing PyTorch `AdamW` optimizers and cosine annealing schedulers.
- Dynamic VRAM usage is strictly throttled to 8GB ceilings via `torch.amp.autocast`.
- Automatic memory garbage collection triggers routinely between epochs to prevent OOM errors.

### 4. Evaluation stage

- `src/evaluation/evaluate_model.py` computes:
  - aggregate metrics (accuracy, macro precision/recall/F1)
  - calibration (ECE, temperature scaling)
  - uncertainty and OOD reports
  - robustness suite metrics
- Reports are saved under `reports/` and reliability plots under `plots/`.

### 5. Inference safety and prediction

- `src/pipeline/predict.py` applies model inference + safety checks from `src/pipeline/inference_guard.py`.
- Safety includes confidence, entropy, and OOD-style gating.

Formal acceptance rule used by the safety gate:

$$
	ext{accept}(x)=\mathbb{1}\left[p_{\max}(x) \ge \tau_c\;\land\;\frac{H(p(x))}{\log K} \le \tau_h\right]
$$

Where $p_{\max}$ is top-class probability, $H(p)$ is predictive entropy,
$K$ is number of classes, $\tau_c$ is confidence threshold,
and $\tau_h$ is entropy threshold.

### 6. Web serving

- `src/web/app.py` starts a Flask app with:
  - upload + prediction endpoint
  - health endpoint
  - control panel endpoints to launch train/evaluate/visualize pipelines

## Mermaid Diagrams

For the full system workflow, including training, evaluation, inference, web serving, safety gates, and artifact flow, see [docs/WORKFLOW.md](docs/WORKFLOW.md).

## Main Scripts and Their Purpose

| Script | Purpose | Typical Output |
|---|---|---|
| `src/main.py` | Command runner | Runs child script and manages logging mode |
| `src/training/train_model.py` | Primary training pipeline | Checkpoints, logs, trained model `.pt` files |
| `src/evaluation/evaluate_model.py` | Full PyTorch evaluation pipeline | `reports/evaluation_report.json`, reliability plots |
| `src/pipeline/predict.py` | CLI/API inference utility | prediction output and optional visualization |
| `src/web/app.py` | Flask web UI + control API | web dashboard at port 5000 |

## Quick Start Commands

### Quick Command Table

| Goal | Command |
|---|---|
| Run web app | `uv run python src/web/app.py` |
| Predict one image | `uv run leaf-disease-predict --image path/to/leaf.jpg` |
| Train (default) | `uv run python src/training/train_model.py` |
| Train (DINOv3) | `uv run python src/training/train_model.py --base-model dinov3_vits14` |
| Evaluate | `uv run leaf-disease-evaluate --model-path models/leaf_disease_classifier.pt` |
| Generate figures | `uv run leaf-disease-figures` |
| Run tests | `uv run pytest -v` |
| Strict type checking | `uv run mypy .` |

### Predict a single image

```bash
uv run leaf-disease-predict --image path/to/leaf.jpg
```

Predict CLI options (from `uv run leaf-disease-predict --help`):

- `--image`, `-i`: Path to a single image file or a directory of images
- `--model`, `-m`: Path to a saved `.pt` model file
- `--save`, `-s`: Path to save the prediction visualization

### Train

```bash
# Default backbone
uv run python src/training/train_model.py

# Explicit backbone
uv run python src/training/train_model.py --base-model dinov3_vits14

# Specific parameters
uv run python src/training/train_model.py \
  --base-model EfficientNetV2B0 \
  --train-fraction 0.25 \
  --optimizer AdamW
```

### Evaluate

```bash
# Explicit model path
uv run leaf-disease-evaluate --model-path models/leaf_disease_classifier.pt
```

### Generate figures

```bash
uv run leaf-disease-figures
```

## Where Outputs Go

- Models: `models/`
- Plots: `plots/`
- Logs: `logs/` and `models/logs/`
- Evaluation reports: `reports/`

## Troubleshooting

- GPU not detected:
  - Verify CUDA toolkit installation matches PyTorch Nightly (`cu132`).
- OOM during training:
  - You are restricted to an 8GB ceiling. Automatic GC handles most issues, but manually lower batch size if OOM persists.
- Missing dataset folders:
  - verify `dataset/train`, `dataset/val`, and `dataset/test` exist and are class-folder structured.

## License

MIT License. See `LICENSE`.
