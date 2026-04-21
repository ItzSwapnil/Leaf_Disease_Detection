# Leaf Disease Detection

Production-focused plant leaf disease classification with TensorFlow/Keras.
This repository includes end-to-end scripts for training, refinement,
evaluation, safety-guarded inference, and a Flask web UI.

![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue.svg)
![TensorFlow 2.21+](https://img.shields.io/badge/TensorFlow-2.21%2B-orange.svg)
![Keras 3.13+](https://img.shields.io/badge/Keras-3.13%2B-red.svg)
![Classes 46](https://img.shields.io/badge/Classes-46-brightgreen.svg)
![Dataset 259K+](https://img.shields.io/badge/Dataset-259K%2B-blue.svg)
![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Abstract

This repository presents a production-oriented plant leaf disease recognition
system that combines modern deep backbones, calibrated confidence estimates,
and safety-aware inference controls. The workflow spans dataset preparation,
training, post-training refinement, robustness evaluation, and deployment via
CLI and web endpoints. The design objective is not only high classification
performance, but also trustworthy predictions through uncertainty-aware gating
and auditable evaluation artifacts.

## Scientific Highlights

| Dimension | Design Choice | Scientific Rationale |
|---|---|---|
| Representation learning | EfficientNetV2 + DINOv3 backbones | Strong transfer performance across plant-pathology textures |
| Generalization | MixUp, CutMix, RandAugment, class balancing | Reduces overfitting and improves minority-class behavior |
| Reliability | Temperature scaling + calibration metrics | Aligns confidence with empirical correctness |
| Safety | Confidence, entropy, and OOD-style rejection | Mitigates high-risk low-trust predictions |
| Reproducibility | Manifested splits + report artifacts | Enables repeatable experiments and audit trails |

## Overview

This project supports three backbone families and one unified workflow:

- EfficientNetV2 variants (lightweight to larger CNN backbones)
- DINOv3 (ViT-based backbone)
- A shared classification head, calibration pipeline, and safety guard layer

Main workflows included:

- Train from dataset split folders
- Fine-tune and refine checkpoints
- Evaluate calibration, robustness, and uncertainty
- Run predictions from CLI or web app

## Dataset Used

This repository expects an expanded PlantVillage-style dataset called
"PlantVillage-46" in local split folders.

Primary dataset used for this project:

- Mendeley Data (exact dataset package used): https://data.mendeley.com/datasets/32vfdrj76m/1

Related base references (background attribution):

- PlantVillage original dataset (GitHub): https://github.com/spMohanty/PlantVillage-Dataset
- PlantVillage mirror (Kaggle): https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

Important notes:

- The training setup in this repo uses 46 classes (13 healthy + 33 disease)
  and 16 crops.
- Class index mapping is tracked in `models/class_indices.json`.
- Exact reproducibility requires the same class folder names and split layout.

### Dataset Summary Table

| Item | Value |
|---|---|
| Primary dataset | Mendeley Data `32vfdrj76m/1` |
| Dataset URL | https://data.mendeley.com/datasets/32vfdrj76m/1 |
| Total classes | 46 |
| Healthy vs disease classes | 13 healthy / 33 disease |
| Supported crops | 16 |
| Files scanned | 259,135 |
| Unique SHA-1 items | 234,115 |
| Exact duplicates removed | 25,020 |
| Final unique items after dedupe | 234,115 |

Current local corpus stats (from `reports/leakage_free_split_summary.json`):

- Files scanned: 259,135
- Unique SHA-1 items: 234,115
- Exact duplicates removed: 25,020
- Final unique items after dedupe: 234,115

Expected dataset layout:

```text
dataset/
  train/
    <class_name>/
      *.jpg|*.jpeg|*.png|...
  val/
    <class_name>/
      *
  test/
    <class_name>/
      *
```

### Split Summary Table (Leakage-Free Manifest)

| Split | Images | Share of final unique items |
|---|---:|---:|
| Train | 198,975 | 84.99% |
| Validation | 17,536 | 7.49% |
| Test | 17,604 | 7.52% |

Source: `reports/leakage_free_split_summary.json` (`split_counts` totals).

Recommended dataset setup steps:

1. Download and extract the Mendeley dataset archive.
2. Ensure split folders are available as `dataset/train`, `dataset/val`, and `dataset/test`.
3. Ensure each split contains class subfolders with image files.
4. Validate counts and class discovery:

```bash
uv run python tools/dataset/count_dataset.py
```

## Environment Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/ItzSwapnil/Leaf_Disease_Detection.git
cd Leaf_Disease_Detection
uv sync --python 3.13
```

### 2. Verify installation

```bash
uv run python main.py --help
uv run pytest -q
```

### 3. Reproducibility checklist

| Check | Action |
|---|---|
| Dependency lock | Use `uv sync --python 3.13` in a clean workspace |
| Dataset integrity | Run `uv run python tools/dataset/count_dataset.py` |
| Split leakage audit | Run `uv run python tools/dataset/create_leakage_free_split.py` |
| Determinism control | Set `RUN_SEED` when comparing experiments |
| Artifact traceability | Keep `reports/` outputs and `models/logs/` histories |

## How Everything Works (End-to-End)

### 1. Data loading and preprocessing

- Dataset paths come from `config.py`: `dataset/train`, `dataset/val`, `dataset/test`.
- Images are loaded with `keras.utils.image_dataset_from_directory`.
- Preprocessing is routed through `preprocessing.py`.
- Fine-tune, refine, and evaluate flows lock preprocessing to the detected
  loaded-model backbone to avoid mismatch.

### 2. Model construction

- Backbone registry and factories are defined in `backbones.py`.
- `train_model.py` builds the classifier head and attaches it to the selected
  backbone.
- Backbones can be selected from CLI (`--base-model`) or env (`LEAF_BASE_MODEL`).

### 3. Training stages

- Stage A: head warm-up with frozen backbone.
- Stage B: full/partial unfreeze for fine-tuning.
- Optional data balancing and augmentation:
  - class equalizer
  - MixUp / CutMix
  - RandAugment

### 4. Refinement stage

- `refine_model.py` performs post-fine-tune refinement with rolling
  pre-overfit restoration.
- Produces the deployment-ready refined model file.

### 5. Evaluation stage

- `evaluate_model.py` computes:
  - aggregate metrics (accuracy, macro precision/recall/F1)
  - calibration (ECE, temperature scaling)
  - uncertainty and OOD reports
  - robustness suite metrics
- Reports are saved under `reports/` and reliability plots under `plots/`.

### 6. Inference safety and prediction

- `predict.py` applies model inference + safety checks from `inference_guard.py`.
- Safety includes confidence, entropy, and OOD-style gating.
- For incompatible legacy ViT checkpoints, a KerasHub compatibility shim is
  applied automatically.

Formal acceptance rule used by the safety gate:

$$
	ext{accept}(x)=\mathbb{1}\left[p_{\max}(x) \ge \tau_c\;\land\;\frac{H(p(x))}{\log K} \le \tau_h\right]
$$

Where $p_{\max}$ is top-class probability, $H(p)$ is predictive entropy,
$K$ is number of classes, $\tau_c$ is confidence threshold,
and $\tau_h$ is entropy threshold.

### 7. Web serving

- `app.py` starts a Flask app with:
  - upload + prediction endpoint
  - health endpoint
  - control panel endpoints to launch train/fine-tune/refine/evaluate/figures

## Mermaid Diagrams

### End-to-end Training and Deployment Flow

```mermaid
flowchart TD
    A[Download dataset<br/>Mendeley 32vfdrj76m/1] --> B[Arrange folders<br/>dataset/train, dataset/val, dataset/test]
    B --> C[Train<br/>train_model.py]
    C --> D[Fine-tune<br/>fine_tune_model.py]
    D --> E[Refine<br/>refine_model.py]
    E --> F[Evaluate<br/>evaluate_model.py]
    F --> G{Metrics and safety acceptable?}
    G -- Yes --> H[Serve and infer<br/>app.py or predict.py]
    G -- No --> I[Adjust config and retrain]
    I --> C
    H --> J[Generate figures<br/>scripts/generate_figures.py]
```

### Inference Request Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant W as Web UI or CLI
    participant P as Predictor
    participant M as Keras Model
    participant G as Inference Guard

    U->>W: Provide image
    W->>P: Request prediction
    P->>M: model.predict(image)
    M-->>P: Class probabilities
    P->>G: confidence, entropy, OOD checks
    G-->>P: accept or reject
    P-->>W: prediction plus confidence and reason
    W-->>U: final result
```

## Main Scripts and Their Purpose

| Script | Purpose | Typical Output |
|---|---|---|
| `main.py` | Command runner for serve/train/fine_tune/refine/evaluate/visualize | Runs child script and manages logging mode |
| `train_model.py` | Primary training pipeline | checkpoints, logs, class indices, trained model artifacts |
| `fine_tune_model.py` | Continue training from saved checkpoint | updated model + logs |
| `refine_model.py` | Strict overfit-aware refinement | `models/leaf_disease_refined.keras` |
| `evaluate_model.py` | Full evaluation and report generation | `reports/evaluation_report.json`, `reports/evaluation_report.md`, reliability plots |
| `predict.py` | CLI/API inference utility | prediction output and optional visualization |
| `app.py` | Flask web UI + control API | web dashboard at port 5000 |
| `tools/dataset/create_leakage_free_split.py` | dedupe + stratified split generation | leakage manifest and summary JSON |

## Quick Start Commands

### Quick Command Table

| Goal | Command |
|---|---|
| Run web app | `uv run python app.py` |
| Predict one image | `uv run python predict.py --image path/to/leaf.jpg` |
| Predict a directory | `uv run python predict.py --image path/to/folder` |
| Train (default) | `uv run python train_model.py` |
| Train (DINOv3) | `uv run python train_model.py --base-model DINOv3` |
| Fine-tune | `uv run python fine_tune_model.py` |
| Refine | `uv run python refine_model.py --model-path models/leaf_disease_classifier.keras --output-path models/leaf_disease_refined.keras` |
| Evaluate | `uv run python evaluate_model.py --model-path models/leaf_disease_refined.keras` |
| Generate figures | `uv run python scripts/generate_figures.py` |
| Run tests | `uv run pytest -v` |

### Run web app

```bash
uv run python app.py
# Open http://127.0.0.1:5000
```

### Predict a single image

```bash
uv run python predict.py --image path/to/leaf.jpg
```

### Predict a folder

```bash
uv run python predict.py --image path/to/folder
```

Predict CLI options (from `uv run python predict.py --help`):

- `--image`, `-i`: Path to a single image file or a directory of images
- `--model`, `-m`: Path to a saved `.keras` model file
- `--save`, `-s`: Path to save the prediction visualization

### Train

```bash
# Default backbone
uv run python train_model.py

# Explicit backbone
uv run python train_model.py --base-model DINOv3

# Train fraction, optimizer, save mode
uv run python train_model.py \
  --base-model EfficientNetV2B0 \
  --train-fraction 0.25 \
  --optimizer AdamW \
  --save-mode with_optimizer \
  --class-equalizer on
```

### Fine-tune and refine

```bash
uv run python fine_tune_model.py

uv run python refine_model.py \
  --model-path models/leaf_disease_classifier.keras \
  --output-path models/leaf_disease_refined.keras
```

### Evaluate

```bash
# Uses resolver fallback when omitted
uv run python evaluate_model.py

# Explicit model path
uv run python evaluate_model.py --model-path models/leaf_disease_refined.keras
```

### Generate figures

```bash
uv run python scripts/generate_figures.py
uv run python scripts/generate_publication_figures.py
```

### Run tests

```bash
uv run pytest -v
```

## Visual Gallery (Plots)

If plots are missing, generate them first:

```bash
uv run python scripts/generate_figures.py
uv run python scripts/generate_publication_figures.py
```

Core model plots:

| DINOv3 Confusion Matrix | DINOv3 Learning Curves | Backbone Comparison |
|---|---|---|
| [![DINOv3 confusion](plots/DINOv3/confusion_matrix.png)](plots/DINOv3/confusion_matrix.png) | [![DINOv3 learning curves](plots/DINOv3/learning_curves.png)](plots/DINOv3/learning_curves.png) | [![Backbone comparison](plots/DINOv3/ablation_backbone_comparison.png)](plots/DINOv3/ablation_backbone_comparison.png) |

Additional analysis plots:

| Class Distribution | Robustness (Blur) | Robustness (Brightness/Contrast) |
|---|---|---|
| [![Class distribution](plots/others/class_distribution.png)](plots/others/class_distribution.png) | [![Blur robustness](plots/others/robustness_blur_degradation.png)](plots/others/robustness_blur_degradation.png) | [![Brightness contrast robustness](plots/others/robustness_brightness_contrast_matrix.png)](plots/others/robustness_brightness_contrast_matrix.png) |

Backbone comparison examples:

| EfficientNetV2-B0 Confusion | EfficientNetV2-S Learning Curves | DINOv3 Sample Predictions |
|---|---|---|
| [![B0 confusion](plots/EfficientNetV2-B0/confusion_matrix.png)](plots/EfficientNetV2-B0/confusion_matrix.png) | [![V2-S learning curves](plots/EfficientNetV2-S/learning_curves.png)](plots/EfficientNetV2-S/learning_curves.png) | [![DINOv3 sample predictions](plots/DINOv3/sample_predictions.png)](plots/DINOv3/sample_predictions.png) |

## Command Runner (`main.py`)

The command runner exposes these tasks:

```text
serve, train, fine_tune, refine, evaluate, visualize, resume, validate
```

Examples:

```bash
uv run python main.py serve
uv run python main.py train --base-model DINOv3
uv run python main.py evaluate --archive-logs
```

## Dataset Integrity and Leakage Utilities

Count split/class images:

```bash
uv run python tools/dataset/count_dataset.py
```

Create leakage-free split manifest (dry run):

```bash
uv run python tools/dataset/create_leakage_free_split.py
```

Materialize cleaned split tree with hardlinks:

```bash
uv run python tools/dataset/create_leakage_free_split.py \
  --materialize \
  --output-root dataset_clean \
  --prefix-sha1
```

Optional near-duplicate filtering (dHash Hamming <= 1):

```bash
uv run python tools/dataset/create_leakage_free_split.py \
  --dhash-threshold 1 \
  --materialize \
  --output-root dataset_clean
```

Outputs:

- `reports/leakage_free_split_manifest.csv`
- `reports/leakage_free_split_summary.json`

## Control API Endpoints (Flask)

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | UI page |
| `/predict` | POST | Upload image and predict |
| `/health` | GET | Model and app health status |
| `/control/actions` | GET | List available control actions |
| `/control/run/<action>` | POST | Launch workflow job |
| `/control/jobs` | GET | List jobs and status |
| `/control/stop/<job_id>` | POST | Stop active job |

Example:

```bash
curl -X POST -F "file=@leaf.jpg" http://127.0.0.1:5000/predict
```

## Key Environment Variables

| Variable | Purpose |
|---|---|
| `LEAF_BASE_MODEL` | default backbone selection |
| `LEAF_BATCH_SIZE` | training batch size override |
| `LEAF_TRAIN_DATA_FRACTION` | per-class train sampling fraction |
| `LEAF_SAVE_LOG_ARCHIVE` | keep timestamped archives when set to 1 |
| `LEAF_SAVE_RUN_MANIFESTS` | persist run manifests when set to 1 |
| `LEAF_CONFIDENCE_REJECT_THRESHOLD` | inference confidence gate |
| `LEAF_ENTROPY_REJECT_THRESHOLD` | entropy rejection gate |
| `LEAF_ROBUSTNESS_EVAL_ENABLED` | enable robustness suite in evaluation |
| `LEAF_MC_DROPOUT_ENABLED` | enable MC-dropout uncertainty evaluation |

## Where Outputs Go

- Models: `models/`
- Plots: `plots/`
- Logs: `logs/` and `models/logs/`
- Evaluation reports: `reports/evaluation_report.json`, `reports/evaluation_report.md`
- Leakage reports: `reports/leakage_free_split_manifest.csv`,
  `reports/leakage_free_split_summary.json`

## Troubleshooting

- GPU not detected:
  - verify TensorFlow/CUDA compatibility for your GPU and drivers.
- Slow first run on newer NVIDIA GPUs (for example RTX 50xx / compute capability 12.0a):
  - TensorFlow may JIT-compile CUDA kernels from PTX when matching binaries are not bundled.
  - The first GPU run can be significantly slower; subsequent runs are typically faster.
- OOM during training:
  - lower batch size, for example: `LEAF_BATCH_SIZE=8 uv run python train_model.py`.
- Model loading failure for older ViT checkpoint:
  - keep `keras-hub` installed; compatibility shim is applied automatically.
- Missing dataset folders:
  - verify `dataset/train`, `dataset/val`, and `dataset/test` exist and are class-folder structured.

## References

1. Tan, M., and Le, Q. "EfficientNetV2: Smaller Models and Faster Training." ICML 2021.
2. Loshchilov, I., and Hutter, F. "Decoupled Weight Decay Regularization." ICLR 2019.
3. Zhang, H. et al. "mixup: Beyond Empirical Risk Minimization." ICLR 2018.
4. Yun, S. et al. "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features." ICCV 2019.
5. Guo, C. et al. "On Calibration of Modern Neural Networks." ICML 2017.

## License

MIT License. See `LICENSE`.
