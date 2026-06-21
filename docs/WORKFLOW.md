# Leaf Disease Detection Workflow

This document describes the full project workflow from local setup through
training, evaluation, inference, and web serving within the pure PyTorch
architecture ecosystem.

## End-to-End System Diagram

```mermaid
flowchart TD
    %% =========================
    %% Inputs and environment
    %% =========================
    subgraph ENV["Environment and Configuration"]
        OS["Windows host + WSL Bash runtime"]
        UV["uv dependency and command runner"]
        PY["Python 3.14 virtual environment"]
        CFG["src/utils/config.py<br/>paths, flags, thresholds, image size"]
        HW["src/training/train_model.py<br/>PyTorch GPU setup, AMP autocast, memory ceilings"]
        CLI["src/main.py<br/>serve | train | evaluate | visualize"]
        OS --> UV --> PY
        PY --> CFG
        CFG --> HW
        CLI --> CFG
    end

    %% =========================
    %% Dataset and labels
    %% =========================
    subgraph DATA["Dataset and Label Sources"]
        RAW["PlantVillage-style image corpus<br/>Mendeley 32vfdrj76m/1"]
        SPLITS["dataset/train<br/>dataset/val<br/>dataset/test"]
        CLASSDIRS["Class folders<br/>Plant___disease_or_healthy"]
        CLASSIDX["models/class_indices.json<br/>index to class-name mapping"]
        MANIFESTS["Optional reports/manifests<br/>dedupe, leakage checks, split summaries"]
        RAW --> SPLITS --> CLASSDIRS
        CLASSDIRS --> CLASSIDX
        SPLITS --> MANIFESTS
    end

    %% =========================
    %% Model construction
    %% =========================
    subgraph MODEL["Classifier Model Construction"]
        BACKBONES["src/core/backbones.py<br/>EfficientNetV2 variants, DINOv3/ViT"]
        PREPROC["src/core/preprocessing.py<br/>backbone-specific PyTorch transforms"]
        HEAD["PyTorch nn.Module classifier head<br/>AdaptiveAvgPool2d + BatchNorm1d + Linear"]
        LOSS["torch.nn.CrossEntropyLoss"]
        BASEMODEL["PyTorch LeafDiseaseModel"]
        CFG --> BACKBONES
        BACKBONES --> PREPROC --> HEAD --> BASEMODEL
        LOSS --> BASEMODEL
    end

    %% =========================
    %% Training input pipeline
    %% =========================
    subgraph TRAINPIPE["Training Data Pipeline"]
        FILESCAN["PyTorch ImageFolder loaders"]
        DECODE["Decode original RGB image<br/>resize to IMG_SIZE"]
        TORCHDATA["PyTorch DataLoader<br/>Batch generation with prefetching"]
        AUG["Training augmentations<br/>torchvision.transforms.v2<br/>random resized crop, color jitter, flip"]
        LABELS["Disease labels"]
        CLASSDIRS --> FILESCAN --> DECODE
        DECODE --> TORCHDATA
        LABELS --> TORCHDATA --> AUG
    end

    %% =========================
    %% Inference & Serving
    %% =========================
    subgraph INFERENCE["Inference Guard & Output"]
        REQ["Image Request (CLI/Web)"]
        INFER["model(x)<br/>PyTorch TorchScript / nn.Module"]
        PROBS["torch.softmax(logits, dim=1)"]
        GUARD["src/pipeline/inference_guard.py<br/>Confidence Thresholding<br/>Entropy Rejection"]
        OUT["Final Prediction Output"]
        
        REQ --> INFER --> PROBS --> GUARD --> OUT
    end
```

## Step 1: Pre-training Operations

- The user ensures the `dataset/` tree exists (train, val, test splits).
- Configuration (`src/utils/config.py`) loads environmental overrides like VRAM limits, batch sizes, and model targets.
- PyTorch DataLoaders are dynamically built to ingest, resize, and augment the `dataset/train` batches efficiently in the background using `num_workers`.

## Step 2: Training & Calibration

- Model architecture (`src/training/train_model.py`) relies exclusively on the PyTorch standard library and `torchvision`.
- Native 16-bit mixed-precision (`torch.amp.autocast(device_type="cuda")`) enforces hardware optimization and enforces 8GB VRAM budgets.
- Weights are aggressively synced to disk at epoch conclusions, alongside learning rate drops via `CosineAnnealingLR`.

## Step 3: Evaluation & Validation

- Using `uv run leaf-disease-evaluate`, the raw PyTorch model `.pt` file is passed through the evaluation pipeline.
- Validation accuracy, macro F1, and Out-Of-Distribution (OOD) tests are calculated.
- Evaluators output results to `reports/evaluation_report.json` and generate charts in `plots/`.

## Step 4: Safeguarded Inference

- Real-world predictions (via `uv run leaf-disease-predict` or the web UI) are strictly gated.
- Output logits are converted to normalized probabilities via `torch.softmax()`.
- The `inference_guard.py` script applies logical entropy gating and strict confidence thresholds. If a prediction's maximum likelihood falls below the confidence cutoff or registers with excess entropy, it is explicitly flagged and rejected.
