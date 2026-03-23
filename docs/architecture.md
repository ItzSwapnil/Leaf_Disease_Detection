# System Architecture Documentation

## Overall Architecture

This document describes the end-to-end architecture for training, evaluating, and serving the leaf disease detection model.

```mermaid
flowchart TB
    classDef layer fill:#0f172a,stroke:#1e293b,color:#f8fafc,stroke-width:1.5px
    classDef service fill:#e2e8f0,stroke:#64748b,color:#0f172a
    classDef data fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef model fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef output fill:#fef3c7,stroke:#d97706,color:#7c2d12

    subgraph APP[Application Layer]
        PRED["predict.py — Inference API"]
        EVAL["model_evaluation.py — Evaluation"]
        TRAIN["model_training.py — Training"]
        FT["model_fine_tuning.py — Fine-tuning"]
        VIZ["visualization_pipeline.py — Visualization"]
        MPATH["model_paths.py — Shared Model Resolver"]
        TUTILS["training_utils.py — Shared Training Components"]
    end

    subgraph MODEL[Model Layer]
        BASE["EfficientNetV2-S — Feature Extractor"]
        HEAD["Classification Head — GAP + BN + Dense + Dropout"]
        FINAL["46-class Softmax Output"]
    end

    subgraph DATA[Data Layer]
        TR["dataset/train"]
        VA["dataset/val"]
        TE["dataset/test"]
        PRE["Preprocessing — Resize + Normalize + Augment"]
    end

    subgraph STORAGE[Storage Layer]
        MSTORE["models/*.keras — Trained Weights"]
        CINDEX["models/class_indices.json"]
        PSTORE["plots/*.png"]
    end

    TR --> PRE
    VA --> PRE
    TE --> PRE
    PRE --> TRAIN
    PRE --> FT
    PRE --> EVAL
    TRAIN --> BASE --> HEAD --> FINAL
    FT --> BASE
    EVAL --> FINAL
    PRED --> FINAL
    VIZ --> FINAL

    FINAL --> MSTORE
    FINAL --> CINDEX
    VIZ --> PSTORE
    MPATH --> MSTORE
    TUTILS --> TRAIN
    TUTILS --> FT

    class APP,MODEL,DATA,STORAGE layer
    class PRED,EVAL,TRAIN,FT,VIZ,TUTILS service
    class TR,VA,TE,PRE data
    class BASE,HEAD,FINAL model
    class MSTORE,CINDEX,PSTORE output
```

---

## Neural Network Architecture

### Backbone + Classification Head

```mermaid
flowchart LR
    classDef block fill:#f8fafc,stroke:#475569,color:#0f172a
    classDef strong fill:#bfdbfe,stroke:#1d4ed8,color:#1e3a8a,stroke-width:2px

    IN["Input 224x224x3"] --> EFF["EfficientNetV2-S (ImageNet Pretrained)"]
    EFF --> GAP["GlobalAveragePooling2D (1280)"]
    GAP --> BN["BatchNormalization"]
    BN --> D1["Dense 512 + Swish"]
    D1 --> DO1["Dropout (0.4)"]
    DO1 --> D2["Dense 256 + Swish"]
    D2 --> DO2["Dropout (0.2)"]
    DO2 --> OUT["Dense 46 + Softmax"]

    class IN,EFF,GAP,BN,D1,DO1,D2,DO2 block
    class OUT strong
```

### Parameter Summary

- Base model (EfficientNetV2-S) parameters: ~20.2M
- Full model parameters: ~21.4M
- Phase 1 trainable parameters (head only): ~1.2M
- Phase 2 trainable parameters (full model): ~21.4M

---

## Training Pipeline

### Core Two-Phase Strategy

```mermaid
flowchart LR
    classDef p1 fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef p2 fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef p3 fill:#fee2e2,stroke:#dc2626,color:#7f1d1d

    P1["Phase 1 — Head-Only Warmup (5 epochs)"] --> P2["Phase 2 — Full Fine-Tuning (10 epochs)"]
    P2 --> P3["Optional Extended Fine-Tune (model_fine_tuning.py)"]

    class P1 p1
    class P2 p2
    class P3 p3
```

Learning rate control uses warmup + cosine annealing schedules (Loshchilov & Hutter, 2017).

### SOTA Augmentation

| Technique | Description |
| --- | --- |
| ImageDataGenerator | Rotation, flip, shift, zoom, brightness, shear |
| MixUp (Zhang et al., 2018) | Convex combination of images and labels |
| CutMix (Yun et al., 2019) | Random region cut-and-paste between images |
| Label smoothing (0.1) | Soft target regularisation |

### Data Preprocessing

```mermaid
flowchart LR
    A["Load Image"] --> B["Resize to 224x224"]
    B --> C["EfficientNet preprocess_input"]
    C --> D{"Split"}
    D --> E["Training — Augment + MixUp/CutMix"]
    D --> F["Validation/Test — No augmentation"]
```

---

## Inference Pipeline

```mermaid
sequenceDiagram
    participant U as User
    participant API as Flask API
    participant PRE as Preprocessing
    participant MOD as Trained Model
    participant MAP as Class Mapping

    U->>API: Upload leaf image
    API->>PRE: Validate + resize + normalize
    PRE->>MOD: Tensor (1, 224, 224, 3)
    MOD-->>API: Probability vector (46 classes)
    API->>MAP: Resolve class id to label
    MAP-->>API: Disease label
    API-->>U: Disease + confidence + guidance
```

---

## File Responsibilities

| Module | Purpose | Key Outputs |
| --- | --- | --- |
| `model_training.py` | Primary two-phase training workflow | Checkpoint and model artifacts |
| `model_fine_tuning.py` | Resume/precision training from checkpoint | Improved model weights |
| `model_evaluation.py` | Validation metrics and error analysis | JSON/Markdown reports |
| `predict.py` | Inference API and CLI | Disease label and confidence |
| `visualization_pipeline.py` | Standard figure generation | Confusion matrix, curves, sample predictions |
| `training_utils.py` | Shared training components (LR, loss, augmentation) | Used by all training scripts |
| `model_paths.py` | Shared model path resolution | Consistent `.keras` fallback behavior |
| `preprocessing.py` | Backbone-aware input normalisation | Used by all inference paths |
| `hardware.py` | GPU detection and distribution strategy | Used at startup by all scripts |

---

## Deployment Considerations

### Runtime and Performance

- Training: 8 GB system RAM minimum, 16 GB recommended. NVIDIA GPU with 8+ GB VRAM.
- Inference: 2 GB RAM is usually sufficient for single-image predictions.
- CPU threading can be tuned via `INTRA_OP_THREADS` / `INTER_OP_THREADS` in `config.py`.

### GPU Notes

- Only NVIDIA GPUs (CUDA) are supported by TensorFlow's GPU backend.
- AMD integrated GPUs (e.g., Radeon 860M) are not visible to TensorFlow.
- Mixed precision (`float16`) is enabled automatically when a GPU is detected.

### Scalability

- Suitable for Flask API deployment behind a reverse proxy.
- Can be containerized for cloud or edge environments.
- Model export path supports conversion workflows such as TensorFlow Lite.

---

## Empirical Visualizations

The following generated figures provide empirical support for training behavior and classification quality.

![Learning Curves](../plots/learning_curves.png)
![Confusion Matrix](../plots/confusion_matrix.png)
![Model Architecture](../plots/model_architecture.png)
