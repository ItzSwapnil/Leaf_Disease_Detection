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
        PRED[predict.py\nInference API]
        EVAL[model_evaluation.py\nEvaluation]
        TRAIN[model_training.py\nTraining]
        FT[model_fine_tuning.py\nFine-tuning]
        VIZ[visualization_pipeline.py\nVisualization]
        MPATH[model_paths.py\nShared Model Resolver]
    end

    subgraph MODEL[Model Layer]
        BASE[EfficientNetV2B0\nFeature Extractor]
        HEAD[Custom Classifier Head\nGAP + BN + Dense + Dropout]
        FINAL[46-class Softmax Output]
    end

    subgraph DATA[Data Layer]
        TR[dataset/train]
        VA[dataset/val]
        TE[dataset/test]
        PRE[Preprocessing\nResize + Normalize + Augment]
    end

    subgraph STORAGE[Storage Layer]
        MSTORE[models/*.keras\nTrained Weights]
        CINDEX[models/class_indices.json]
        PSTORE[plots/*.png]
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

    class APP,MODEL,DATA,STORAGE layer
    class PRED,EVAL,TRAIN,FT,VIZ service
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

    IN[Input\n224x224x3] --> EFF[EfficientNetV2B0\nImageNet Pretrained]
    EFF --> GAP[GlobalAveragePooling2D\n1280 features]
    GAP --> BN[BatchNormalization]
    BN --> D1[Dense 1024 + ReLU]
    D1 --> DO[Dropout 0.4]
    DO --> OUT[Dense 46 + Softmax]

    class IN,EFF,GAP,BN,D1,DO block
    class OUT strong
```

### Parameter Summary

- Base model parameters: approximately 5.9M
- Full model parameters: approximately 7.3M
- Fine-tuned trainable parameters: approximately 2.1M

---

## Training Pipeline

### Core Two-Phase Strategy

```mermaid
flowchart LR
    classDef p1 fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef p2 fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef p3 fill:#fee2e2,stroke:#dc2626,color:#7f1d1d

    P1[Phase 1\nFeature Extraction\nHead Training] --> P2[Phase 2\nIn-script Fine-tuning\nTop Layers Unfrozen]
    P2 --> P3[Optional Extended Fine-tune\nvia fine_tune_model.py]

    class P1 p1
    class P2 p2
    class P3 p3
```

Learning-rate control uses optimizer schedules (cosine restarts). ReduceLROnPlateau is disabled to avoid conflicts with schedule-managed learning rates.

### Data Preprocessing

```mermaid
flowchart LR
    A[Load Image] --> B[Resize to 224x224]
    B --> C[EfficientNet Preprocess\nNormalize to expected range]
    C --> D{Split}
    D --> E[Training\nRotation + Horizontal Flip]
    D --> F[Validation/Test\nNo Augmentation]
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
    API-->>U: Disease + confidence
```

---

## File Responsibilities

| Module | Purpose | Key Outputs |
| ------ | ------- | ----------- |
| `model_training.py` | Primary training workflow | Checkpoint and model artifacts |
| `model_fine_tuning.py` | Resume/precision training | Improved model weights |
| `model_evaluation.py` | Validation and test metrics | Accuracy and evaluation summaries |
| `predict.py` | Inference logic | Disease label and confidence |
| `visualization_pipeline.py` | Plot generation | Confusion matrix, curves, analysis visuals |
| `model_paths.py` | Shared model path resolution | Consistent `.keras` fallback behavior |

---

## Deployment Considerations

### Runtime and Performance

- Training: 8 GB RAM minimum, 16 GB recommended.
- Inference: 2 GB RAM is usually sufficient for single-image predictions.
- CPU threading can be tuned to improve throughput.

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
