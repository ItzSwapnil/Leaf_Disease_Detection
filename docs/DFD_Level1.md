# Data Flow Diagram - Level 1 (Detailed Process View)

## Detailed System Decomposition

Level 1 expands the core system into major internal processes for data preprocessing, model training, storage, and inference.

```mermaid
flowchart TB
    classDef source fill:#f8fafc,stroke:#64748b,color:#0f172a
    classDef proc fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef model fill:#e0e7ff,stroke:#4f46e5,color:#312e81
    classDef store fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef out fill:#fef3c7,stroke:#d97706,color:#7c2d12

    SRC[("Training Dataset")]
    UIN["User Input Image"]

    subgraph P1 ["1.0 Data Preprocessing"]
        P11["1.1 Image Loading"]
        P12["1.2 Resize 224x224"]
        P13["1.3 Normalize / Preprocess"]
        P14["1.4 Data Augmentation"]
        P11 --> P12 --> P13 --> P14
    end

    subgraph P2 ["2.0 Model Training"]
        T1["2.1 Feature Extraction"]
        T2["2.2 Fine-tuning"]
        T3["2.3 Precision Training"]
        T4["2.4 Model Export"]
        T1 --> T2 --> T3 --> T4
    end

    subgraph P3 ["3.0 Model Storage"]
        D1[("Checkpoint Model")]
        D2[("Final Model")]
        D3[("Class Indices")]
    end

    subgraph P4 ["4.0 Inference Pipeline"]
        I1["4.1 Input Preprocessing"]
        I2["4.2 Model Prediction"]
        I3["4.3 Disease Diagnosis"]
        I1 --> I2 --> I3
    end

    RES["Disease + Confidence"]

    SRC --> P11
    P14 --> T1
    T4 --> D1
    T4 --> D2
    T4 --> D3

    UIN --> I1
    D2 --> I2
    D3 --> I3
    I3 --> RES

    class SRC,UIN source
    class P11,P12,P13,P14,T1,T2,T3,T4,I1,I2,I3 proc
    class D1,D2,D3 store
    class RES out
```

---

## Process Descriptions

### 1.0 Data Preprocessing

| Sub-Process | Input | Output | Description |
| ----------- | ----- | ------ | ----------- |
| 1.1 Image Loading | Raw files | Image objects | Loads samples from dataset directories |
| 1.2 Resize | Variable dimensions | 224x224 tensors | Standardizes image size |
| 1.3 Normalize | Pixel values | Model-ready tensors | Applies EfficientNet preprocessing |
| 1.4 Augmentation | Training tensors | Augmented tensors | Improves generalization via transforms |

### 2.0 Model Training

| Sub-Process | Input | Output | Description |
| ----------- | ----- | ------ | ----------- |
| 2.1 Feature Extraction | Preprocessed data | Warm-up weights | Trains head with frozen base |
| 2.2 Fine-tuning | Warm-up weights | Tuned backbone | Unfreezes selected layers |
| 2.3 Precision Training | Tuned backbone | High-accuracy model | Uses low LR for final convergence |
| 2.4 Model Export | Trained weights | Model artifacts | Saves deployable files |

### 3.0 Model Storage

| Data Store | Contents | Purpose |
| ---------- | -------- | ------- |
| D1 | Checkpoint model | Best intermediate validation state |
| D2 | Final model | Production inference model |
| D3 | Class index mapping | Label id to disease name mapping |

### 4.0 Inference Pipeline

| Sub-Process | Input | Output | Description |
| ----------- | ----- | ------ | ----------- |
| 4.1 Input Preprocessing | User image | Model tensor | Aligns input format with training pipeline |
| 4.2 Model Prediction | Preprocessed tensor | Probability vector | Computes class probabilities |
| 4.3 Disease Diagnosis | Probability vector | Disease + confidence | Selects top class and confidence score |

---

## Data Dictionary

| Data Flow | Type | Format | Description |
| --------- | ---- | ------ | ----------- |
| Raw Images | Image | JPEG/PNG | Leaf image samples from users and dataset |
| Preprocessed Tensors | Tensor | (batch, 224, 224, 3) | Normalized model input batches |
| Model Artifacts | Binary | .keras | Saved model weights and topology |
| Prediction Output | Object | JSON-like | Disease name and confidence value |

---

Previous: See [docs/DFD_Level0.md](docs/DFD_Level0.md) for system context.

Related empirical figures: [plots/confusion_matrix.png](../plots/confusion_matrix.png), [plots/sample_predictions.png](../plots/sample_predictions.png).
