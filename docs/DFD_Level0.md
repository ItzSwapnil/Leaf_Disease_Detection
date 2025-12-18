# Data Flow Diagram - Level 0 (Context Diagram)

## 🎯 System Overview

The Level 0 DFD shows the entire Leaf Disease Detection System as a single process with its external entities.

```
                                    ┌─────────────────────┐
                                    │                     │
                                    │       FARMER /      │
                                    │    AGRICULTURAL     │
                                    │     SPECIALIST      │
                                    │                     │
                                    └──────────┬──────────┘
                                               │
                                               │ Leaf Image
                                               │
                                               ▼
┌─────────────────────┐           ┌─────────────────────────────────┐           ┌─────────────────────┐
│                     │           │                                 │           │                     │
│                     │  Training │                                 │ Disease   │                     │
│    IMAGE/TRAINING   │   Data    │      LEAF DISEASE DETECTION     │ Diagnosis │   DISEASE REPORT    │
│       DATABASE      │ ─────────►│           SYSTEM                │──────────►│      OUTPUT         │
│                     │           │                                 │           │                     │
│                     │           │                                 │           │                     │
└─────────────────────┘           └─────────────────────────────────┘           └─────────────────────┘
                                               │
                                               │ Model Updates
                                               │ & Metrics
                                               ▼
                                    ┌─────────────────────┐
                                    │                     │
                                    │    MODEL STORAGE    │
                                    │     (H5 Files)      │
                                    │                     │
                                    └─────────────────────┘
```

## 📊 External Entities

| Entity | Description |
|--------|-------------|
| **Farmer/Agricultural Specialist** | End user who uploads leaf images for disease detection |
| **Image/Training Database** | Repository of labeled plant disease images for training |
| **Disease Report Output** | Generated diagnosis with disease name and confidence score |
| **Model Storage** | Persistent storage for trained model weights |

## 🔄 Data Flows

| Flow | From | To | Description |
|------|------|----|-------------|
| **Leaf Image** | User | System | RGB image of plant leaf for analysis |
| **Training Data** | Database | System | Labeled images for model training |
| **Disease Diagnosis** | System | Report | Predicted disease class with probability |
| **Model Updates** | System | Storage | Trained model weights and checkpoints |

## 📝 Process Description

**Process 0: Leaf Disease Detection System**

The central system that:
1. Receives plant leaf images from users
2. Processes images through a trained CNN model
3. Classifies the leaf into one of 46 disease categories
4. Returns the diagnosis with confidence percentage
5. Can be retrained with new data to improve accuracy

---

*Next: See [DFD_Level1.md](DFD_Level1.md) for detailed process breakdown*
