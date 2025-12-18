# 🌿 Leaf Disease Detection System

A deep learning-based plant leaf disease detection system using **EfficientNetV2** that can identify **46 different plant diseases** across 14 crop types with **94.47% accuracy**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-94.47%25-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Training](#-training)
- [Results](#-results)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project implements an automated plant disease detection system using Convolutional Neural Networks (CNNs). The system can analyze images of plant leaves and identify various diseases, helping farmers and agricultural professionals take timely action.

### Supported Crops & Diseases

| Crop | Diseases Detected |
|------|------------------|
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

---

## ✨ Features

- **High Accuracy**: 94.47% validation accuracy on 46 disease classes
- **Transfer Learning**: Uses EfficientNetV2B0 pretrained on ImageNet
- **CPU Optimized**: Runs efficiently on systems without GPU
- **Easy Prediction**: Simple API for single image or batch predictions
- **Visualization Tools**: Generates learning curves, confusion matrices, and class distribution plots

---

## 📊 Dataset

The dataset contains **~240,000 images** split into:
- **Training**: ~70% of data
- **Validation**: ~15% of data  
- **Testing**: ~15% of data

```
dataset/
├── train/          # Training images (46 classes)
├── val/            # Validation images (46 classes)
└── test/           # Test images (46 classes)
```

Each class folder contains images of leaves with that specific condition.

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE (160x160x3)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              EfficientNetV2B0 (ImageNet Pretrained)          │
│                    Feature Extraction                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Global Average Pooling 2D                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Batch Normalization                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Dense Layer (1024 units, ReLU)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Dropout (0.4)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Dense Layer (46 units, Softmax)                 │
│                    Disease Classification                    │
└─────────────────────────────────────────────────────────────┘
```

### Training Strategy
1. **Phase 1**: Freeze base model, train classifier head (10 epochs, LR=0.002)
2. **Phase 2**: Unfreeze top 50 layers, fine-tune (15 epochs, LR=0.0001)
3. **Phase 3**: Resume training with very low LR (1e-6) to reach 99%

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/ItzSwapnil/Leaf_Disease_Detection.git
cd Leaf_Disease_Detection

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 Usage

### Quick Prediction

```python
from predict import LeafDiseasePredictor

# Initialize predictor
predictor = LeafDiseasePredictor()

# Predict on a single image
result = predictor.predict('path/to/leaf_image.jpg')
print(f"Disease: {result['disease']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

### Command Line

```bash
# Predict on a single image
python predict.py --image path/to/image.jpg

# Validate model accuracy
python validation.py
```

### Batch Prediction

```python
predictor = LeafDiseasePredictor()
results = predictor.predict_batch(['image1.jpg', 'image2.jpg', 'image3.jpg'])
```

---

## 📁 Project Structure

```
Leaf_Disease_Detection/
│
├── dataset/                    # Dataset directory
│   ├── train/                  # Training images
│   ├── val/                    # Validation images
│   └── test/                   # Test images
│
├── models/                     # Saved models
│   ├── 1_10th_precision_model.h5   # Previous checkpoint (94.46%)
│   └── 99pct_final_reached.h5      # Best model (94.47%)
│
├── plots/                      # Generated visualizations
│   ├── learning_curves.png
│   ├── confusion_matrix.png
│   └── class_distribution.png
│
├── docs/                       # Documentation
│   ├── DFD_Level0.md
│   ├── DFD_Level1.md
│   └── architecture.md
│
├── config.py                   # Configuration settings
├── train_99pct.py              # Main training script
├── resume_training.py          # Continue training from checkpoint
├── validation.py               # Model validation
├── predict.py                  # Prediction interface
├── generate_visualizations.py  # Create plots and graphs
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🎓 Training

### Train from Scratch

```bash
python train_99pct.py
```

### Resume Training (to reach 99%)

```bash
python resume_training.py
```

### Validate Model

```bash
python validation.py
```

---

## 📈 Results

### Current Model Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 94.47% |
| **Model Size** | ~15 MB |
| **Inference Time** | ~50ms/image |
| **Number of Classes** | 46 |

### Training Progress

The model was trained in multiple phases:
- Phase 1 (Transfer Learning): ~85% accuracy
- Phase 2 (Fine-tuning): ~92% accuracy  
- Phase 3 (Precision Training): ~94.47% accuracy
- Peak during training: 96.50% validation accuracy

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
IMG_SIZE = 160           # Input image size
BATCH_SIZE = 32          # Training batch size
NUM_CLASSES = 46         # Number of disease classes
LEARNING_RATE = 0.002    # Initial learning rate
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset adapted from PlantVillage and other agricultural research datasets
- EfficientNetV2 architecture by Google Research
- TensorFlow and Keras communities

---

## 📧 Contact

**Swapnil** - [@ItzSwapnil](https://github.com/ItzSwapnil)

Project Link: [https://github.com/ItzSwapnil/Leaf_Disease_Detection](https://github.com/ItzSwapnil/Leaf_Disease_Detection)
