# 🌿 Plant Leaf Disease Detection System - Quick Reference

## 📁 Files Created

### Training Scripts
- **[train_model.py](train_model.py)** - Main training script using EfficientNetB3 (recommended)
- **[train_model_mobilenet.py](train_model_mobilenet.py)** - Alternative using MobileNetV2 (faster)

### Prediction & Evaluation
- **[predict.py](predict.py)** - Make predictions on new images
- **[evaluate_model.py](evaluate_model.py)** - Comprehensive model evaluation with metrics
- **[check_setup.py](check_setup.py)** - Verify system setup before training

### Utilities
- **[quick_start.sh](quick_start.sh)** - Interactive setup and launch script
- **[model_comparison.py](model_comparison.py)** - Compare models and get recommendations
- **[requirements.txt](requirements.txt)** - Python dependencies
- **[README_MODEL.md](README_MODEL.md)** - Complete documentation

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Setup
```bash
python check_setup.py
```

### Step 3: Train Model
```bash
# Option A: EfficientNetB3 (best accuracy)
python train_model.py

# Option B: MobileNetV2 (faster)
python train_model_mobilenet.py

# Option C: Interactive
./quick_start.sh
```

## 📊 Model Comparison

| Feature | EfficientNetB3 | MobileNetV2 |
|---------|----------------|-------------|
| **Accuracy** | 95-97% | 93-95% |
| **Speed** | Moderate | Fast |
| **Training Time** | 2-4 hours | 1-2 hours |
| **GPU Memory** | ~1.5GB | ~800MB |
| **Image Size** | 300x300 | 224x224 |
| **Best For** | Production | Mobile/Quick tests |

## 🎯 Usage Examples

### Training
```bash
# Train with EfficientNetB3 (recommended for accuracy)
python train_model.py

# Train with MobileNetV2 (faster, good for experiments)
python train_model_mobilenet.py

# Monitor training
tensorboard --logdir logs
```

### Evaluation
```bash
# Evaluate the trained model
python evaluate_model.py

# Output: confusion matrix, classification report, performance metrics
```

### Prediction
```bash
# Predict single image
python predict.py path/to/leaf_image.jpg

# Predict folder of images
python predict.py path/to/image_folder/

# Programmatic usage
python -c "
from predict import LeafDiseasePredictor
predictor = LeafDiseasePredictor()
results = predictor.predict('image.jpg', top_k=3)
print(results)
"
```

## 📈 Expected Results

### Dataset Statistics
- **Total Images**: 260,000+
- **Training**: 220,498 images
- **Validation**: 19,419 images
- **Test**: 19,218 images
- **Classes**: 46 disease categories

### Performance Metrics (EfficientNetB3)
- Test Accuracy: **95-97%**
- Top-3 Accuracy: **98-99%**
- Training Time: **2-4 hours** (GPU)
- Per-class Accuracy: Most >90%

### Performance Metrics (MobileNetV2)
- Test Accuracy: **93-95%**
- Top-3 Accuracy: **97-98%**
- Training Time: **1-2 hours** (GPU)
- Per-class Accuracy: Most >85%

## 🛠️ Advanced Configuration

### Modify Training Parameters

Edit the training script:
```python
IMG_SIZE = 300        # Image size (224, 300, 384)
BATCH_SIZE = 32       # Batch size (16, 32, 64)
EPOCHS = 50          # Maximum epochs
LEARNING_RATE = 0.001 # Initial learning rate
```

### Try Different Models

Replace the base model in training script:
```python
# Options:
from tensorflow.keras.applications import (
    EfficientNetB0,  # Lighter
    EfficientNetB3,  # Balanced (default)
    EfficientNetB4,  # More accurate
    MobileNetV2,     # Fastest
    ResNet50,        # Classic
    InceptionV3      # Another option
)
```

## 📂 Output Files

After training:
```
models/
├── best_model_finetuned.h5     # Best model (EfficientNet)
├── best_model_mobilenet.h5      # Best model (MobileNet)
├── final_model.h5               # Final model after all epochs
└── class_indices.json           # Class name mappings

plots/
├── training_history.png         # Training curves
├── confusion_matrix.png         # Confusion matrix
├── performance_summary.png      # Performance dashboard
└── top_errors.png              # Error analysis

logs/
└── [timestamp]/                # TensorBoard logs
```

## 🔧 Troubleshooting

### Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 16  # or even 8
```

### Slow Training
```bash
# Check GPU usage
nvidia-smi

# Use MobileNetV2
python train_model_mobilenet.py

# Reduce image size
IMG_SIZE = 224
```

### Poor Accuracy
- Train for more epochs
- Try EfficientNetB3 instead of MobileNetV2
- Increase data augmentation
- Fine-tune more layers

## 📚 Documentation

- **[README_MODEL.md](README_MODEL.md)** - Complete documentation
- **[model_comparison.py](model_comparison.py)** - Model comparison details
- **[check_setup.py](check_setup.py)** - System verification

## 🎓 Architecture Overview

### EfficientNetB3 Architecture
```
Input (300x300x3)
    ↓
EfficientNetB3 Base (ImageNet pretrained)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(512, relu) + Dropout(0.5)
    ↓
BatchNormalization
    ↓
Dense(256, relu) + Dropout(0.3)
    ↓
Dense(46, softmax)
```

### Training Strategy
1. **Phase 1** (20 epochs): Train with frozen base model
2. **Phase 2** (30 epochs): Fine-tune top 20% of base layers
3. **Callbacks**: Early stopping, LR reduction, model checkpointing
4. **Augmentation**: Rotation, shift, zoom, flip, brightness

## 💡 Tips for Best Results

1. **Use GPU**: Training on CPU is extremely slow
2. **Monitor Training**: Use TensorBoard to watch progress
3. **Start Small**: Try MobileNetV2 first for quick validation
4. **Scale Up**: Use EfficientNetB3 for final production model
5. **Fine-tune**: Let Phase 2 run to completion for best accuracy
6. **Evaluate Thoroughly**: Use evaluate_model.py to understand performance

## 🎉 Success Indicators

Your model is working well if:
- ✅ Validation accuracy > 90% by epoch 20
- ✅ No significant overfitting (train vs val gap < 5%)
- ✅ Test accuracy matches validation accuracy
- ✅ Most classes have >85% individual accuracy
- ✅ Prediction confidence is high (>80%) for correct predictions

## 📞 Support

Run these for help:
```bash
python check_setup.py      # Verify setup
python model_comparison.py # Compare models
./quick_start.sh          # Interactive guide
```

---

**Ready to train? Pick your model and run:**
```bash
python train_model.py              # EfficientNetB3 (recommended)
python train_model_mobilenet.py    # MobileNetV2 (faster)
```

**Good luck! 🚀🌿**
