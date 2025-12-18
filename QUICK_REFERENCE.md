# 🌿 Plant Leaf Disease Detection System - Quick Reference

## 📁 Files Created

### Training Scripts
- **[train_model_fast.py](train_model_fast.py)** - ⚡ **FAST training under 2 hours** (recommended to start)
- **[train_model.py](train_model.py)** - Main training script using EfficientNetB3 (best accuracy)
- **[train_model_mobilenet.py](train_model_mobilenet.py)** - Alternative using MobileNetV2

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
# Option A: Fast Training - Under 2 hours ⚡ (recommended to start)
python train_model_fast.py

# Option B: EfficientNetB3 (best accuracy)
python train_model.py

# Option C: MobileNetV2 (balanced)
python train_model_mobilenet.py

# Option D: Interactive
./quick_start.sh
```

## 📊 Model Comparison

| Feature | **Fast (NEW)** | EfficientNetB3 | MobileNetV2 |
|---------|----------------|----------------|-------------|
| **Accuracy** | **92-95%** | 95-97% | 93-95% |
| **Speed** | **Very Fast** | Moderate | Fast |
| **Training Time** | **1-1.5 hours** ⚡ | 3-5 hours | 2-3 hours |
| **GPU Memory** | **~4GB** | ~8GB | ~6GB |
| **Image Size** | **160x160** | 300x300 | 224x224 |
| **Epochs** | **15** | 50 | 50 |
| **Best For** | **Quick training** | Production | Balanced |

## 🎯 Usage Examples

### Training
```bash
# Train FAST - Under 2 hours ⚡ (recommended to start)
python train_model_fast.py

# Train with EfficientNetB3 (best accuracy)
python train_model.py

# Train with MobileNetV2 (balanced option)
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
- Training Time: **2-3 hours** (GPU)
- Per-class Accuracy: Most >85%

### Performance Metrics (Fast Training) ⚡
- Test Accuracy: **92-95%**
- Top-3 Accuracy: **96-98%**
- Training Time: **1-1.5 hours** (GPU)
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
├── best_model_fast_finetuned.h5  # Best model (Fast) ⚡
├── best_model_finetuned.h5       # Best model (EfficientNet)
├── best_model_mobilenet.h5       # Best model (MobileNet)
├── final_model_fast.h5           # Final fast model
├── final_model.h5                # Final model after all epochs
├── class_indices_fast.json       # Class mappings (Fast)
├── class_indices.json            # Class name mappings
└── training_stats_fast.json      # Training metrics (Fast)

plots/
├── training_history_fast.png    # Training curves (Fast) ⚡
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

# Use Fast Training (under 2 hours)
python train_model_fast.py

# Use MobileNetV2
python train_model_mobilenet.py

# Reduce image size
IMG_SIZE = 160  # For fast training
```

### Poor Accuracy
- Train for more epochs
- Try EfficientNetB3 instead of MobileNetV2
- Increase data augmentation
- Fine-tune more layers

## 📚 Documentation

- **[README_FAST_TRAINING.md](README_FAST_TRAINING.md)** - ⚡ Fast training guide (under 2 hours)
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
3. **Start Fast**: Try Fast Training first for quick validation (under 2 hours) ⚡
4. **Scale Up**: Use EfficientNetB3 for final production model (best accuracy)
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
python train_model_fast.py         # Fast Training - Under 2 hours ⚡ (recommended)
python train_model.py              # EfficientNetB3 (best accuracy)
python train_model_mobilenet.py    # MobileNetV2 (balanced)
```

**See [README_FAST_TRAINING.md](README_FAST_TRAINING.md) for fast training details!**

**Good luck! 🚀🌿**
