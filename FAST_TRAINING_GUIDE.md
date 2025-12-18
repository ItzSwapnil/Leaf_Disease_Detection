# 🚀 Fast Training Guide - Under 2 Hours

## Overview

This guide explains how to train the Leaf Disease Detection model in **under 2 hours** while maintaining **high accuracy (>90%)**.

## Problem

The standard training approaches were taking **4 hours per cycle**, which is too long for rapid iteration and testing.

## Solution

We've created an optimized training script (`train_model_fast.py`) that completes all training cycles in under 2 hours through several key optimizations:

### Key Optimizations

1. **Lighter Architecture**: MobileNetV3Small
   - Faster than EfficientNetB3 and MobileNetV2
   - Significantly fewer parameters (~2M vs 12M)
   - Still maintains high accuracy (>90%)

2. **Mixed Precision Training**
   - Uses float16 for computations (float32 for final layer)
   - 2-3x faster on modern GPUs
   - Requires GPU with compute capability >= 7.0

3. **Optimized Image Size**
   - 224x224 instead of 300x300
   - 44% fewer pixels to process
   - Faster data loading and augmentation

4. **Larger Batch Size**
   - Increased from 32 to 64
   - Better GPU utilization
   - Fewer iterations per epoch

5. **Reduced Epochs with Early Stopping**
   - 30 max epochs instead of 50
   - Aggressive early stopping (patience=5)
   - Stops automatically when performance plateaus

6. **Streamlined Data Augmentation**
   - Less aggressive augmentation
   - Faster preprocessing
   - Still maintains generalization

7. **Efficient Callbacks**
   - Reduced learning rate patience (2 instead of 3-4)
   - Disabled histogram logging in TensorBoard
   - Optimized checkpoint saving

## Quick Start

### Prerequisites

```bash
# Ensure you have a CUDA-capable GPU
nvidia-smi

# Install dependencies
pip install -r requirements.txt
```

### Run Fast Training

```bash
python train_model_fast.py
```

That's it! The script will:
1. ✅ Load and prepare data with optimized settings
2. ✅ Build the fast MobileNetV3Small model
3. ✅ Train Phase 1: Frozen base model (~30-40 minutes)
4. ✅ Train Phase 2: Fine-tuning (~30-40 minutes)
5. ✅ Evaluate on test set
6. ✅ Save models and generate reports

## Expected Results

### Training Time
- **Total**: 60-100 minutes (1-1.7 hours)
- **Phase 1**: 30-40 minutes
- **Phase 2**: 30-40 minutes

### Accuracy
- **Test Accuracy**: 90-93%
- **Top-3 Accuracy**: 96-98%
- **Most classes**: >85% accuracy

### Model Size
- **Parameters**: ~2.5M (vs 12M for EfficientNetB3)
- **File Size**: ~10MB (vs 40MB for EfficientNetB3)

## Output Files

After training completes, you'll find:

```
models/
├── best_model_fast_finetuned.h5     # Best model (use this for predictions)
├── final_model_fast.h5               # Final model after all epochs
├── class_indices_fast.json           # Class name mappings
└── training_summary_fast.json        # Training statistics

plots/
└── training_history_fast.png         # Training curves with time/accuracy

logs/
└── fast_[timestamp]/                 # TensorBoard logs
```

## Configuration Options

You can modify these settings in `train_model_fast.py`:

```python
# Core settings
IMG_SIZE = 224          # Image size (224 recommended)
BATCH_SIZE = 64         # Batch size (increase if you have more GPU memory)
EPOCHS = 30            # Maximum epochs (early stopping will likely stop sooner)
LEARNING_RATE = 0.002  # Initial learning rate
```

### For Even Faster Training

If you need to train even faster and can accept slightly lower accuracy:

```python
IMG_SIZE = 192          # Smaller images
BATCH_SIZE = 96         # Larger batches (if GPU memory allows)
EPOCHS = 25            # Fewer max epochs
```

### For Better Accuracy

If you have a bit more time and want higher accuracy:

```python
IMG_SIZE = 224          # Keep standard size
BATCH_SIZE = 48         # Smaller batch (more updates)
EPOCHS = 40            # More epochs
LEARNING_RATE = 0.001  # Lower learning rate
```

## Comparison with Other Models

| Model | Training Time | Accuracy | Parameters | Best For |
|-------|--------------|----------|------------|----------|
| **MobileNetV3Small (Fast)** | **1-1.7 hrs** | **90-93%** | **2.5M** | **Quick training** |
| MobileNetV2 | 1-2 hrs | 93-95% | 3.5M | Mobile deployment |
| EfficientNetB3 | 2-4 hrs | 95-97% | 12M | Maximum accuracy |

## Using the Trained Model

### For Predictions

```bash
# Single image
python predict.py path/to/image.jpg --model models/best_model_fast_finetuned.h5 --classes models/class_indices_fast.json

# Multiple images
python predict.py path/to/folder/ --model models/best_model_fast_finetuned.h5 --classes models/class_indices_fast.json
```

### In Python Code

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# Load model and class indices
model = load_model('models/best_model_fast_finetuned.h5')
with open('models/class_indices_fast.json', 'r') as f:
    class_indices = json.load(f)
    
# Reverse mapping
idx_to_class = {v: k for k, v in class_indices.items()}

# Predict
img = image.load_img('test_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
top_pred_idx = np.argmax(predictions[0])
confidence = predictions[0][top_pred_idx]

print(f"Prediction: {idx_to_class[top_pred_idx]}")
print(f"Confidence: {confidence*100:.2f}%")
```

## Troubleshooting

### "Out of memory" error

Reduce batch size:
```python
BATCH_SIZE = 32  # or even 16
```

### Mixed precision not working

Check GPU compatibility:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

If compute capability < 7.0, the script will fall back to float32 (slightly slower but still fast).

### Training is still slow

1. **Check GPU is being used**:
   ```bash
   nvidia-smi
   # Should show GPU utilization during training
   ```

2. **Verify TensorFlow sees GPU**:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

3. **Use smaller image size**:
   ```python
   IMG_SIZE = 192  # or even 160
   ```

### Accuracy is lower than expected

1. **Train for a few more epochs** (modify EPOCHS)
2. **Try slightly lower learning rate** (0.001 instead of 0.002)
3. **Increase batch size** if GPU memory allows
4. **Use the standard MobileNetV2 or EfficientNetB3** if you can spare the extra time

## Tips for Best Results

1. ✅ **Use a GPU** - Training on CPU will take 10-20x longer
2. ✅ **Monitor with TensorBoard** - `tensorboard --logdir logs/`
3. ✅ **Let it complete both phases** - Fine-tuning is important
4. ✅ **Check training summary** - Review `training_summary_fast.json` for statistics
5. ✅ **Validate on test set** - Run `evaluate_model.py` for detailed metrics

## When to Use Fast Training

✅ **Use fast training when:**
- You need to iterate quickly
- You're experimenting with different approaches
- Training time is critical
- 90-93% accuracy is sufficient
- You need a model for mobile/edge deployment

❌ **Use standard training (EfficientNetB3) when:**
- Maximum accuracy is critical (95-97%)
- Training time is not a constraint
- You're building a production system
- You need the absolute best performance

## Advanced: Benchmarking

To benchmark your system:

```bash
# Run fast training with timing
time python train_model_fast.py

# Check GPU utilization during training
watch -n 1 nvidia-smi
```

The script automatically tracks and reports:
- Phase 1 time
- Phase 2 time
- Total training time
- Whether 2-hour target was met

## Summary

The fast training approach achieves:
- ⚡ **60-100 minute total training time**
- 🎯 **90-93% test accuracy**
- 📦 **Compact 10MB model**
- 🚀 **Perfect for rapid iteration**

This makes it ideal when you need to train quickly while still maintaining high accuracy for leaf disease detection.

---

**Questions?** Open an issue or check the main [README_MODEL.md](README_MODEL.md) for more details.
