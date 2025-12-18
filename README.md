# Leaf Disease Detection 🌿🔬

A deep learning system for detecting plant diseases from leaf images with **multiple training options** to suit your needs.

## 🚀 Quick Start

### Choose Your Training Approach:

| Approach | Time | Accuracy | Best For |
|----------|------|----------|----------|
| **🚀 Fast Training** | **1-1.7 hrs** | **90-93%** | **Quick iteration, time-constrained** |
| ⚡ MobileNetV2 | 1-2 hrs | 93-95% | Mobile deployment |
| 🎯 EfficientNetB3 | 2-4 hrs | 95-97% | Maximum accuracy |

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python check_setup.py
```

### Training Options

#### Option 1: Fast Training (Recommended for Quick Results)
**Complete training in under 2 hours with 90-93% accuracy**

```bash
python train_model_fast.py
```

See [FAST_TRAINING_GUIDE.md](FAST_TRAINING_GUIDE.md) for detailed instructions.

#### Option 2: MobileNetV2 (Balanced)
```bash
python train_model_mobilenet.py
```

#### Option 3: EfficientNetB3 (Maximum Accuracy)
```bash
python train_model.py
```

## 📊 Dataset

- **Total Images**: 260,000+
- **Classes**: 46 disease categories
- **Plants**: Apple, Corn, Grape, Tomato, Rice, Potato, and more

## 🔧 Usage

### Make Predictions

```bash
# Single image
python predict.py path/to/image.jpg

# Folder of images
python predict.py path/to/folder/
```

### Evaluate Model

```bash
python evaluate_model.py
```

## 📚 Documentation

- **[FAST_TRAINING_GUIDE.md](FAST_TRAINING_GUIDE.md)** - Complete guide for training in under 2 hours
- **[README_MODEL.md](README_MODEL.md)** - Detailed documentation for all models
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick reference guide

## 🎯 Performance Comparison

| Model | Training Time | Test Accuracy | Model Size | Use Case |
|-------|--------------|---------------|------------|----------|
| MobileNetV3Small (Fast) | 1-1.7 hrs | 90-93% | 10MB | Quick training, iteration |
| MobileNetV2 | 1-2 hrs | 93-95% | 15MB | Mobile deployment |
| EfficientNetB3 | 2-4 hrs | 95-97% | 40MB | Production, max accuracy |

## 💡 New: Fast Training Feature

**Problem**: Standard training taking 4 hours per cycle was too slow for rapid iteration.

**Solution**: New optimized training script that completes in under 2 hours:
- Uses MobileNetV3Small architecture
- Mixed precision training (GPU acceleration)
- Optimized hyperparameters
- Maintains 90-93% accuracy

Perfect for when you need results quickly!

## 🛠️ System Requirements

- **Python**: 3.8+
- **GPU**: Recommended (CUDA-capable)
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ for dataset

## 📦 Output Files

After training, you'll get:
- Trained model files (.h5)
- Class indices (JSON)
- Training history plots (PNG)
- TensorBoard logs
- Training summary (JSON)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

See LICENSE file for details.

---

**Need help?** Check the documentation or open an issue!
