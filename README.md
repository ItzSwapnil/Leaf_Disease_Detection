# 🌿 Plant Leaf Disease Detection System

A highly accurate deep learning system for detecting plant diseases from leaf images using transfer learning with state-of-the-art CNN architectures.

## 🚀 Quick Start

### Fast Training (Under 2 Hours) ⚡
```bash
python train_model_fast.py
```
**Perfect for**: Quick results, testing, limited time/resources  
**Accuracy**: 92-95% | **Time**: 60-90 minutes

### Standard Training (Maximum Accuracy) 🎯
```bash
python train_model.py              # EfficientNetB3 (best accuracy)
python train_model_mobilenet.py    # MobileNetV2 (faster alternative)
```
**Perfect for**: Production models, maximum accuracy  
**Accuracy**: 95-97% | **Time**: 3-5 hours

## 📊 Model Comparison

| Model | Script | Accuracy | Time | Use Case |
|-------|--------|----------|------|----------|
| **Fast (NEW)** | `train_model_fast.py` | **92-95%** | **1-1.5h** | **Quick training, testing** |
| MobileNetV2 | `train_model_mobilenet.py` | 93-95% | 2-3h | Balanced option |
| EfficientNetB3 | `train_model.py` | 95-97% | 3-5h | Maximum accuracy |

## 📖 Documentation

- **[Fast Training Guide](README_FAST_TRAINING.md)** - Train in under 2 hours ⚡
- **[Complete Documentation](README_MODEL.md)** - Full system documentation
- **[Quick Reference](QUICK_REFERENCE.md)** - Command reference

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python check_setup.py
```

## 🎯 Features

- ✅ Multiple model options (Fast, MobileNetV2, EfficientNetB3)
- ✅ 46 disease classes across 15+ plant species
- ✅ 260,000+ training images
- ✅ Transfer learning with ImageNet weights
- ✅ Mixed precision training for speed
- ✅ Comprehensive evaluation tools
- ✅ Easy prediction API
- ✅ TensorBoard integration

## 📈 Dataset

- **Training**: 220,498 images
- **Validation**: 19,419 images
- **Test**: 19,218 images
- **Classes**: 46 disease categories
- **Plants**: Apple, Corn, Grape, Tomato, Rice, Potato, and more

## 💻 Usage Examples

### Training
```bash
# Fast training (recommended to start)
python train_model_fast.py

# Standard training
python train_model.py
```

### Prediction
```bash
# Single image
python predict.py path/to/image.jpg

# Batch prediction
python predict.py path/to/folder/
```

### Evaluation
```bash
python evaluate_model.py
```

### Dataset Visualization
```bash
python visualize_dataset.py
```

## 🎓 Choosing the Right Model

### Use Fast Training When:
- ⚡ Need results in under 2 hours
- 🧪 Prototyping and experimentation
- 💻 Limited GPU resources
- 📱 Mobile/edge deployment
- 🔄 Quick iterations needed

### Use Standard Training When:
- 🎯 Need maximum accuracy (95-97%)
- 🏭 Production deployment
- ⏰ Time is not a constraint
- 💪 Have powerful GPU available

## 📊 Expected Performance

| Model | Test Accuracy | Top-3 Accuracy | Training Time |
|-------|--------------|----------------|---------------|
| Fast | 92-95% | 96-98% | 1-1.5 hours |
| MobileNetV2 | 93-95% | 97-98% | 2-3 hours |
| EfficientNetB3 | 95-97% | 98-99% | 3-5 hours |

## 🔍 Project Structure

```
Leaf_Disease_Detection/
├── train_model_fast.py          # Fast training (under 2 hours)
├── train_model.py               # EfficientNetB3 training
├── train_model_mobilenet.py     # MobileNetV2 training
├── predict.py                   # Make predictions
├── evaluate_model.py            # Model evaluation
├── visualize_dataset.py         # Dataset analysis
├── check_setup.py               # Verify setup
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── README_FAST_TRAINING.md      # Fast training guide
├── README_MODEL.md              # Complete documentation
└── QUICK_REFERENCE.md           # Command reference
```

## 🚦 Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Check your setup**
   ```bash
   python check_setup.py
   ```

3. **Choose your training approach**
   ```bash
   # For quick results (recommended)
   python train_model_fast.py
   
   # For maximum accuracy
   python train_model.py
   ```

4. **Evaluate your model**
   ```bash
   python evaluate_model.py
   ```

5. **Make predictions**
   ```bash
   python predict.py test_image.jpg
   ```

## 📚 Additional Resources

- **[Fast Training Guide](README_FAST_TRAINING.md)** - Detailed guide for training under 2 hours
- **[Model Documentation](README_MODEL.md)** - Complete system documentation
- **[Project Summary](PROJECT_SUMMARY.txt)** - Quick overview
- **[Quick Reference](QUICK_REFERENCE.md)** - Command cheat sheet

## 🛠️ System Requirements

### Minimum
- Python 3.8+
- GPU: NVIDIA GTX 1060 (6GB)
- RAM: 8GB
- CUDA 11.0+

### Recommended
- Python 3.9+
- GPU: NVIDIA RTX 2060 (8GB)
- RAM: 16GB
- CUDA 11.8+

## 💡 Tips

- Start with **fast training** to validate your setup
- Use **TensorBoard** to monitor training: `tensorboard --logdir logs/`
- Check **training plots** after each run
- Use the **best_model** files for inference
- Adjust batch size based on your GPU memory

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: PlantVillage Dataset
- Models: EfficientNet (Google), MobileNetV2 (Google)
- Framework: TensorFlow/Keras

---

**Happy Disease Detection! 🌿🔬✨**

For detailed documentation, see [README_MODEL.md](README_MODEL.md)  
For fast training guide, see [README_FAST_TRAINING.md](README_FAST_TRAINING.md)
