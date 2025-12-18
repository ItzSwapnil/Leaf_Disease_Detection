# Plant Leaf Disease Detection System 🌿🔬

A highly accurate deep learning system for detecting plant diseases from leaf images using transfer learning with EfficientNetB3.

## Features ✨

- **High Accuracy**: Uses EfficientNetB3 with transfer learning for superior performance
- **46 Disease Classes**: Detects diseases across multiple plant species (Apple, Corn, Grape, Tomato, Rice, etc.)
- **Two-Phase Training**: Initial training with frozen base + fine-tuning for optimal results
- **Data Augmentation**: Extensive augmentation for robust generalization
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and performance analysis
- **Easy Prediction**: Simple API for predicting on new images

## Dataset 📊

The dataset contains **260,000+ images** across 46 disease classes:

- **Training**: 220,498 images
- **Validation**: 19,419 images  
- **Test**: 19,218 images

### Supported Plants & Diseases

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Rice, Soybean, Squash, Strawberry, Tomato, Wheat

## Installation 🔧

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Leaf_Disease_Detection

# Install dependencies
pip install -r requirements.txt
```

## Usage 🚀

### 1. Train the Model

```bash
python train_model.py
```

**Training Details:**
- **Phase 1**: 20 epochs with frozen EfficientNetB3 base
- **Phase 2**: 30 epochs with fine-tuning top 20% of base layers
- **Image Size**: 300x300 pixels
- **Batch Size**: 32
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

**Expected Training Time:**
- With GPU: ~2-4 hours
- With CPU: ~20-30 hours (not recommended)

**Output:**
- `models/best_model_finetuned.h5` - Best model based on validation accuracy
- `models/final_model.h5` - Final model after all epochs
- `models/class_indices.json` - Class name mappings
- `plots/training_history.png` - Training curves
- `logs/` - TensorBoard logs

### 2. Evaluate the Model

```bash
python evaluate_model.py
```

**Output:**
- Confusion matrix visualization
- Per-class accuracy analysis
- Classification report with precision, recall, F1-score
- Error analysis showing challenging classes
- Performance summary dashboard

### 3. Make Predictions

#### Single Image Prediction

```bash
python predict.py path/to/your/image.jpg
```

#### Batch Prediction

```bash
python predict.py path/to/image/folder/
```

#### Programmatic Usage

```python
from predict import LeafDiseasePredictor

# Initialize predictor
predictor = LeafDiseasePredictor(
    model_path='models/best_model_finetuned.h5',
    class_indices_path='models/class_indices.json'
)

# Predict and visualize
predictor.predict_and_visualize('test_image.jpg', top_k=3)

# Get prediction results
results = predictor.predict('test_image.jpg', top_k=5)
print(results)
```

## Model Architecture 🏗️

```
EfficientNetB3 (Pre-trained on ImageNet)
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

**Key Features:**
- **Base Model**: EfficientNetB3 (more accurate than MobileNetV2)
- **Transfer Learning**: Pre-trained weights from ImageNet
- **Regularization**: Dropout and BatchNormalization to prevent overfitting
- **Fine-tuning**: Top 20% of base layers unfrozen in Phase 2

## Performance 📈

Expected performance metrics (varies based on training):

- **Test Accuracy**: 94-97%
- **Top-3 Accuracy**: 98-99%
- **Per-Class Accuracy**: Most classes >90%

## Project Structure 📁

```
Leaf_Disease_Detection/
├── dataset/
│   ├── train/          # Training images (220k)
│   ├── val/            # Validation images (19k)
│   └── test/           # Test images (19k)
├── models/             # Saved models
├── plots/              # Visualization outputs
├── logs/               # TensorBoard logs
├── train_model.py      # Main training script
├── evaluate_model.py   # Evaluation script
├── predict.py          # Prediction script
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Training Tips 💡

1. **GPU Acceleration**: Ensure CUDA is properly configured for faster training
2. **Batch Size**: Adjust based on your GPU memory (reduce if OOM errors occur)
3. **Early Stopping**: Model will stop early if no improvement for 8 epochs
4. **Learning Rate**: Automatically reduced when validation loss plateaus
5. **TensorBoard**: Monitor training in real-time:
   ```bash
   tensorboard --logdir logs/
   ```

## Advanced Options ⚙️

### Modify Training Parameters

Edit `train_model.py`:

```python
IMG_SIZE = 300        # Image size (224, 300, 384)
BATCH_SIZE = 32       # Batch size (16, 32, 64)
EPOCHS = 50          # Maximum epochs
LEARNING_RATE = 0.001 # Initial learning rate
```

### Try Different Base Models

Replace EfficientNetB3 with:
- `EfficientNetB0` - Faster, slightly less accurate
- `EfficientNetB4` - Slower, potentially more accurate
- `ResNet50` - Good alternative
- `InceptionV3` - Another strong option

## Troubleshooting 🔧

### Out of Memory Error

Reduce batch size in `train_model.py`:
```python
BATCH_SIZE = 16  # or even 8
```

### Slow Training

- Verify GPU is being used: `nvidia-smi`
- Reduce image size: `IMG_SIZE = 224`
- Use a lighter model: `EfficientNetB0`

### Poor Accuracy

- Train for more epochs
- Increase learning rate slightly
- Try different data augmentation parameters
- Fine-tune more layers of base model

## Citation 📄

If you use this system in your research, please cite:

```bibtex
@software{leaf_disease_detection,
  title={Plant Leaf Disease Detection System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Leaf_Disease_Detection}
}
```

## License 📜

See LICENSE file for details.

## Acknowledgments 🙏

- Dataset: PlantVillage Dataset
- Base Model: EfficientNet (Google Research)
- Framework: TensorFlow/Keras

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact 📧

For questions or issues, please open an issue on GitHub.

---

**Happy Disease Detection! 🌿✨**
