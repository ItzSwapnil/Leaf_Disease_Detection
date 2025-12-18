"""
Configuration for Leaf Disease Detection Project
"""
import os

# ============================================================
#                    MODEL CONFIGURATION
# ============================================================

# Image settings
IMG_SIZE = 160                    # Input image dimensions (160x160)
NUM_CLASSES = 46                  # Number of disease classes

# Training hyperparameters
BATCH_SIZE = 32                   # Images per batch (use 16 for low RAM)
EPOCHS_PHASE1 = 10                # Transfer learning epochs
EPOCHS_PHASE2 = 15                # Fine-tuning epochs
LEARNING_RATE_PHASE1 = 0.002      # Initial learning rate
LEARNING_RATE_PHASE2 = 0.0001     # Fine-tuning learning rate
LEARNING_RATE_RESUME = 1e-6       # Resume training learning rate

# Training steps (for 1/10th data sampling)
STEPS_PER_EPOCH = 750
VALIDATION_STEPS = 100

# Regularization
DROPOUT_RATE = 0.4
LABEL_SMOOTHING = 0.1

# ============================================================
#                       PATHS
# ============================================================

# Base directory
BASE_DIR = '/workspaces/Leaf_Disease_Detection'

# Dataset paths
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset/train')
VAL_DIR = os.path.join(BASE_DIR, 'dataset/val')
TEST_DIR = os.path.join(BASE_DIR, 'dataset/test')

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CHECKPOINT_PATH = os.path.join(MODELS_DIR, '1_10th_precision_model.h5')
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, '99pct_final_reached.h5')
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, 'class_indices.json')

# Output paths
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# ============================================================
#                    CPU OPTIMIZATION
# ============================================================

# Threading configuration for CPU training
INTRA_OP_THREADS = 4
INTER_OP_THREADS = 4

# ============================================================
#                    MODEL ARCHITECTURE
# ============================================================

# Base model
BASE_MODEL = 'EfficientNetV2B0'

# Classification head
DENSE_UNITS = 1024

# Layers to unfreeze during fine-tuning
UNFREEZE_LAYERS = 50

# ============================================================
#                    CALLBACKS
# ============================================================

EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# ============================================================
#                    CURRENT MODEL STATUS
# ============================================================

CURRENT_ACCURACY = 94.47  # Validation accuracy (%) - 99pct_final_reached.h5
TARGET_ACCURACY = 99.0    # Target accuracy (%)
MODEL_COMPLETE = False    # Training complete flag (96.5% achieved during training)

# Best model file
BEST_MODEL = 'models/99pct_final_reached.h5'  # 94.47% accuracy
