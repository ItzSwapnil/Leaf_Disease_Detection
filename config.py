"""Central configuration for training, inference, and evaluation."""

from pathlib import Path

# ============================================================
#                    MODEL CONFIGURATION
# ============================================================

# Image settings
IMG_SIZE = 224                    # Input image dimensions (increased for GPU inference)
NUM_CLASSES = 46                  # Number of disease classes

# Training hyperparameters
BATCH_SIZE = 16                   # Images per batch (adjusted for larger images)
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

# Base directory (auto-detected from this file's location)
BASE_DIR = Path(__file__).resolve().parent

# Dataset paths
TRAIN_DIR = BASE_DIR / 'dataset' / 'train'
VAL_DIR = BASE_DIR / 'dataset' / 'val'
TEST_DIR = BASE_DIR / 'dataset' / 'test'

# Model paths
MODELS_DIR = BASE_DIR / 'models'
CHECKPOINT_MODEL_PATH = MODELS_DIR / 'leaf_disease_checkpoint.keras'
FINAL_MODEL_PATH = MODELS_DIR / 'leaf_disease_classifier.keras'
CLASS_INDICES_PATH = MODELS_DIR / 'class_indices.json'

# Output paths
PLOTS_DIR = BASE_DIR / 'plots'
LOGS_DIR = BASE_DIR / 'logs'

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

CURRENT_ACCURACY = None   # Populate from latest evaluation
TARGET_ACCURACY = 99.0    # Target accuracy (%)
MODEL_COMPLETE = False    # Training complete flag

# Best model file
BEST_MODEL = str(FINAL_MODEL_PATH)

# Compatibility aliases used by existing scripts.
CHECKPOINT_PATH = str(CHECKPOINT_MODEL_PATH)
FINAL_MODEL_PATH = str(FINAL_MODEL_PATH)
CLASS_INDICES_PATH = str(CLASS_INDICES_PATH)
TRAIN_DIR = str(TRAIN_DIR)
VAL_DIR = str(VAL_DIR)
TEST_DIR = str(TEST_DIR)
MODELS_DIR = str(MODELS_DIR)
PLOTS_DIR = str(PLOTS_DIR)
LOGS_DIR = str(LOGS_DIR)
