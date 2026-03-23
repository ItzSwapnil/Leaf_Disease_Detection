"""Central configuration for training, inference, and evaluation pipelines.

All hyperparameters, filesystem paths, and feature flags are defined here.
Individual scripts read these values at import time. Runtime overrides are
available via environment variables for CI/CD and reproducibility workflows.

Current configuration is optimised for 99%+ top-1 accuracy on the
PlantVillage-46 dataset using EfficientNetV2-S transfer learning on an
NVIDIA RTX 5060 Laptop GPU (8 GB VRAM).

References:
    - EfficientNetV2: Tan & Le, "EfficientNetV2: Smaller Models and Faster
      Training", ICML 2021.
    - AdamW: Loshchilov & Hutter, "Decoupled Weight Decay Regularization",
      ICLR 2019.
    - MixUp: Zhang et al., "mixup: Beyond Empirical Risk Minimization",
      ICLR 2018.
    - CutMix: Yun et al., "CutMix: Regularization Strategy to Train Strong
      Classifiers with Localizable Features", ICCV 2019.
    - Label Smoothing: Szegedy et al., "Rethinking the Inception Architecture
      for Computer Vision", CVPR 2016.
"""

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean from an environment variable with a safe fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default)
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


# ============================================================
#                    IMAGE & CLASS SETTINGS
# ============================================================

IMG_SIZE = 224                    # EfficientNetV2-S native resolution
NUM_CLASSES = 46                  # PlantVillage-46 disease classes

# ============================================================
#                 TRAINING HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 64                   # Fits 8 GB VRAM at 224x224 + fp16
EPOCHS_PHASE1 = 5                 # Phase 1: frozen-backbone head warm-up
EPOCHS_PHASE2 = 10                # Phase 2: full-model fine-tuning
LEARNING_RATE_PHASE1 = 2e-3       # Aggressive LR for head-only training
LEARNING_RATE_PHASE2 = 1e-4       # Lower LR for backbone fine-tuning
LEARNING_RATE_RESUME = 1e-6       # Resume/continued training LR

# Step control (0 = use all batches in the dataset)
STEPS_PER_EPOCH = 0
VALIDATION_STEPS = 0

# Optimiser settings (AdamW + cosine annealing)
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
WARMUP_EPOCHS = 1
ACCUMULATION_STEPS = 1            # No gradient accumulation needed at BS=64

# Regularisation
DROPOUT_RATE = 0.4
LABEL_SMOOTHING = 0.1
USE_FOCAL_LOSS = False            # CrossEntropy + label smoothing preferred
FOCAL_GAMMA = 2.0
USE_MIXUP = True
MIXUP_ALPHA = 0.3
USE_CUTMIX = True
CUTMIX_ALPHA = 1.0

# Exponential Moving Average of weights for better generalisation
USE_OPTIMIZER_EMA = True
EMA_MOMENTUM = 0.999

# RandAugment automatic data augmentation (Cubuk et al., 2020)
USE_RANDAUGMENT = True
RANDAUGMENT_NUM_LAYERS = 2
RANDAUGMENT_MAGNITUDE = 9

# Test-Time Augmentation for inference
USE_TTA = True
TTA_AUGMENTS = 5

# Logging behaviour
SAVE_LOG_ARCHIVE = _env_bool("LEAF_SAVE_LOG_ARCHIVE", False)
SAVE_RUN_MANIFESTS = _env_bool(
    "LEAF_SAVE_RUN_MANIFESTS",
    SAVE_LOG_ARCHIVE,
)

# ============================================================
#                       FILESYSTEM PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Dataset directories
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "val"
TEST_DIR = BASE_DIR / "dataset" / "test"

# Model artefact paths
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_MODEL_PATH = MODELS_DIR / "leaf_disease_checkpoint.keras"
FINAL_MODEL_FILE_PATH = MODELS_DIR / "leaf_disease_classifier.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"

# Output directories
PLOTS_DIR = BASE_DIR / "plots"
LOGS_DIR = BASE_DIR / "logs"

# ============================================================
#                     CPU OPTIMISATION
# ============================================================

INTRA_OP_THREADS = 4
INTER_OP_THREADS = 4

# ============================================================
#                    MODEL ARCHITECTURE
# ============================================================

# Backbone: EfficientNetV2-S offers the best accuracy/speed trade-off
# for transfer learning on laptop-class hardware.
BASE_MODEL = "EfficientNetV2S"

# Classification head
DENSE_UNITS = 512

# Layer unfreezing for Phase 2 (-1 = unfreeze all layers)
UNFREEZE_LAYERS = -1



# Fine-tuning strategy
FINE_TUNE_UNFREEZE_LAYERS = -1
FINE_TUNE_EPOCHS = 10
FINE_TUNE_BATCH_SIZE = 64
FINE_TUNE_LEARNING_RATE = 5e-5
FINE_TUNE_DATA_FRACTION = 1.0
FINE_TUNE_MAX_STEPS_PER_EPOCH = 0
FINE_TUNE_VAL_MAX_STEPS = 0

# ============================================================
#                       CALLBACKS
# ============================================================

EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# ============================================================
#                    ACCURACY TARGET
# ============================================================

TARGET_ACCURACY = 99.0  # Engineering target (%)

# Best model path alias
BEST_MODEL = str(FINAL_MODEL_FILE_PATH)

# ============================================================
#           STRING ALIASES (backward compatibility)
# ============================================================

CHECKPOINT_PATH = str(CHECKPOINT_MODEL_PATH)
FINAL_MODEL_PATH = str(FINAL_MODEL_FILE_PATH)
CLASS_INDICES_PATH = str(CLASS_INDICES_PATH)
TRAIN_DIR = str(TRAIN_DIR)
VAL_DIR = str(VAL_DIR)
TEST_DIR = str(TEST_DIR)
MODELS_DIR = str(MODELS_DIR)
PLOTS_DIR = str(PLOTS_DIR)
LOGS_DIR = str(LOGS_DIR)
