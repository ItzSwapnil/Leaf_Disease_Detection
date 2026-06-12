"""Central configuration for training, inference, and evaluation pipelines.

All hyperparameters, filesystem paths, and feature flags are defined here.
Individual scripts read these values at import time. Runtime overrides are
available via environment variables for CI/CD and reproducibility workflows.

Current configuration is optimised for 99%+ top-1 accuracy on the
PlantVillage-46 dataset using EfficientNetV2-B0 transfer learning on an
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


def _env_int(name: str, default: int) -> int:
    """Parse an integer from an environment variable with a safe fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return int(default)
    try:
        return int(raw_value.strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    """Parse a float from an environment variable with a safe fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return float(default)
    try:
        return float(raw_value.strip())
    except Exception:
        return float(default)


def _env_csv(name: str) -> list[str]:
    """Parse a comma-separated list from an environment variable."""
    raw_value = os.getenv(name, "")
    return [token.strip() for token in raw_value.split(",") if token.strip()]


def _env_int_list(name: str, default_csv: str) -> tuple[int, ...]:
    """Parse a comma-separated int list with fallback defaults."""
    raw_value = os.getenv(name, default_csv)
    values: list[int] = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except Exception:
            continue

    if not values:
        fallback = [
            part.strip() for part in default_csv.split(",") if part.strip()
        ]
        values = [int(part) for part in fallback]

    return tuple(values)


def _env_float_list(name: str, default_csv: str) -> tuple[float, ...]:
    """Parse a comma-separated float list with fallback defaults."""
    raw_value = os.getenv(name, default_csv)
    values: list[float] = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except Exception:
            continue

    if not values:
        fallback = [
            part.strip() for part in default_csv.split(",") if part.strip()
        ]
        values = [float(part) for part in fallback]

    return tuple(values)


# ============================================================
#                    IMAGE & CLASS SETTINGS
# ============================================================

IMG_SIZE = 224  # EfficientNetV2-B0 native resolution
NUM_CLASSES = 40  # Total classes after removing 6 incomplete families
NUM_CROPS = 10  # Total unique plant families

# ============================================================
#           HEAVY AUGMENTATION (background invariance)
# ============================================================
# These augmentations force the model to learn disease-specific
# leaf features instead of exploiting the plain background.

USE_RANDOM_RESIZED_CROP = _env_bool("LEAF_USE_RANDOM_RESIZED_CROP", True)
RANDOM_CROP_SCALE_MIN = _env_float("LEAF_RANDOM_CROP_SCALE_MIN", 0.6)
RANDOM_CROP_SCALE_MAX = _env_float("LEAF_RANDOM_CROP_SCALE_MAX", 1.0)
RANDOM_CROP_RATIO_MIN = _env_float("LEAF_RANDOM_CROP_RATIO_MIN", 0.75)
RANDOM_CROP_RATIO_MAX = _env_float("LEAF_RANDOM_CROP_RATIO_MAX", 1.33)

USE_COLOR_JITTER = _env_bool("LEAF_USE_COLOR_JITTER", True)
COLOR_JITTER_BRIGHTNESS = _env_float("LEAF_COLOR_JITTER_BRIGHTNESS", 0.3)
COLOR_JITTER_CONTRAST = _env_float("LEAF_COLOR_JITTER_CONTRAST", 0.3)
COLOR_JITTER_SATURATION = _env_float("LEAF_COLOR_JITTER_SATURATION", 0.3)
COLOR_JITTER_HUE = _env_float("LEAF_COLOR_JITTER_HUE", 0.1)

USE_GAUSSIAN_BLUR = _env_bool("LEAF_USE_GAUSSIAN_BLUR", True)
GAUSSIAN_BLUR_PROB = _env_float("LEAF_GAUSSIAN_BLUR_PROB", 0.3)
GAUSSIAN_BLUR_SIGMA_MIN = _env_float("LEAF_GAUSSIAN_BLUR_SIGMA_MIN", 0.1)
GAUSSIAN_BLUR_SIGMA_MAX = _env_float("LEAF_GAUSSIAN_BLUR_SIGMA_MAX", 2.0)

USE_GAUSSIAN_NOISE = _env_bool("LEAF_USE_GAUSSIAN_NOISE", True)
GAUSSIAN_NOISE_PROB = _env_float("LEAF_GAUSSIAN_NOISE_PROB", 0.3)
GAUSSIAN_NOISE_SIGMA = _env_float("LEAF_GAUSSIAN_NOISE_SIGMA", 0.05)

USE_RANDOM_ERASING = _env_bool("LEAF_USE_RANDOM_ERASING", True)
RANDOM_ERASING_PROB = _env_float("LEAF_RANDOM_ERASING_PROB", 0.3)
RANDOM_ERASING_SCALE_MIN = _env_float("LEAF_RANDOM_ERASING_SCALE_MIN", 0.05)
RANDOM_ERASING_SCALE_MAX = _env_float("LEAF_RANDOM_ERASING_SCALE_MAX", 0.33)
USE_BACKGROUND_RANDOMIZATION = _env_bool(
    "LEAF_USE_BACKGROUND_RANDOMIZATION", True
)

# ============================================================
#                 ATTENTION GUIDANCE HYPERPARAMETERS
# ============================================================
USE_ATTENTION_GUIDANCE = _env_bool("LEAF_USE_ATTENTION_GUIDANCE", True)
ATTENTION_BG_PENALTY_WEIGHT = _env_float(
    "LEAF_ATTENTION_BG_PENALTY_WEIGHT", 5.0
)
ATTENTION_SPARSITY_WEIGHT = _env_float("LEAF_ATTENTION_SPARSITY_WEIGHT", 0.3)
ATTENTION_DISEASE_REWARD_WEIGHT = _env_float(
    "LEAF_ATTENTION_DISEASE_REWARD_WEIGHT", 0.05
)
# Multi-block ViT attention regularization: regularize multiple
# encoder blocks for consistent attention throughout the network.
# Default: blocks 8, 10, 11 of the 12-block ViT-Base encoder.
ATTENTION_VIT_BLOCK_INDICES = _env_int_list(
    "LEAF_ATTENTION_VIT_BLOCK_INDICES", "8,10,11"
)
# Backward-compatible single-block fallback (only used if
# ATTENTION_VIT_BLOCK_INDICES is empty, which shouldn't happen).
ATTENTION_VIT_BLOCK_IDX = _env_int("LEAF_ATTENTION_VIT_BLOCK_IDX", 10)

# ============================================================
#                 TRAINING HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 32  # Safer default for 8 GB laptop GPUs at 224x224 + fp16
EPOCHS_PHASE1 = _env_int("LEAF_EPOCHS_PHASE1", 5)
EPOCHS_PHASE2 = _env_int("LEAF_EPOCHS_PHASE2", 10)
LEARNING_RATE_PHASE1 = 2e-4  # Safe LR for head-only + class equalizer
LEARNING_RATE_PHASE2 = (
    5e-5  # Lower LR for backbone fine-tuning (reduced to limit overfitting)
)
LEARNING_RATE_RESUME = 1e-6  # Resume/continued training LR

# Step control (0 = use all batches in the dataset)
STEPS_PER_EPOCH = 0
VALIDATION_STEPS = 0
TRAIN_DATA_FRACTION = _env_float("LEAF_TRAIN_DATA_FRACTION", 1.0)

# Optimiser settings (AdamW + cosine annealing)
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.02  # Increased for stronger regularization
LR_SCHEDULER = "cosine"
WARMUP_EPOCHS = 1
ACCUMULATION_STEPS = _env_int("LEAF_ACCUMULATION_STEPS", 1)

# Regularisation
DROPOUT_RATE = 0.5  # Increased from 0.4 for stronger regularization
LABEL_SMOOTHING = 0.15  # Increased from 0.1 to discourage overconfidence
USE_FOCAL_LOSS = False  # CrossEntropy + label smoothing preferred
USE_HIERARCHICAL_LOSS = _env_bool("LEAF_USE_HIERARCHICAL_LOSS", True)
FOCAL_GAMMA = 2.0
USE_MIXUP = _env_bool("LEAF_USE_MIXUP", False)
MIXUP_ALPHA = 0.3
USE_CUTMIX = _env_bool("LEAF_USE_CUTMIX", False)
CUTMIX_ALPHA = 1.0
MIXUP_PROB = _env_float("LEAF_MIXUP_PROB", 0.4)
CUTMIX_PROB = _env_float("LEAF_CUTMIX_PROB", 0.4)
NORMAL_PROB = _env_float("LEAF_NORMAL_PROB", 0.2)

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

# Evaluation and uncertainty analysis
CALIBRATION_BINS = _env_int("LEAF_CALIBRATION_BINS", 10)
TEMPERATURE_SCALING_STEPS = _env_int("LEAF_TEMPERATURE_SCALING_STEPS", 400)
TEMPERATURE_SCALING_LR = _env_float("LEAF_TEMPERATURE_SCALING_LR", 0.01)
BOOTSTRAP_SAMPLES = _env_int("LEAF_BOOTSTRAP_SAMPLES", 2000)
BOOTSTRAP_SEED = _env_int("LEAF_BOOTSTRAP_SEED", 42)
CONFIDENCE_REJECT_THRESHOLD = _env_float(
    "LEAF_CONFIDENCE_REJECT_THRESHOLD", 0.85
)
# Entropy threshold mode:
# - <= 1.0: normalized entropy ratio (0..1)
# - > 1.0: entropy in bits
ENTROPY_REJECT_THRESHOLD = _env_float("LEAF_ENTROPY_REJECT_THRESHOLD", 0.8)
OOD_MSP_THRESHOLD = _env_float("LEAF_OOD_MSP_THRESHOLD", 0.7)
OOD_MAX_SAMPLES = _env_int("LEAF_OOD_MAX_SAMPLES", 2048)
OOD_MAHALANOBIS_REG = _env_float("LEAF_OOD_MAHALANOBIS_REG", 1e-3)
MC_DROPOUT_ENABLED = _env_bool("LEAF_MC_DROPOUT_ENABLED", True)
MC_DROPOUT_PASSES = _env_int("LEAF_MC_DROPOUT_PASSES", 10)
MC_DROPOUT_MAX_SAMPLES = _env_int("LEAF_MC_DROPOUT_MAX_SAMPLES", 2048)
ROBUSTNESS_EVAL_ENABLED = _env_bool("LEAF_ROBUSTNESS_EVAL_ENABLED", True)
ROBUSTNESS_MAX_SAMPLES = _env_int("LEAF_ROBUSTNESS_MAX_SAMPLES", 512)
ROBUSTNESS_SEED = _env_int("LEAF_ROBUSTNESS_SEED", 42)
ROBUSTNESS_BLUR_SIGMAS = _env_float_list(
    "LEAF_ROBUSTNESS_BLUR_SIGMAS", "0.5,1.0,2.0"
)
ROBUSTNESS_BRIGHTNESS_FACTORS = _env_float_list(
    "LEAF_ROBUSTNESS_BRIGHTNESS_FACTORS", "0.8,0.6,1.2"
)
ROBUSTNESS_NOISE_SIGMAS = _env_float_list(
    "LEAF_ROBUSTNESS_NOISE_SIGMAS", "0.01,0.03,0.05"
)
ROBUSTNESS_FOG_LEVELS = _env_float_list(
    "LEAF_ROBUSTNESS_FOG_LEVELS", "0.1,0.2"
)
ROBUSTNESS_OCCLUSION_FRACS = _env_float_list(
    "LEAF_ROBUSTNESS_OCCLUSION_FRACS", "0.1,0.2"
)

# Logging behaviour
SAVE_LOG_ARCHIVE = _env_bool("LEAF_SAVE_LOG_ARCHIVE", False)
SAVE_RUN_MANIFESTS = _env_bool(
    "LEAF_SAVE_RUN_MANIFESTS",
    SAVE_LOG_ARCHIVE,
)

# ============================================================
#                       FILESYSTEM PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

# Dataset directories
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "val"
TEST_DIR = BASE_DIR / "dataset" / "test"

# Model artefact paths
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_MODEL_PATH = (
    MODELS_DIR / "leaf_disease_checkpoint.keras"
)  # train output
CLASSIFIER_MODEL_PATH = (
    MODELS_DIR / "leaf_disease_classifier.keras"
)  # fine-tune output
REFINED_MODEL_PATH = MODELS_DIR / "leaf_disease_refined.keras"  # refine output
# Strict canonical model path used across inference/evaluation/figure generation.
FINAL_MODEL_FILE_PATH = MODELS_DIR / "leaf_disease_refined.keras"
EFFNET_MODEL_FILE_PATH = (
    MODELS_DIR / "EfficientNetv2B0" / "leaf_disease_EfficientNetV2-B0.keras"
)
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
MCNEMAR_BASELINE_MODEL_PATH = os.getenv(
    "LEAF_MCNEMAR_BASELINE_MODEL_PATH",
    str(MODELS_DIR / "leaf_disease_effnetv2b1.keras"),
)
ENSEMBLE_MODEL_PATHS = _env_csv("LEAF_ENSEMBLE_MODELS")
OOD_DIR = os.getenv("LEAF_OOD_DIR", str(BASE_DIR / "dataset" / "ood"))

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

# Backbone: EfficientNetV2-B0 offers the best accuracy/speed trade-off
# for transfer learning on laptop-class hardware.
BASE_MODEL = "DINOv3"

# Classification head
DENSE_UNITS = 512

# Layer unfreezing for Phase 2 (-1 = unfreeze all layers)
UNFREEZE_LAYERS = -1


# Fine-tuning strategy
FINE_TUNE_UNFREEZE_LAYERS = _env_int("LEAF_FINE_TUNE_UNFREEZE_LAYERS", -1)
FINE_TUNE_EPOCHS = _env_int("LEAF_FINE_TUNE_EPOCHS", 10)
FINE_TUNE_BATCH_SIZE = _env_int("LEAF_FINE_TUNE_BATCH_SIZE", 32)
FINE_TUNE_LEARNING_RATE = _env_float("LEAF_FINE_TUNE_LEARNING_RATE", 5e-5)
FINE_TUNE_DATA_FRACTION = _env_float("LEAF_FINE_TUNE_DATA_FRACTION", 1.0)
FINE_TUNE_MAX_STEPS_PER_EPOCH = _env_int(
    "LEAF_FINE_TUNE_MAX_STEPS_PER_EPOCH", 0
)
FINE_TUNE_VAL_MAX_STEPS = _env_int("LEAF_FINE_TUNE_VAL_MAX_STEPS", 0)

# Stop when overfitting emerges (instead of only waiting on val_accuracy plateau)
OVERFITTING_STOP_ENABLED = _env_bool("LEAF_OVERFITTING_STOP_ENABLED", True)
OVERFITTING_STOP_MIN_GAP = _env_float("LEAF_OVERFITTING_STOP_MIN_GAP", 0.04)
OVERFITTING_STOP_PATIENCE = _env_int("LEAF_OVERFITTING_STOP_PATIENCE", 2)

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
EFFNET_BEST_MODEL = str(EFFNET_MODEL_FILE_PATH)

# ============================================================
#           STRING ALIASES (backward compatibility)
# ============================================================

CHECKPOINT_PATH = str(CHECKPOINT_MODEL_PATH)
CLASSIFIER_PATH = str(CLASSIFIER_MODEL_PATH)
REFINED_PATH = str(REFINED_MODEL_PATH)
FINAL_MODEL_PATH = str(FINAL_MODEL_FILE_PATH)
EFFNET_MODEL_PATH = str(EFFNET_MODEL_FILE_PATH)
CLASS_INDICES_PATH = str(CLASS_INDICES_PATH)
TRAIN_DIR = str(TRAIN_DIR)
VAL_DIR = str(VAL_DIR)
TEST_DIR = str(TEST_DIR)
MODELS_DIR = str(MODELS_DIR)
PLOTS_DIR = str(PLOTS_DIR)
LOGS_DIR = str(LOGS_DIR)
