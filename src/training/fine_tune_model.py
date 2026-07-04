import argparse
import os
import random
import sys

# Add project root to sys.path to support running directly as a script
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.amp import GradScaler

from src.training.train_model import train_one_epoch, validate, LeafDiseaseModel
from src.training.training_progress import ProgressEmitter
from src.training.training_utils import (
    BestModelSaver,
    _build_heavy_augmentation_layer,
    build_dynamic_yolo_dataset,
    build_loss,
    build_optimizer,
    get_mixup_cutmix_transforms,
    parse_class_structure,
    resolve_augmentation_probabilities,
    unfreeze_backbone_layers,
    OverfittingStopper,
)
from src.utils.config import (
    BASE_MODEL,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLASSIFIER_MODEL_PATH,
    CUTMIX_PROB,
    DENSE_UNITS,
    DROPOUT_RATE,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    LEARNING_RATE_PHASE2,
    MIXUP_ALPHA,
    MIXUP_PROB,
    NORMAL_PROB,
    NUM_CLASSES,
    OPTIMIZER,
    TRAIN_DIR,
    USE_MIXUP,
    USE_OPTIMIZER_EMA,
    USE_RANDAUGMENT,
    VAL_DIR,
    NUM_WORKERS,
    INTRA_OP_THREADS,
    INTER_OP_THREADS,
    UNFREEZE_LAYERS,
    OVERFITTING_STOP_ENABLED,
    OVERFITTING_STOP_MIN_GAP,
    OVERFITTING_STOP_PATIENCE,
)

try:
    from src.utils.config import CUTMIX_ALPHA, USE_CUTMIX
except ImportError:
    USE_CUTMIX = False
    CUTMIX_ALPHA = 1.0

VALID_OPTIMIZERS = {"adamw": "AdamW", "adam": "Adam", "sgd": "SGD", "rmsprop": "RMSprop"}

def _normalize_optimizer_name(raw_value):
    key = str(raw_value or OPTIMIZER or "AdamW").strip().lower()
    if key not in VALID_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer '{raw_value}'")
    return VALID_OPTIMIZERS[key]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--save-mode", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    # CPU Thread optimization
    torch.set_num_threads(INTRA_OP_THREADS)
    try:
        torch.set_num_interop_threads(INTER_OP_THREADS)
    except RuntimeError:
        pass

    # CUDA benchmark & TF32 optimizations for RTX GPUs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    from src.core.backbones import resolve_backbone_name
    backbone_name = resolve_backbone_name(args.base_model or os.getenv("LEAF_BASE_MODEL"), default=BASE_MODEL)
    optimizer_name = _normalize_optimizer_name(args.optimizer or os.getenv("LEAF_TRAIN_OPTIMIZER") or OPTIMIZER)
    
    # Save mode mapping
    save_mode = "with_optimizer"
    if args.save_mode or os.getenv("LEAF_SAVE_MODE"):
        from src.training.train_model import _normalize_save_mode
        save_mode = _normalize_save_mode(args.save_mode or os.getenv("LEAF_SAVE_MODE"))

    train_fraction = args.train_fraction if args.train_fraction is not None else float(os.getenv("LEAF_TRAIN_FRACTION", 1.0))
    batch_size = int(os.getenv("LEAF_BATCH_SIZE", BATCH_SIZE))
    seed = int(os.environ.get("RUN_SEED", 42))
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset Loading
    train_class_names = sorted(entry.name for entry in os.scandir(TRAIN_DIR) if entry.is_dir())
    crop_names = sorted(list(set(name.split("___")[0] for name in train_class_names)))
    val_class_names = sorted(entry.name for entry in os.scandir(VAL_DIR) if entry.is_dir())

    _skip_yolo = backbone_name == "DINOv3"
    num_workers = args.num_workers if args.num_workers is not None else NUM_WORKERS
    train_loader = build_dynamic_yolo_dataset(TRAIN_DIR, train_class_names, batch_size, shuffle=True, seed=seed, use_yolo=not _skip_yolo, fraction=train_fraction, num_workers=num_workers)
    val_loader = build_dynamic_yolo_dataset(VAL_DIR, train_class_names, batch_size, shuffle=False, seed=seed, use_yolo=not _skip_yolo, fraction=1.0, num_workers=num_workers)

    # Model Setup
    healthy_partners = parse_class_structure(train_class_names)
    model = LeafDiseaseModel(backbone_name, len(train_class_names), len(crop_names), healthy_partners).to(device)

    # Map class to crop index
    class_to_crop_idx = []
    for name in train_class_names:
        crop_family = name.split("___")[0]
        crop_idx = crop_names.index(crop_family)
        class_to_crop_idx.append(crop_idx)
    model.set_class_to_crop_mapping(class_to_crop_idx)

    # Load existing checkpoint from Phase 1
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        actual_state = state.get("model_state_dict", state)
        model.load_state_dict(actual_state)
    else:
        print(f"[Warning] Base checkpoint {CHECKPOINT_PATH} not found. Starting with random weights.")

    # Unfreeze Backbone for Fine-Tuning
    print("\nStarting Phase 2 (Fine-Tuning Backbone)")
    unfreeze_layers = int(os.getenv("LEAF_UNFREEZE_LAYERS", UNFREEZE_LAYERS))
    unfreeze_backbone_layers(model.backbone.backbone, unfreeze_layers)

    criterion = build_loss(class_weight=None, class_names=train_class_names)
    
    # We read from configuration/environment variables for Fine-Tuning
    learning_rate_ft = float(os.getenv("LEAF_FINE_TUNE_LEARNING_RATE", LEARNING_RATE_PHASE2))
    epochs_ft = int(os.getenv("LEAF_FINE_TUNE_EPOCHS", EPOCHS_PHASE2))

    optimizer_ft = build_optimizer(model, learning_rate_ft, optimizer_name)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, epochs_ft)
    scaler = GradScaler('cuda')

    heavy_augment = _build_heavy_augmentation_layer() if USE_RANDAUGMENT else None
    mixup_fn, cutmix_fn = get_mixup_cutmix_transforms(USE_MIXUP, USE_CUTMIX, MIXUP_ALPHA, CUTMIX_ALPHA, NUM_CLASSES)
    mixup_prob, cutmix_prob, _ = resolve_augmentation_probabilities(USE_MIXUP, USE_CUTMIX, MIXUP_PROB, CUTMIX_PROB, NORMAL_PROB)

    # Save to CLASSIFIER_MODEL_PATH
    saver = BestModelSaver(CLASSIFIER_MODEL_PATH, save_mode=save_mode, backbone_name=backbone_name)

    # Setup EMA model
    ema_model = None
    if USE_OPTIMIZER_EMA:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
        print("Using Exponential Moving Average (EMA) for model weights.")

    # Setup ProgressEmitter
    # Phase 2 starts after Phase 1 epochs
    total_epochs = int(EPOCHS_PHASE1) + epochs_ft
    progress_emitter = ProgressEmitter("training_phase2", total_epochs=total_epochs, completed_epochs_before=int(EPOCHS_PHASE1))
    progress_emitter.on_train_begin(len(train_loader), initial_epoch=int(EPOCHS_PHASE1))

    stopper = None
    if OVERFITTING_STOP_ENABLED:
        stopper = OverfittingStopper(
            min_gap=OVERFITTING_STOP_MIN_GAP,
            pvariance=OVERFITTING_STOP_PATIENCE, # typo fallback in OverfittingStopper arguments
        ) if hasattr(OverfittingStopper, "pvariance") else OverfittingStopper(
            min_gap=OVERFITTING_STOP_MIN_GAP,
            patience=OVERFITTING_STOP_PATIENCE,
            verbose=1
        )

    for epoch in range(epochs_ft):
        epoch_idx = int(EPOCHS_PHASE1) + epoch
        progress_emitter.on_epoch_begin(epoch_idx)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer_ft, scaler, criterion, device,
            mixup_fn, cutmix_fn, mixup_prob, cutmix_prob,
            None, progress_emitter, epoch_idx, heavy_augment_fn=heavy_augment,
            ema_model=ema_model
        )
        val_loss, val_acc = validate(ema_model if ema_model is not None else model, val_loader, criterion, device)
        print(f"Epoch {epoch_idx+1}/{total_epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
        saver.step(epoch_idx, ema_model.module if ema_model is not None else model, optimizer_ft, val_acc)
        progress_emitter.on_epoch_end(epoch_idx)
        scheduler_ft.step()

        if stopper is not None:
            stopper.step(epoch_idx, train_loss, val_loss, train_acc, val_acc)
            if stopper.stop_training:
                print(f"Early stopping triggered at epoch {epoch_idx+1} due to overfitting.")
                break

if __name__ == "__main__":
    main()
