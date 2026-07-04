import argparse
import os
import random
import re
import sys

# Add project root to sys.path to support running directly as a script
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.core.backbones import (
    BackboneWrapper,
    resolve_backbone_name,
)
from src.training.training_progress import ProgressEmitter
from src.training.training_utils import (
    BestModelSaver,
    FamilyDeviationClassifier,
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
    ACCUMULATION_STEPS,
    BASE_MODEL,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CUTMIX_PROB,
    DENSE_UNITS,
    DROPOUT_RATE,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    LEARNING_RATE_PHASE1,
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
VALID_SAVE_MODES = {"with_optimizer", "without_optimizer", "all"}

def _parse_fraction(raw_value, default_value):
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = float(default_value)
    return max(1e-6, min(1.0, value))

def _parse_class_equalizer(arg_value):
    if arg_value is not None:
        return str(arg_value).strip().lower() == "on"
    env_value = os.getenv("LEAF_CLASS_EQUALIZER")
    if env_value is None:
        return True
    return str(env_value).strip().lower() in {"1", "true", "yes", "y", "on"}

def _normalize_optimizer_name(raw_value):
    key = str(raw_value or OPTIMIZER or "AdamW").strip().lower()
    if key not in VALID_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer '{raw_value}'")
    return VALID_OPTIMIZERS[key]

def _normalize_save_mode(raw_value):
    mode = str(raw_value or "with_optimizer").strip().lower().replace("-", "_")
    if mode not in VALID_SAVE_MODES:
        raise ValueError(f"Unsupported save mode '{raw_value}'")
    return mode

def _canonical_class_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())

def _resolve_validation_class_aliases(val_class_names, train_class_names):
    train_set = set(train_class_names)
    train_by_canonical = {}
    for train_name in train_class_names:
        key = _canonical_class_name(train_name)
        train_by_canonical[key] = train_name

    val_to_train = {}
    unmatched = []
    for val_name in val_class_names:
        if val_name in train_set:
            val_to_train[val_name] = val_name
            continue
        key = _canonical_class_name(val_name)
        mapped_train = train_by_canonical.get(key)
        if mapped_train is None:
            unmatched.append(val_name)
            continue
        val_to_train[val_name] = mapped_train
    return val_to_train

class LeafDiseaseModel(nn.Module):
    def __init__(self, backbone_name, num_classes, num_crops, healthy_partners):
        super().__init__()
        self.backbone = BackboneWrapper(backbone_name, pretrained=True)
        in_features = getattr(self.backbone, "out_features")

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if 'vit' not in backbone_name.lower() else nn.Identity()

        features_dim = int(str(in_features))
        self.head_bn = nn.BatchNorm1d(features_dim)
        self.head_dense_1 = nn.Sequential(
            nn.Linear(features_dim, DENSE_UNITS),
            nn.SiLU()
        )
        self.head_dropout_1 = nn.Dropout(DROPOUT_RATE)
        self.head_dense_2 = nn.Sequential(
            nn.Linear(DENSE_UNITS, DENSE_UNITS // 2),
            nn.SiLU()
        )
        self.head_dropout_2 = nn.Dropout(DROPOUT_RATE * 0.5)

        self.crop_logits = nn.Linear(DENSE_UNITS // 2, num_crops)
        self.disease_logits = FamilyDeviationClassifier(DENSE_UNITS // 2, num_classes, healthy_partners)
        self.register_buffer("class_to_crop_idx", torch.zeros(num_classes, dtype=torch.long))

    def set_class_to_crop_mapping(self, mapping_list):
        self.class_to_crop_idx.copy_(torch.tensor(mapping_list, dtype=torch.long))

    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = self.pool(features).flatten(1)
        elif len(features.shape) == 3:
            features = features.mean(dim=1)

        x = self.head_bn(features)
        x = self.head_dense_1(x)
        x = self.head_dropout_1(x)
        x = self.head_dense_2(x)
        x = self.head_dropout_2(x)

        crop_out = self.crop_logits(x)
        disease_out = self.disease_logits(x)

        return {
            "crop_output": crop_out,
            "disease_output": disease_out
        }

def train_one_epoch(
    model, dataloader, optimizer, scaler, criterion,
    device, mixup_fn, cutmix_fn, mixup_prob, cutmix_prob,
    interval_logger, progress_emitter, epoch,
    heavy_augment_fn=None, class_weight_tensor=None,
    ema_model=None
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    accum_steps = int(os.getenv("LEAF_ACCUMULATION_STEPS", ACCUMULATION_STEPS))
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Train", leave=False)
    for batch_idx, (images, masks, labels) in enumerate(pbar):
        images, masks, labels = (
            images.to(device, non_blocking=True),
            masks.to(device, non_blocking=True),
            labels.to(device, non_blocking=True)
        )

        if heavy_augment_fn:
            images = heavy_augment_fn(images)

        route = random.random()
        mixed_labels = None
        if route < mixup_prob and mixup_fn:
            images, mixed_labels = mixup_fn(images, labels)
        elif route < mixup_prob + cutmix_prob and cutmix_fn:
            images, mixed_labels = cutmix_fn(images, labels)

        # Determine bf16 support dynamically
        use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())
        dtype = torch.bfloat16 if use_bf16 else torch.float16

        with autocast(device_type=device.type, dtype=dtype):
            outputs = model(images)
            disease_out = outputs["disease_output"]
            crop_out = outputs["crop_output"]

            if mixed_labels is not None:
                mixed_labels = mixed_labels.squeeze(1)
                loss_disease = -torch.sum(
                    mixed_labels * torch.log_softmax(disease_out, dim=-1),
                    dim=-1
                ).mean()
                
                active_model = ema_model.module if ema_model is not None else model
                if hasattr(active_model, "module"):
                    active_model = getattr(active_model, "module")
                class_to_crop = getattr(active_model, "class_to_crop_idx")
                num_crops = getattr(active_model, "crop_logits").out_features
                
                mixed_crop_labels = torch.zeros(
                    (mixed_labels.size(0), num_crops),
                    device=mixed_labels.device
                )
                mixed_crop_labels.scatter_add_(
                    1,
                    class_to_crop.unsqueeze(0).expand(
                        mixed_labels.size(0), -1
                    ).to(mixed_labels.device),
                    mixed_labels
                )
                
                loss_crop = -torch.sum(
                    mixed_crop_labels * torch.log_softmax(crop_out, dim=-1),
                    dim=-1
                ).mean()
            else:
                loss_disease = criterion(disease_out, labels)
                
                active_model = ema_model.module if ema_model is not None else model
                if hasattr(active_model, "module"):
                    active_model = getattr(active_model, "module")
                class_to_crop = getattr(active_model, "class_to_crop_idx")
                
                crop_labels = class_to_crop[labels]
                loss_crop = nn.CrossEntropyLoss()(crop_out, crop_labels)
                
            loss = (loss_disease + 0.5 * loss_crop) / accum_steps

        if scaler is not None and not use_bf16:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema_model is not None:
                    ema_model.update_parameters(model)
        else:
            loss.backward()
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                if ema_model is not None:
                    ema_model.update_parameters(model)

        running_loss += float(loss.item() * accum_steps) * images.size(0)
        _, predicted = torch.max(disease_out, 1)
        if mixed_labels is not None:
            _, target_max = torch.max(mixed_labels, 1)
            correct += int((predicted == target_max).sum().item())
        else:
            correct += int((predicted == labels).sum().item())
        total += images.size(0)

        if interval_logger:
            interval_logger.on_train_batch_end(
                batch_idx,
                logs={"loss": loss.item() * accum_steps, "accuracy": correct / total}
            )
        if progress_emitter:
            progress_emitter.on_train_batch_end(batch_idx)

        pbar.set_postfix({
            'loss': f"{running_loss/total:.4f}",
            'acc': f"{correct/total:.4f}"
        })

    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Validate", leave=False)
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    for images, masks, labels in pbar:
        images, masks, labels = images.to(device, non_blocking=True), masks.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=dtype):
            outputs = model(images)
            disease_out = outputs["disease_output"]
            crop_out = outputs["crop_output"]
            
            loss_disease = criterion(disease_out, labels)
            
            active_model = getattr(model, "module") if hasattr(model, "module") else model
            class_to_crop = getattr(active_model, "class_to_crop_idx")
            
            crop_labels = class_to_crop[labels]
            loss_crop = nn.CrossEntropyLoss()(crop_out, crop_labels)
            
            loss = loss_disease + 0.5 * loss_crop

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(disease_out, 1)
        correct += (predicted == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--save-mode", default=None)
    parser.add_argument("--class-equalizer", choices=["on", "off"], default=None)
    parser.add_argument("--must-review", choices=["on", "off"], default=None)
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

    backbone_name = resolve_backbone_name(args.base_model or os.getenv("LEAF_BASE_MODEL"), default=BASE_MODEL)
    optimizer_name = _normalize_optimizer_name(args.optimizer or os.getenv("LEAF_TRAIN_OPTIMIZER") or OPTIMIZER)
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

    # Phase 1: Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion = build_loss(class_weight=None, class_names=train_class_names)
    optimizer = build_optimizer(model, LEARNING_RATE_PHASE1, optimizer_name)
    scaler = GradScaler('cuda')

    heavy_augment = _build_heavy_augmentation_layer() if USE_RANDAUGMENT else None
    mixup_fn, cutmix_fn = get_mixup_cutmix_transforms(USE_MIXUP, USE_CUTMIX, MIXUP_ALPHA, CUTMIX_ALPHA, NUM_CLASSES)
    mixup_prob, cutmix_prob, _ = resolve_augmentation_probabilities(USE_MIXUP, USE_CUTMIX, MIXUP_PROB, CUTMIX_PROB, NORMAL_PROB)

    saver = BestModelSaver(CHECKPOINT_PATH, save_mode=save_mode, backbone_name=backbone_name)

    # Setup EMA model
    ema_model = None
    if USE_OPTIMIZER_EMA:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
        print("Using Exponential Moving Average (EMA) for model weights.")

    # Setup ProgressEmitter
    total_epochs = int(EPOCHS_PHASE1) + int(EPOCHS_PHASE2)
    progress_emitter = ProgressEmitter("training_phase1", total_epochs=total_epochs)

    # Phase 1 Loop
    print("Starting Phase 1 (Frozen Backbone)")
    progress_emitter.on_train_begin(len(train_loader), initial_epoch=0)
    for epoch in range(int(EPOCHS_PHASE1)):
        progress_emitter.on_epoch_begin(epoch)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device,
            mixup_fn, cutmix_fn, mixup_prob, cutmix_prob,
            None, progress_emitter, epoch, heavy_augment_fn=heavy_augment,
            ema_model=ema_model
        )
        val_loss, val_acc = validate(ema_model if ema_model is not None else model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{int(EPOCHS_PHASE1)} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
        saver.step(epoch, ema_model.module if ema_model is not None else model, optimizer, val_acc)
        progress_emitter.on_epoch_end(epoch)

    # Phase 2: Unfreeze Backbone for Fine-Tuning
    print("\nStarting Phase 2 (Fine-Tuning Backbone)")
    progress_emitter.stage = "training_phase2"
    unfreeze_layers = int(os.getenv("LEAF_UNFREEZE_LAYERS", UNFREEZE_LAYERS))
    unfreeze_backbone_layers(model.backbone.backbone, unfreeze_layers)

    # Note: Define optimizer_ft, scheduler_ft here as per logic requirements
    optimizer_ft = build_optimizer(model, LEARNING_RATE_PHASE2, optimizer_name)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, int(EPOCHS_PHASE2))

    stopper = None
    if OVERFITTING_STOP_ENABLED:
        stopper = OverfittingStopper(
            min_gap=OVERFITTING_STOP_MIN_GAP,
            patience=OVERFITTING_STOP_PATIENCE,
            verbose=1
        )

    for epoch in range(int(EPOCHS_PHASE2)):
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
