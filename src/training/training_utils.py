import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2

from src.utils.config import (
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST,
    COLOR_JITTER_HUE,
    COLOR_JITTER_SATURATION,
    GAUSSIAN_BLUR_PROB,
    GAUSSIAN_BLUR_SIGMA_MAX,
    GAUSSIAN_BLUR_SIGMA_MIN,
    IMG_SIZE,
    LABEL_SMOOTHING,
    OPTIMIZER,
    RANDOM_CROP_RATIO_MAX,
    RANDOM_CROP_RATIO_MIN,
    RANDOM_CROP_SCALE_MAX,
    RANDOM_CROP_SCALE_MIN,
    RANDOM_ERASING_PROB,
    RANDOM_ERASING_SCALE_MAX,
    RANDOM_ERASING_SCALE_MIN,
    USE_COLOR_JITTER,
    USE_GAUSSIAN_BLUR,
    USE_RANDOM_ERASING,
    USE_RANDOM_RESIZED_CROP,
    WEIGHT_DECAY,
)

# -----------------------------------------------------------------------------
# Learning Rate Schedulers
# -----------------------------------------------------------------------------

def build_warmup_cosine_schedule(optimizer: torch.optim.Optimizer, peak_lr: float, min_lr: float, warmup_steps: int, total_steps: int):
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=max(1, warmup_steps))
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=min_lr)
    return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

# -----------------------------------------------------------------------------
# Optimizers & Loss
# -----------------------------------------------------------------------------

def build_optimizer(model: nn.Module, learning_rate: float, optimizer_name: Optional[str] = None):
    name = str(optimizer_name or OPTIMIZER or "AdamW").strip().lower()

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer '{name}'. Expected: AdamW, Adam, SGD, RMSprop.")

def build_loss(class_weight: Optional[Dict[int, float]] = None, class_names: Optional[list[str]] = None):
    weights = None
    if class_weight:
        # Sort weights by class index
        w_list = [class_weight[i] for i in range(len(class_weight))]
        weights = torch.tensor(w_list, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    return criterion

# -----------------------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------------------

def _build_heavy_augmentation_layer():
    """Builds a comprehensive v2 transforms pipeline for heavy data augmentation."""
    transforms_list = []

    if USE_RANDOM_RESIZED_CROP:
        transforms_list.append(v2.RandomResizedCrop(
            size=(IMG_SIZE, IMG_SIZE),
            scale=(RANDOM_CROP_SCALE_MIN, RANDOM_CROP_SCALE_MAX),
            ratio=(RANDOM_CROP_RATIO_MIN, RANDOM_CROP_RATIO_MAX)
        ))

    transforms_list.append(v2.RandomHorizontalFlip())
    transforms_list.append(v2.RandomRotation(degrees=27))

    if USE_COLOR_JITTER:
        transforms_list.append(v2.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CONTRAST,
            saturation=COLOR_JITTER_SATURATION,
            hue=COLOR_JITTER_HUE
        ))

    if USE_GAUSSIAN_BLUR:
        transforms_list.append(v2.RandomApply([
            v2.GaussianBlur(kernel_size=7, sigma=(GAUSSIAN_BLUR_SIGMA_MIN, GAUSSIAN_BLUR_SIGMA_MAX))
        ], p=GAUSSIAN_BLUR_PROB))

    if USE_RANDOM_ERASING:
        transforms_list.append(v2.RandomErasing(
            p=RANDOM_ERASING_PROB,
            scale=(RANDOM_ERASING_SCALE_MIN, RANDOM_ERASING_SCALE_MAX)
        ))

    # Note: Gaussian noise can be added dynamically inside the training loop if needed.
    return v2.Compose(transforms_list)

def resolve_augmentation_probabilities(use_mixup, use_cutmix, mixup_prob, cutmix_prob, normal_prob):
    mix = float(mixup_prob) if use_mixup else 0.0
    cut = float(cutmix_prob) if use_cutmix else 0.0
    normal = float(normal_prob)
    total = mix + cut + normal
    if total <= 0.0:
        return 0.0, 0.0, 1.0
    return mix / total, cut / total, normal / total

def get_mixup_cutmix_transforms(use_mixup, use_cutmix, mixup_alpha, cutmix_alpha, num_classes):
    """Returns v2.MixUp and v2.CutMix instances."""
    mixup_transform = v2.MixUp(alpha=mixup_alpha, num_classes=num_classes) if use_mixup else None
    cutmix_transform = v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes) if use_cutmix else None
    return mixup_transform, cutmix_transform

# -----------------------------------------------------------------------------
# Training Callbacks
# -----------------------------------------------------------------------------

class BestModelSaver:
    def __init__(self, model_path: str, save_mode: str = "with_optimizer", verbose: int = 1, backbone_name: str | None = None):
        self.model_path = model_path
        self.save_mode = save_mode
        self.verbose = verbose
        self.backbone_name = backbone_name
        self.best_acc = float('-inf')

    def step(self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, acc: float):
        if acc > self.best_acc:
            self.best_acc = acc
            state = {'model_state_dict': model.state_dict()}
            if self.save_mode == "with_optimizer":
                state['optimizer_state_dict'] = optimizer.state_dict()

            # Save standard model
            torch.save(state, self.model_path)

            # Save an explicitly named copy with the backbone for user convenience
            if self.backbone_name:
                named_path = self.model_path.replace(".pt", f"_{self.backbone_name}.pt")
                if named_path != self.model_path:
                    torch.save(state, named_path)

            # Save no-optimizer variant if requested
            if self.save_mode == "all" or self.save_mode == "without_optimizer":
                no_opt_path = self.model_path.replace(".pt", "_no_optimizer.pt")
                if no_opt_path == self.model_path:
                    no_opt_path += "_no_opt"
                torch.save({'model_state_dict': model.state_dict()}, no_opt_path)

            if self.verbose:
                print(f"Saved improved model at epoch {epoch + 1}: acc={acc:.6f}")

class OverfittingStopper:
    def __init__(self, min_gap: float = 0.05, patience: int = 2, verbose: int = 1):
        self.min_gap = min_gap
        self.patience = patience
        self.verbose = verbose
        self.bad_epochs = 0
        self.stop_training = False

    def step(self, epoch: int, train_loss: float, val_loss: float, train_acc: float, val_acc: float):
        gap = train_acc - val_acc
        overfitting_now = (val_loss > train_loss) and (gap >= self.min_gap)

        if overfitting_now:
            self.bad_epochs += 1
            if self.verbose:
                print(f"Overfitting signal: epoch={epoch + 1}, gap={gap:.4f} ({self.bad_epochs}/{self.patience})")
            if self.bad_epochs >= self.patience:
                self.stop_training = True
        else:
            self.bad_epochs = 0

class RollingPreOverfitRestorer:
    def __init__(self, min_gap: float = 0.0, patience: int = 1, snapshot_count: int = 10, snapshot_dir: str = "snapshots", strict: bool = True, verbose: int = 1):
        self.min_gap = min_gap
        self.patience = patience
        self.snapshot_count = max(1, snapshot_count)
        self.strict = strict
        self.verbose = verbose
        self.bad_epochs = 0
        self.stop_training = False

        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.safe_snapshots: list[Path] = []
        self.best_snapshot_path = self.snapshot_dir / "best_safe.pt"
        self.best_snapshot_metric = float("-inf")
        self.initial_snapshot_path = self.snapshot_dir / "initial_safe.pt"

    def on_train_begin(self, model: nn.Module):
        torch.save(model.state_dict(), self.initial_snapshot_path)

    def step(self, epoch: int, model: nn.Module, train_loss: float, val_loss: float, train_acc: float, val_acc: float):
        gap = train_acc - val_acc
        loss_overfit = val_loss > train_loss
        gap_overfit = gap > self.min_gap

        if self.strict:
            overfitting_now = loss_overfit or gap_overfit
        else:
            overfitting_now = loss_overfit and gap_overfit

        if overfitting_now:
            self.bad_epochs += 1
            if self.verbose:
                print(f"Rolling restore monitor: gap={gap:.4f} ({self.bad_epochs}/{self.patience})")

            if self.bad_epochs >= self.patience:
                # Restore best safe model
                if self.best_snapshot_metric != float("-inf"):
                    model.load_state_dict(torch.load(self.best_snapshot_path, weights_only=True))
                    if self.verbose:
                        print("Stopping training: Restored best safe weights.")
                else:
                    model.load_state_dict(torch.load(self.initial_snapshot_path, weights_only=True))
                    if self.verbose:
                        print("Stopping training: Restored initial weights.")
                self.stop_training = True
                return True # Indicates restored
        else:
            self.bad_epochs = 0
            # Save safe snapshot
            snapshot_path = self.snapshot_dir / f"safe_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), snapshot_path)
            self.safe_snapshots.append(snapshot_path)

            if val_acc >= self.best_snapshot_metric:
                torch.save(model.state_dict(), self.best_snapshot_path)
                self.best_snapshot_metric = val_acc

            if len(self.safe_snapshots) > self.snapshot_count:
                oldest = self.safe_snapshots.pop(0)
                if oldest.exists():
                    oldest.unlink()
        return False

# -----------------------------------------------------------------------------
# Family Deviation Classifier
# -----------------------------------------------------------------------------

def parse_class_structure(class_names: list[str]) -> list[int]:
    family_of_class = []
    healthy_class_of_family = {}
    for idx, name in enumerate(class_names):
        if "___" in name:
            family, subclass = name.split("___", 1)
        else:
            family = name.split()[0]
            subclass = name
        family_of_class.append(family)
        if "healthy" in subclass.lower():
            healthy_class_of_family[family] = idx

    healthy_partner_indices = []
    for idx, name in enumerate(class_names):
        family = family_of_class[idx]
        partner_idx = healthy_class_of_family.get(family, -1)
        if partner_idx == idx:
            healthy_partner_indices.append(-1)
        else:
            healthy_partner_indices.append(partner_idx)
    return healthy_partner_indices

class FamilyDeviationClassifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int, healthy_partners: list[int]):
        super().__init__()
        self.num_classes = num_classes
        self.healthy_partners = list(healthy_partners)

        self.fc = nn.Linear(num_features, num_classes)

        gather_indices = [idx if idx != -1 else 0 for idx in self.healthy_partners]
        mask = [1.0 if idx != -1 else 0.0 for idx in self.healthy_partners]

        self.register_buffer("gather_indices", torch.tensor(gather_indices, dtype=torch.long))
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))

    def forward(self, x):
        raw_logits = self.fc(x)
        partner_logits = raw_logits[:, self.gather_indices]
        logits = raw_logits + partner_logits * self.mask
        return logits

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

import cv2


class LeafYOLODataset(torch.utils.data.Dataset):
    def __init__(self, filepaths, labels, num_classes, use_yolo=True, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.num_classes = num_classes
        self.use_yolo = use_yolo
        self.transform = transform
        self.detector = None

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        if self.use_yolo and self.detector is None:
            # Lazy init to avoid multiprocessing issues
            from src.core.yolo_leaf import YOLOLeafDetector
            self.detector = YOLOLeafDetector()

        path = self.filepaths[idx]
        label = self.labels[idx]

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            img_bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        if self.use_yolo:
            assert self.detector is not None
            focus_mask = self.detector.get_focus_mask(img_bgr)
            focus_mask = cv2.resize(focus_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            if len(focus_mask.shape) == 2:
                focus_mask = np.expand_dims(focus_mask, axis=-1)
        else:
            focus_mask = np.ones((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

        image_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))).contiguous().float() / 255.0
        mask_tensor = torch.from_numpy(focus_mask.transpose((2, 0, 1))).contiguous().float()

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask_tensor, label

def collect_dataset_files(dir_path: str | Path, class_names: list[str]) -> tuple[list[str], list[int]]:
    dir_path = Path(dir_path)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    filepaths = []
    labels = []
    for category_name in class_names:
        category_dir = dir_path / category_name
        if not category_dir.exists():
            continue
        idx = class_to_idx[category_name]
        for entry in os.scandir(category_dir):
            if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                filepaths.append(entry.path)
                labels.append(idx)
    return filepaths, labels

def build_dynamic_yolo_dataset(
    dir_path: str | Path,
    class_names: list[str],
    batch_size: int,
    shuffle: bool,
    seed: Optional[int] = None,
    use_yolo: bool = True,
    transform=None,
    num_workers: int = 0,
    drop_last: bool = False,
    fraction: float = 1.0
):
    filepaths, labels = collect_dataset_files(dir_path, class_names)

    if fraction < 1.0:
        combined = list(zip(filepaths, labels))
        rng = random.Random(seed if seed is not None else 42)
        rng.shuffle(combined)
        limit = max(1, int(len(combined) * fraction))
        combined = combined[:limit]
        if not combined:
            filepaths, labels = [], []
        else:
            filepaths_t, labels_t = zip(*combined)
            filepaths, labels = list(filepaths_t), list(labels_t)

    dataset = LeafYOLODataset(filepaths, labels, len(class_names), use_yolo=use_yolo, transform=transform)

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
        drop_last=drop_last
    )
    return loader
