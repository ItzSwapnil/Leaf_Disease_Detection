from __future__ import annotations

import inspect
import math
from typing import Dict, Optional, Sequence

import keras
import numpy as np

from config import (
    ACCUMULATION_STEPS,
    BATCH_SIZE,
    EMA_MOMENTUM,
    FOCAL_GAMMA,
    LABEL_SMOOTHING,
    OPTIMIZER,
    USE_FOCAL_LOSS,
    USE_OPTIMIZER_EMA,
    WEIGHT_DECAY,
)

# Learning rate schedule

@keras.saving.register_keras_serializable(package="training_utils")
class WarmupCosineSchedule(keras.optimizers.schedules.LearningRateSchedule):
    

    def __init__(self, peak_lr: float, min_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(max(0, warmup_steps))
        self.total_steps = int(max(1, total_steps))

    def __call__(self, step):
        step = keras.ops.cast(step, "float32")
        warmup_steps = keras.ops.cast(max(1, self.warmup_steps), "float32")
        total_steps = keras.ops.cast(self.total_steps, "float32")

        if self.warmup_steps > 0:
            warmup_progress = keras.ops.minimum(1.0, step / warmup_steps)
            warmup_lr = self.peak_lr * warmup_progress
        else:
            warmup_lr = keras.ops.cast(self.peak_lr, "float32")

        decay_start = keras.ops.cast(self.warmup_steps, "float32")
        decay_steps = keras.ops.maximum(1.0, total_steps - decay_start)
        decay_progress = keras.ops.minimum(
            1.0, keras.ops.maximum(0.0, (step - decay_start) / decay_steps)
        )
        cosine_decay = 0.5 * (1.0 + keras.ops.cos(math.pi * decay_progress))
        cosine_lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay

        return keras.ops.where(step < decay_start, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }

# Callbacks

class BestModelSaver(keras.callbacks.Callback):
    

    def __init__(
        self,
        model_path: str,
        monitor: str = "val_accuracy",
        mode: str = "max",
        initial_best: Optional[float] = None,
        verbose: int = 1,
    ):
        super().__init__()
        self.model_path = model_path
        self.monitor = monitor
        self.mode = mode
        self.verbose = int(verbose)
        if initial_best is None:
            self.best = float("-inf") if mode == "max" else float("inf")
        else:
            self.best = float(initial_best)

    def _is_better(self, current: float) -> bool:
        return current > self.best if self.mode == "max" else current < self.best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        current = float(current)
        if self._is_better(current):
            self.best = current
            self.model.save(self.model_path)
            if self.verbose:
                print(
                    f"Saved improved model at epoch {epoch + 1}: "
                    f"{self.monitor}={current:.6f}"
                )

class OverfittingStopper(keras.callbacks.Callback):
    

    def __init__(self, min_gap: float = 0.05, patience: int = 2, verbose: int = 1):
        super().__init__()
        self.min_gap = float(min_gap)
        self.patience = int(patience)
        self.verbose = int(verbose)
        self.bad_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")

        if None in (loss, val_loss, acc, val_acc):
            return

        gap = float(acc) - float(val_acc)
        overfitting_now = (float(val_loss) > float(loss)) and (gap >= self.min_gap)

        if overfitting_now:
            self.bad_epochs += 1
            if self.verbose:
                print(
                    f"Overfitting signal: epoch={epoch + 1}, "
                    f"train_loss={float(loss):.4f}, val_loss={float(val_loss):.4f}, "
                    f"acc_gap={gap:.4f} ({self.bad_epochs}/{self.patience})"
                )
            if self.bad_epochs >= self.patience:
                if self.verbose:
                    print("Stopping training: persistent overfitting detected.")
                self.model.stop_training = True
        else:
            self.bad_epochs = 0

# Loss construction

def build_loss(class_weight: Optional[Dict[int, float]]):
    
    if not USE_FOCAL_LOSS or not class_weight:
        return (
            keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            class_weight,
        )

    alpha = [class_weight[idx] for idx in sorted(class_weight.keys())]
    alpha = np.array(alpha, dtype=np.float32)
    alpha = (alpha / np.sum(alpha)).tolist()
    focal = keras.losses.CategoricalFocalCrossentropy(
        alpha=alpha,
        gamma=FOCAL_GAMMA,
        label_smoothing=max(0.0, LABEL_SMOOTHING * 0.3),
    )
    return focal, None

# Class weighting

def compute_class_weights_from_flow(train_flow) -> Optional[Dict[int, float]]:
    
    classes = np.array(train_flow.classes, dtype=np.int64)
    if classes.size == 0:
        return None

    num_classes = int(classes.max()) + 1
    counts = np.bincount(classes, minlength=num_classes).astype(np.float64)
    if np.any(counts <= 0.0):
        return None

    inv = 1.0 / np.sqrt(counts)
    inv /= float(np.mean(inv))
    inv = np.clip(inv, 0.5, 3.0)
    return {int(idx): float(w) for idx, w in enumerate(inv)}

def compute_class_weights_from_directory(
    train_dir: str, class_names: Sequence[str]
) -> Optional[Dict[int, float]]:
    
    import os

    counts = []
    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            counts.append(0.0)
            continue
        sample_count = sum(1 for entry in os.scandir(class_dir) if entry.is_file())
        counts.append(float(sample_count))

    count_arr = np.array(counts, dtype=np.float64)
    if count_arr.size == 0 or np.any(count_arr <= 0.0):
        return None

    inv = 1.0 / np.sqrt(count_arr)
    inv /= float(np.mean(inv))
    inv = np.clip(inv, 0.5, 3.0)
    return {int(idx): float(w) for idx, w in enumerate(inv)}

# Mixup & cutmix augmentation

def mixup_numpy_batch(
    images: np.ndarray, labels: np.ndarray, alpha: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    
    if alpha <= 0.0 or images.shape[0] < 2:
        return images, labels

    batch_size = images.shape[0]
    lam = np.random.beta(alpha, alpha, size=batch_size).astype(np.float32)
    lam_x = lam.reshape((-1, 1, 1, 1))
    lam_y = lam.reshape((-1, 1))
    indices = np.random.permutation(batch_size)
    mixed_images = images * lam_x + images[indices] * (1.0 - lam_x)
    mixed_labels = labels * lam_y + labels[indices] * (1.0 - lam_y)
    return mixed_images, mixed_labels

def cutmix_numpy_batch(
    images: np.ndarray, labels: np.ndarray, alpha: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    
    if alpha <= 0.0 or images.shape[0] < 2:
        return images, labels

    batch_size = images.shape[0]
    lam = np.random.beta(alpha, alpha)
    indices = np.random.permutation(batch_size)

    h, w = images.shape[1], images.shape[2]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)

    mixed_images = images.copy()
    mixed_images[:, y1:y2, x1:x2, :] = images[indices, y1:y2, x1:x2, :]

    lam_adj = 1.0 - (y2 - y1) * (x2 - x1) / float(h * w)
    mixed_labels = labels * lam_adj + labels[indices] * (1.0 - lam_adj)
    return mixed_images, mixed_labels

def mixup_cutmix_generator(
    base_generator,
    mixup_alpha: float = 0.3,
    cutmix_alpha: float = 1.0,
    use_mixup: bool = True,
    use_cutmix: bool = True,
):
    
    while True:
        images, labels = next(base_generator)
        if use_mixup and use_cutmix:
            if np.random.random() < 0.5:
                images, labels = mixup_numpy_batch(images, labels, alpha=mixup_alpha)
            else:
                images, labels = cutmix_numpy_batch(images, labels, alpha=cutmix_alpha)
        elif use_mixup:
            images, labels = mixup_numpy_batch(images, labels, alpha=mixup_alpha)
        elif use_cutmix:
            images, labels = cutmix_numpy_batch(images, labels, alpha=cutmix_alpha)
        yield images, labels

def mixup_batch_tf(images, labels, alpha: float = 0.2):
    
    import tensorflow as tf

    batch_size = tf.shape(images)[0]
    if alpha <= 0:
        return images, labels

    gamma_1 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    gamma_2 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    lam = gamma_1 / (gamma_1 + gamma_2)

    lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_y = tf.reshape(lam, [batch_size, 1])

    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = images * lam_x + tf.gather(images, indices) * (1.0 - lam_x)
    mixed_labels = labels * lam_y + tf.gather(labels, indices) * (1.0 - lam_y)
    return mixed_images, mixed_labels

# Optimiser

def build_adamw_optimizer(learning_rate):
    
    if OPTIMIZER.lower() != "adamw":
        raise ValueError(f"Unsupported OPTIMIZER '{OPTIMIZER}'. Expected 'AdamW'.")

    optimizer_kwargs = {
        "learning_rate": learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "clipnorm": 1.0,
        "use_ema": USE_OPTIMIZER_EMA,
        "ema_momentum": EMA_MOMENTUM,
    }

    accumulation = int(ACCUMULATION_STEPS)
    if accumulation > 1 and "gradient_accumulation_steps" in inspect.signature(keras.optimizers.AdamW).parameters:
        optimizer_kwargs["gradient_accumulation_steps"] = accumulation
        effective_bs = BATCH_SIZE * accumulation
        print(
            f"Gradient accumulation enabled: steps={accumulation}, "
            f"effective_batch_size={effective_bs}"
        )
    else:
        # If explicitly 1, we omit it to avoid keras validation errors
        # If not supported by the keras version, it is also omitted
        pass

    if USE_OPTIMIZER_EMA:
        print(f"AdamW EMA enabled (momentum={EMA_MOMENTUM}).")
    print(f"AdamW weight_decay={WEIGHT_DECAY}.")
    return keras.optimizers.AdamW(**optimizer_kwargs)

# Misc helpers

def resolve_step_count(config_steps: int, total_samples: int, batch_size: int) -> int:
    
    total_batches = max(1, math.ceil(float(total_samples) / float(batch_size)))
    if int(config_steps) <= 0:
        return total_batches
    return min(int(config_steps), total_batches)

def tensorboard_available() -> bool:
    
    import importlib.util
    return importlib.util.find_spec("tensorboard") is not None
