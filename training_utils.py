from __future__ import annotations

import inspect
import math
from typing import Dict, Optional, Sequence

import tensorflow.keras as keras
import tensorflow as tf
# Provide a compatible `register_keras_serializable` decorator across TF/Keras versions.
try:
    # Preferred import location
    from tensorflow.keras.utils import register_keras_serializable
except Exception:
    try:
        register_keras_serializable = keras.saving.register_keras_serializable  # type: ignore
    except Exception:
        def register_keras_serializable(package=None):
            def decorator(obj):
                return obj
            return decorator
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

@register_keras_serializable(package="training_utils")
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


def count_class_samples_from_directory(
    train_dir: str, class_names: Sequence[str]
) -> tuple[Dict[int, int], int]:
    """Count samples per class directory and return both per-class and total counts."""
    import os

    counts: Dict[int, int] = {}
    total = 0
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            counts[int(class_index)] = 0
            continue
        sample_count = sum(1 for entry in os.scandir(class_dir) if entry.is_file())
        counts[int(class_index)] = int(sample_count)
        total += int(sample_count)
    return counts, int(total)

def compute_class_weights_from_directory(
    train_dir: str, class_names: Sequence[str]
) -> Optional[Dict[int, float]]:
    counts_by_class, _ = count_class_samples_from_directory(train_dir, class_names)
    count_arr = np.array([float(counts_by_class[idx]) for idx in sorted(counts_by_class)], dtype=np.float64)
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


def cutmix_batch_tf(images, labels, alpha: float = 1.0):
    """TensorFlow CutMix for batched image tensors."""
    import tensorflow as tf

    if alpha <= 0:
        return images, labels

    batch_size = tf.shape(images)[0]
    lam = tf.random.gamma(shape=[], alpha=alpha)
    indices = tf.random.shuffle(tf.range(batch_size))

    height = tf.shape(images)[1]
    width = tf.shape(images)[2]
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(height, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(width, tf.float32) * cut_ratio, tf.int32)
    cy = tf.random.uniform([], 0, height, dtype=tf.int32)
    cx = tf.random.uniform([], 0, width, dtype=tf.int32)

    y1 = tf.maximum(0, cy - cut_h // 2)
    y2 = tf.minimum(height, cy + cut_h // 2)
    x1 = tf.maximum(0, cx - cut_w // 2)
    x2 = tf.minimum(width, cx + cut_w // 2)

    row_mask = tf.logical_and(tf.range(height) >= y1, tf.range(height) < y2)
    col_mask = tf.logical_and(tf.range(width) >= x1, tf.range(width) < x2)
    box_mask = tf.cast(tf.logical_and(row_mask[:, None], col_mask[None, :]), images.dtype)
    box_mask = tf.reshape(box_mask, [1, height, width, 1])
    box_mask = tf.broadcast_to(box_mask, tf.shape(images))

    shuffled = tf.gather(images, indices)
    mixed_images = images * (1.0 - box_mask) + shuffled * box_mask

    cut_area = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
    image_area = tf.cast(height * width, tf.float32)
    lam_adj = 1.0 - cut_area / tf.maximum(image_area, 1.0)
    mixed_labels = labels * lam_adj + tf.gather(labels, indices) * (1.0 - lam_adj)
    return mixed_images, mixed_labels


def _build_randaugment_layer(
    num_layers: int,
    magnitude: float,
    value_range: tuple[float, float] = (0.0, 255.0),
):
    """Build a lightweight augmentation stack without external tfds/keras_cv dependencies."""

    magnitude = float(np.clip(magnitude, 0.0, 1.0))
    rotation_factor = 0.08 * magnitude
    translation_factor = 0.10 * magnitude
    zoom_factor = 0.08 * magnitude
    contrast_factor = 0.15 * magnitude
    brightness_factor = 0.10 * magnitude

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(rotation_factor),
            keras.layers.RandomTranslation(translation_factor, translation_factor),
            keras.layers.RandomZoom(-zoom_factor, zoom_factor),
            keras.layers.RandomContrast(contrast_factor),
            keras.layers.Lambda(
                lambda images: tf.clip_by_value(
                    images + tf.random.uniform(tf.shape(images), -brightness_factor, brightness_factor),
                    value_range[0],
                    value_range[1],
                )
            ),
        ],
        name=f"randaugment_like_{int(num_layers)}",
    )


def randaugment_generator(
    base_generator,
    num_layers: int = 2,
    magnitude: float = 9.0,
    value_range: tuple[float, float] = (0.0, 255.0),
):
    """Apply RandAugment to each batch yielded by a numpy generator."""
    layer = _build_randaugment_layer(
        num_layers=int(num_layers),
        magnitude=float(magnitude),
        value_range=value_range,
    )
    while True:
        images, labels = next(base_generator)
        images_tf = tf.convert_to_tensor(images, dtype=tf.float32)
        aug_images = layer(images_tf, training=True)
        yield aug_images.numpy(), labels


def resolve_augmentation_probabilities(
    use_mixup: bool,
    use_cutmix: bool,
    mixup_prob: float,
    cutmix_prob: float,
    normal_prob: float,
) -> tuple[float, float, float]:
    """Resolve and normalize batch routing probabilities."""
    mix = max(0.0, float(mixup_prob)) if use_mixup else 0.0
    cut = max(0.0, float(cutmix_prob)) if use_cutmix else 0.0
    normal = max(0.0, float(normal_prob))
    total = mix + cut + normal
    if total <= 0.0:
        return 0.0, 0.0, 1.0
    return mix / total, cut / total, normal / total


def sample_augmentation_route(
    use_mixup: bool,
    use_cutmix: bool,
    mixup_prob: float,
    cutmix_prob: float,
    normal_prob: float,
) -> str:
    """Sample one augmentation route: mixup, cutmix, or normal."""
    mix_p, cut_p, _ = resolve_augmentation_probabilities(
        use_mixup=use_mixup,
        use_cutmix=use_cutmix,
        mixup_prob=mixup_prob,
        cutmix_prob=cutmix_prob,
        normal_prob=normal_prob,
    )
    route_sample = np.random.random()
    if route_sample < mix_p:
        return "mixup"
    if route_sample < (mix_p + cut_p):
        return "cutmix"
    return "normal"

def mixup_cutmix_generator(
    base_generator,
    mixup_alpha: float = 0.3,
    cutmix_alpha: float = 1.0,
    use_mixup: bool = True,
    use_cutmix: bool = True,
    mixup_prob: float = 0.4,
    cutmix_prob: float = 0.4,
    normal_prob: float = 0.2,
):
    
    while True:
        images, labels = next(base_generator)
        route = sample_augmentation_route(
            use_mixup=use_mixup,
            use_cutmix=use_cutmix,
            mixup_prob=mixup_prob,
            cutmix_prob=cutmix_prob,
            normal_prob=normal_prob,
        )
        if route == "mixup":
            images, labels = mixup_numpy_batch(images, labels, alpha=mixup_alpha)
        elif route == "cutmix":
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
