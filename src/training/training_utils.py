from __future__ import annotations

import inspect
import math
import os
import shutil
import tempfile
import zipfile
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Sequence

import tensorflow as tf
import tensorflow.keras as keras

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

from src.utils.config import (
    ACCUMULATION_STEPS,
    BATCH_SIZE,
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST,
    COLOR_JITTER_HUE,
    COLOR_JITTER_SATURATION,
    EMA_MOMENTUM,
    FOCAL_GAMMA,
    GAUSSIAN_BLUR_PROB,
    GAUSSIAN_BLUR_SIGMA_MAX,
    GAUSSIAN_BLUR_SIGMA_MIN,
    GAUSSIAN_NOISE_PROB,
    GAUSSIAN_NOISE_SIGMA,
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
    USE_BACKGROUND_RANDOMIZATION,
    USE_COLOR_JITTER,
    USE_FOCAL_LOSS,
    USE_GAUSSIAN_BLUR,
    USE_GAUSSIAN_NOISE,
    USE_HIERARCHICAL_LOSS,
    USE_OPTIMIZER_EMA,
    USE_RANDOM_ERASING,
    USE_RANDOM_RESIZED_CROP,
    WEIGHT_DECAY,
)

# Learning rate schedule


@register_keras_serializable(package="training_utils")
class WarmupCosineSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        peak_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ):
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


def _normalize_save_mode(save_mode: str) -> str:
    mode = str(save_mode or "with_optimizer").strip().lower().replace("-", "_")
    if mode not in {"with_optimizer", "without_optimizer", "all"}:
        raise ValueError(
            "Unsupported save mode "
            f"'{save_mode}'. Expected one of: with_optimizer, without_optimizer, all."
        )
    return mode


def _without_optimizer_path(model_path: str) -> str:
    base, ext = os.path.splitext(model_path)
    extension = ext or ".keras"
    return f"{base}_no_optimizer{extension}"


def _strip_optimizer_from_keras_archive(model_path: str) -> None:
    """Rewrite a .keras archive and remove the top-level optimizer group from weights."""
    try:
        import h5py
    except Exception:
        # If h5py is unavailable, keep best-effort behavior from clone save.
        return

    if not str(model_path).lower().endswith(".keras"):
        return

    with tempfile.TemporaryDirectory(
        prefix="leaf_strip_optimizer_"
    ) as tmp_dir:
        extracted_weights_path = os.path.join(tmp_dir, "model.weights.h5")
        stripped_weights_path = os.path.join(
            tmp_dir, "model.weights.stripped.h5"
        )
        rebuilt_archive_path = os.path.join(tmp_dir, "rebuilt.keras")
        other_members: dict[str, bytes] = {}

        with zipfile.ZipFile(model_path, "r") as archive:
            member_names = archive.namelist()
            if "model.weights.h5" not in member_names:
                return

            for member_name in member_names:
                if member_name == "model.weights.h5":
                    with (
                        archive.open(member_name, "r") as src,
                        open(extracted_weights_path, "wb") as dst,
                    ):
                        for chunk in iter(
                            lambda: src.read(4 * 1024 * 1024), b""
                        ):
                            dst.write(chunk)
                else:
                    other_members[member_name] = archive.read(member_name)

        with (
            h5py.File(extracted_weights_path, "r") as src_h5,
            h5py.File(stripped_weights_path, "w") as dst_h5,
        ):
            for top_key in src_h5.keys():
                if str(top_key).lower() == "optimizer":
                    continue
                src_h5.copy(top_key, dst_h5, name=top_key)

        with zipfile.ZipFile(
            rebuilt_archive_path, "w", compression=zipfile.ZIP_STORED
        ) as out:
            if "metadata.json" in other_members:
                out.writestr(
                    "metadata.json", other_members.pop("metadata.json")
                )
            if "config.json" in other_members:
                out.writestr("config.json", other_members.pop("config.json"))
            out.write(stripped_weights_path, arcname="model.weights.h5")
            for member_name in sorted(other_members.keys()):
                out.writestr(member_name, other_members[member_name])

        try:
            os.replace(rebuilt_archive_path, model_path)
        except OSError as exc:
            # EXDEV can happen when temp dir and model dir are on different mounts.
            if getattr(exc, "errno", None) != 18:
                raise
            shutil.copy2(rebuilt_archive_path, model_path)
            os.remove(rebuilt_archive_path)


def _save_model_with_include_optimizer(
    model: keras.Model, model_path: str, include_optimizer: bool
) -> None:
    if include_optimizer:
        try:
            model.save(model_path, include_optimizer=True)
            return
        except TypeError:
            # Older Keras variants may not expose include_optimizer.
            model.save(model_path)
            return

    # Avoid clone_model for optimizer-free exports: some backbones (for example
    # ViT variants) may not deserialize cleanly across keras-hub versions.
    # Save directly, then force-strip optimizer tensors from the archive.
    try:
        model.save(model_path, include_optimizer=False)
    except TypeError:
        model.save(model_path)
    _strip_optimizer_from_keras_archive(model_path)


def save_model_variants(
    model: keras.Model, model_path: str, save_mode: str = "with_optimizer"
) -> list[str]:
    mode = _normalize_save_mode(save_mode)

    if mode == "with_optimizer":
        _save_model_with_include_optimizer(
            model, model_path, include_optimizer=True
        )
        return [model_path]

    if mode == "without_optimizer":
        _save_model_with_include_optimizer(
            model, model_path, include_optimizer=False
        )
        return [model_path]

    no_optimizer_path = _without_optimizer_path(model_path)
    _save_model_with_include_optimizer(
        model, model_path, include_optimizer=True
    )
    _save_model_with_include_optimizer(
        model, no_optimizer_path, include_optimizer=False
    )
    return [model_path, no_optimizer_path]


class BestModelSaver(keras.callbacks.Callback):
    def __init__(
        self,
        model_path: str,
        monitor: str = "val_disease_output_accuracy",
        mode: str = "max",
        initial_best: Optional[float] = None,
        verbose: int = 1,
        save_mode: str = "with_optimizer",
    ):
        super().__init__()
        self.model_path = model_path
        self.monitor = monitor
        self.mode = mode
        self.verbose = int(verbose)
        self.save_mode = str(save_mode or "with_optimizer")
        if initial_best is None:
            self.best = float("-inf") if mode == "max" else float("inf")
        else:
            self.best = float(initial_best)

    def _is_better(self, current: float) -> bool:
        return (
            current > self.best if self.mode == "max" else current < self.best
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        current = float(current)
        if self._is_better(current):
            self.best = current
            saved_paths = save_model_variants(
                self.model, self.model_path, save_mode=self.save_mode
            )
            if self.verbose:
                print(
                    f"Saved improved model at epoch {epoch + 1}: "
                    f"{self.monitor}={current:.6f}"
                )
                if len(saved_paths) > 1:
                    print("Saved variants: " + ", ".join(saved_paths))


class OverfittingStopper(keras.callbacks.Callback):
    def __init__(
        self, min_gap: float = 0.05, patience: int = 2, verbose: int = 1
    ):
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
        val_acc = logs.get("val_disease_output_accuracy") or logs.get("val_accuracy")

        if None in (loss, val_loss, acc, val_acc):
            return

        gap = float(acc) - float(val_acc)
        overfitting_now = (float(val_loss) > float(loss)) and (
            gap >= self.min_gap
        )

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
                    print(
                        "Stopping training: persistent overfitting detected."
                    )
                self.model.stop_training = True
        else:
            self.bad_epochs = 0


class PreOverfitRestorer(keras.callbacks.Callback):
    def __init__(
        self,
        min_gap: float = 0.05,
        patience: int = 2,
        verbose: int = 1,
        save_path: Optional[str] = None,
        save_mode: str = "with_optimizer",
    ):
        super().__init__()
        self.min_gap = float(min_gap)
        self.patience = int(patience)
        self.verbose = int(verbose)
        self.save_path = save_path
        self.save_mode = str(save_mode or "with_optimizer")
        self.bad_epochs = 0
        self.last_safe_weights = None
        self.last_safe_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        acc = logs.get("accuracy")
        val_acc = logs.get("val_disease_output_accuracy") or logs.get("val_accuracy")

        if None in (loss, val_loss, acc, val_acc):
            return

        gap = float(acc) - float(val_acc)
        overfitting_now = (float(val_loss) > float(loss)) and (
            gap >= self.min_gap
        )

        if overfitting_now:
            self.bad_epochs += 1
            if self.verbose:
                print(
                    f"Pre-overfit monitor: epoch={epoch + 1}, "
                    f"train_loss={float(loss):.4f}, val_loss={float(val_loss):.4f}, "
                    f"acc_gap={gap:.4f} ({self.bad_epochs}/{self.patience})"
                )

            if self.bad_epochs >= self.patience:
                restored = False
                if self.last_safe_weights is not None:
                    self.model.set_weights(self.last_safe_weights)
                    restored = True
                    if self.verbose:
                        safe_epoch = (
                            (int(self.last_safe_epoch) + 1)
                            if self.last_safe_epoch is not None
                            else "unknown"
                        )
                        print(
                            "Stopping training: overfitting detected. "
                            f"Restored weights from epoch {safe_epoch}."
                        )
                elif self.verbose:
                    print(
                        "Stopping training: overfitting detected before a safe snapshot "
                        "was available."
                    )

                if restored and self.save_path:
                    save_model_variants(
                        self.model, self.save_path, save_mode=self.save_mode
                    )
                    if self.verbose:
                        print(
                            f"Saved restored pre-overfit model to: {self.save_path}"
                        )

                self.model.stop_training = True
        else:
            self.bad_epochs = 0
            self.last_safe_weights = self.model.get_weights()
            self.last_safe_epoch = int(epoch)


class RollingPreOverfitRestorer(keras.callbacks.Callback):
    def __init__(
        self,
        min_gap: float = 0.0,
        patience: int = 1,
        snapshot_count: int = 10,
        snapshot_dir: Optional[str] = None,
        monitor: str = "val_disease_output_accuracy",
        strict: bool = True,
        verbose: int = 1,
    ):
        super().__init__()
        self.min_gap = float(min_gap)
        self.patience = int(patience)
        self.snapshot_count = max(1, int(snapshot_count))
        self.monitor = str(monitor)
        self.strict = bool(strict)
        self.verbose = int(verbose)
        self.bad_epochs = 0
        self.snapshot_dir = Path(
            snapshot_dir or tempfile.mkdtemp(prefix="leaf_refine_snapshots_")
        )
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.safe_snapshots: deque[dict[str, object]] = deque()
        self.best_snapshot_path = self.snapshot_dir / "best_safe.weights.h5"
        self.best_snapshot_epoch: Optional[int] = None
        self.best_snapshot_metric = float("-inf")
        self.initial_snapshot_path = (
            self.snapshot_dir / "initial_safe.weights.h5"
        )
        self._restored = False

    def on_train_begin(self, logs=None):
        # Keep a guaranteed pre-training safe restore point.
        self.model.save_weights(str(self.initial_snapshot_path))

    def _is_overfitting(self, logs: dict) -> bool:
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        acc = logs.get("accuracy")
        val_acc = logs.get("val_disease_output_accuracy") or logs.get("val_accuracy")
        if None in (loss, val_loss, acc, val_acc):
            return False
        gap = float(acc) - float(val_acc)
        loss_overfit = float(val_loss) > float(loss)
        gap_overfit = gap > float(self.min_gap)
        if self.strict:
            return loss_overfit or gap_overfit
        return loss_overfit and gap_overfit

    def _snapshot_path(self, epoch: int, metric: float) -> Path:
        return self.snapshot_dir / (
            f"safe_epoch_{int(epoch) + 1:03d}_{self.monitor}_{float(metric):.6f}.weights.h5"
        )

    def _save_safe_snapshot(self, epoch: int, metric: float) -> None:
        snapshot_path = self._snapshot_path(epoch, metric)
        self.model.save_weights(str(snapshot_path))
        snapshot = {
            "epoch": int(epoch),
            "metric": float(metric),
            "path": str(snapshot_path),
        }
        self.safe_snapshots.append(snapshot)
        if float(metric) >= float(self.best_snapshot_metric):
            self.model.save_weights(str(self.best_snapshot_path))
            self.best_snapshot_metric = float(metric)
            self.best_snapshot_epoch = int(epoch)
        while len(self.safe_snapshots) > self.snapshot_count:
            oldest = self.safe_snapshots.popleft()
            oldest_path = str(oldest["path"])
            try:
                if os.path.exists(oldest_path):
                    os.remove(oldest_path)
            except Exception:
                pass

    def _restore_best_safe_snapshot(self) -> bool:
        if (
            self.best_snapshot_epoch is not None
            and self.best_snapshot_path.exists()
        ):
            self.model.load_weights(str(self.best_snapshot_path))
            self._restored = True
            if self.verbose:
                print(
                    "Stopping training: overfitting detected. "
                    f"Restored best safe weights from epoch {int(self.best_snapshot_epoch) + 1}."
                )
            return True

        if self.initial_snapshot_path.exists():
            self.model.load_weights(str(self.initial_snapshot_path))
            self._restored = True
            if self.verbose:
                print(
                    "Stopping training: overfitting detected. "
                    "Restored initial pre-training safe weights."
                )
            return True

        return False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self._is_overfitting(logs):
            self.bad_epochs += 1
            if self.verbose:
                print(
                    f"Rolling pre-overfit monitor: epoch={epoch + 1}, "
                    f"train_loss={float(logs.get('loss', 0.0)):.4f}, "
                    f"val_loss={float(logs.get('val_loss', 0.0)):.4f}, "
                    f"acc_gap={float(logs.get('disease_output_accuracy', logs.get('accuracy', 0.0))) - float(logs.get('val_disease_output_accuracy', logs.get('val_accuracy', 0.0))):.4f} "
                    f"({self.bad_epochs}/{self.patience})"
                )
            if self.bad_epochs >= self.patience:
                restored = self._restore_best_safe_snapshot()
                if not restored and self.verbose:
                    print(
                        "Stopping training: overfitting detected before a safe snapshot was available."
                    )
                self.model.stop_training = True
        else:
            self.bad_epochs = 0
            monitor_value = logs.get(self.monitor)
            if monitor_value is not None:
                self._save_safe_snapshot(int(epoch), float(monitor_value))

    def on_train_end(self, logs=None):
        if self._restored:
            return
        if (
            self.best_snapshot_epoch is not None
            and self.best_snapshot_path.exists()
        ):
            self.model.load_weights(str(self.best_snapshot_path))
            if self.verbose:
                print(
                    "Training ended without an overfit stop. "
                    f"Restored best safe weights from epoch {int(self.best_snapshot_epoch) + 1}."
                )
            return
        if self.initial_snapshot_path.exists():
            self.model.load_weights(str(self.initial_snapshot_path))
            if self.verbose:
                print(
                    "Training ended without an overfit stop. "
                    "Restored initial pre-training safe weights."
                )


@register_keras_serializable(package="training_utils")
class HierarchicalLoss(keras.losses.Loss):
    def __init__(
        self,
        class_names: list[str],
        label_smoothing: float = 0.15,
        name: str = "hierarchical_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.class_names = list(class_names)
        self.label_smoothing = float(label_smoothing)

        # Parse families and health states
        families = []
        family_to_id = {}
        class_to_family_id = []
        class_is_healthy = []

        for name_str in self.class_names:
            if "___" in name_str:
                family, subclass = name_str.split("___", 1)
            else:
                words = name_str.split()
                family = words[0]
                subclass = " ".join(words[1:])

            if family not in family_to_id:
                family_to_id[family] = len(families)
                families.append(family)

            class_to_family_id.append(family_to_id[family])
            class_is_healthy.append(
                1.0 if "healthy" in subclass.lower() else 0.0
            )

        self.num_classes = len(self.class_names)
        self.num_families = len(families)
        self.class_to_family_id = class_to_family_id
        self.class_is_healthy = class_is_healthy

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        dtype = tf.float32

        # Construct constant tensors locally within the current traced graph context
        family_matrix = tf.one_hot(
            self.class_to_family_id, self.num_families, dtype=dtype
        )
        healthy_mask = tf.constant(self.class_is_healthy, dtype=dtype)
        diseased_mask = 1.0 - healthy_mask
        healthy_family_matrix = family_matrix * healthy_mask[:, tf.newaxis]
        diseased_family_matrix = family_matrix * diseased_mask[:, tf.newaxis]

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # 1. Family Level Loss
        y_true_fam = tf.matmul(y_true, family_matrix)
        y_pred_fam = tf.matmul(y_pred, family_matrix)
        if self.label_smoothing > 0:
            num_fam = tf.cast(self.num_families, tf.float32)
            y_true_fam_smoothed = y_true_fam * (1.0 - self.label_smoothing) + (
                self.label_smoothing / num_fam
            )
        else:
            y_true_fam_smoothed = y_true_fam
        loss_fam = -tf.reduce_sum(
            y_true_fam_smoothed * tf.math.log(y_pred_fam), axis=-1
        )

        # 2. Healthy vs Diseased Level Loss
        y_true_healthy_fam = tf.matmul(y_true, healthy_family_matrix)
        y_pred_healthy_fam = tf.matmul(y_pred, healthy_family_matrix)
        y_true_diseased_fam = tf.matmul(y_true, diseased_family_matrix)
        y_pred_diseased_fam = tf.matmul(y_pred, diseased_family_matrix)

        cond_pred_healthy = y_pred_healthy_fam / (y_pred_fam + 1e-7)
        cond_pred_diseased = y_pred_diseased_fam / (y_pred_fam + 1e-7)
        cond_pred_healthy = tf.clip_by_value(
            cond_pred_healthy, 1e-7, 1.0 - 1e-7
        )
        cond_pred_diseased = tf.clip_by_value(
            cond_pred_diseased, 1e-7, 1.0 - 1e-7
        )

        loss_health = -tf.reduce_sum(
            y_true_healthy_fam * tf.math.log(cond_pred_healthy)
            + y_true_diseased_fam * tf.math.log(cond_pred_diseased),
            axis=-1,
        )

        # 3. Disease Subclass Level Loss
        y_pred_diseased_fam_gathered = tf.gather(
            y_pred_diseased_fam, self.class_to_family_id, axis=-1
        )
        cond_pred_disease = y_pred / (y_pred_diseased_fam_gathered + 1e-7)
        cond_pred_disease = tf.clip_by_value(
            cond_pred_disease, 1e-7, 1.0 - 1e-7
        )

        loss_disease = -tf.reduce_sum(
            y_true * diseased_mask * tf.math.log(cond_pred_disease), axis=-1
        )

        total_loss = loss_fam + loss_health + loss_disease
        return tf.reduce_mean(total_loss)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "class_names": self.class_names,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config


# Loss construction


def build_loss(
    class_weight: Optional[Dict[int, float]],
    class_names: Optional[list[str]] = None,
):

    if USE_HIERARCHICAL_LOSS and class_names:
        return HierarchicalLoss(
            class_names=class_names, label_smoothing=LABEL_SMOOTHING
        ), None

    if not USE_FOCAL_LOSS or not class_weight:
        return (
            keras.losses.CategoricalCrossentropy(
                label_smoothing=LABEL_SMOOTHING
            ),
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
        sample_count = sum(
            1 for entry in os.scandir(class_dir) if entry.is_file()
        )
        counts[int(class_index)] = int(sample_count)
        total += int(sample_count)
    return counts, int(total)


def compute_class_weights_from_directory(
    train_dir: str, class_names: Sequence[str]
) -> Optional[Dict[int, float]]:
    counts_by_class, _ = count_class_samples_from_directory(
        train_dir, class_names
    )
    count_arr = np.array(
        [float(counts_by_class[idx]) for idx in sorted(counts_by_class)],
        dtype=np.float64,
    )
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
    gamma_1 = tf.random.gamma(shape=[], alpha=alpha)
    gamma_2 = tf.random.gamma(shape=[], alpha=alpha)
    lam = gamma_1 / (gamma_1 + gamma_2 + 1e-8)
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
    box_mask = tf.cast(
        tf.logical_and(row_mask[:, None], col_mask[None, :]), images.dtype
    )
    box_mask = tf.reshape(box_mask, [1, height, width, 1])
    box_mask = tf.broadcast_to(box_mask, tf.shape(images))

    shuffled = tf.gather(images, indices)
    mixed_images = images * (1.0 - box_mask) + shuffled * box_mask

    cut_area = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
    image_area = tf.cast(height * width, tf.float32)
    lam_adj = 1.0 - cut_area / tf.maximum(image_area, 1.0)
    mixed_labels = labels * lam_adj + tf.gather(labels, indices) * (
        1.0 - lam_adj
    )
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
            keras.layers.RandomTranslation(
                translation_factor, translation_factor
            ),
            keras.layers.RandomZoom(-zoom_factor, zoom_factor),
            keras.layers.RandomContrast(contrast_factor),
            keras.layers.Lambda(
                lambda images: tf.clip_by_value(
                    tf.cast(images, tf.float32)
                    + tf.random.uniform(
                        tf.shape(images),
                        -brightness_factor,
                        brightness_factor,
                        dtype=tf.float32,
                    ),
                    value_range[0],
                    value_range[1],
                )
            ),
        ],
        name=f"randaugment_like_{int(num_layers)}",
    )


# ── Heavy augmentation layers for background invariance ──────────────


def _random_resized_crop_batch(
    images, target_size, scale_min, scale_max, ratio_min, ratio_max
):
    """Apply random resized crop to a batch of images (TF ops only)."""
    batch_size = tf.shape(images)[0]
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]

    # Sample scale and aspect ratio per image
    scale = tf.random.uniform([batch_size], scale_min, scale_max)
    log_ratio_min = tf.math.log(ratio_min)
    log_ratio_max = tf.math.log(ratio_max)
    log_ratio = tf.random.uniform([batch_size], log_ratio_min, log_ratio_max)
    ratio = tf.exp(log_ratio)

    # Compute crop dimensions
    area = tf.cast(h * w, tf.float32) * scale
    crop_h = tf.cast(tf.math.sqrt(area / ratio), tf.int32)
    crop_w = tf.cast(tf.math.sqrt(area * ratio), tf.int32)
    crop_h = tf.minimum(crop_h, h)
    crop_w = tf.minimum(crop_w, w)

    # Sample crop offsets
    max_offset_h = h - crop_h
    max_offset_w = w - crop_w
    offset_h = tf.cast(
        tf.random.uniform([batch_size], 0.0, 1.0)
        * tf.cast(max_offset_h, tf.float32),
        tf.int32,
    )
    offset_w = tf.cast(
        tf.random.uniform([batch_size], 0.0, 1.0)
        * tf.cast(max_offset_w, tf.float32),
        tf.int32,
    )

    def _crop_single(args):
        img, oh, ow, ch, cw = args
        cropped = tf.image.crop_to_bounding_box(img, oh, ow, ch, cw)
        return tf.image.resize(cropped, [target_size, target_size])

    cropped_images = tf.map_fn(
        _crop_single,
        (images, offset_h, offset_w, crop_h, crop_w),
        fn_output_signature=images.dtype,
    )
    return cropped_images


def _color_jitter_batch(images, brightness, contrast, saturation, hue):
    """Apply color jitter augmentation to a batch (operates in 0-255 range)."""
    orig_dtype = images.dtype
    images_f32 = tf.cast(images, tf.float32)

    # Brightness
    if brightness > 0:
        factor = tf.random.uniform(
            [], -brightness, brightness, dtype=tf.float32
        )
        images_f32 = images_f32 + factor * 255.0

    # Contrast
    if contrast > 0:
        factor = tf.random.uniform(
            [], 1.0 - contrast, 1.0 + contrast, dtype=tf.float32
        )
        mean = tf.reduce_mean(images_f32, axis=[1, 2], keepdims=True)
        images_f32 = (images_f32 - mean) * factor + mean

    # Saturation (convert to HSV, modify S, convert back)
    if saturation > 0:
        factor = tf.random.uniform(
            [], 1.0 - saturation, 1.0 + saturation, dtype=tf.float32
        )
        images_01 = tf.clip_by_value(images_f32 / 255.0, 0.0, 1.0)
        hsv = tf.image.rgb_to_hsv(images_01)
        h_ch, s_ch, v_ch = hsv[..., 0:1], hsv[..., 1:2], hsv[..., 2:3]
        s_ch = tf.clip_by_value(s_ch * factor, 0.0, 1.0)
        hsv_mod = tf.concat([h_ch, s_ch, v_ch], axis=-1)
        images_f32 = tf.image.hsv_to_rgb(hsv_mod) * 255.0

    # Hue
    if hue > 0:
        delta = tf.random.uniform([], -hue, hue, dtype=tf.float32)
        images_01 = tf.clip_by_value(images_f32 / 255.0, 0.0, 1.0)
        hsv = tf.image.rgb_to_hsv(images_01)
        h_ch = hsv[..., 0:1] + delta
        h_ch = h_ch - tf.floor(h_ch)  # Wrap hue around [0, 1]
        hsv_mod = tf.concat([h_ch, hsv[..., 1:2], hsv[..., 2:3]], axis=-1)
        images_f32 = tf.image.hsv_to_rgb(hsv_mod) * 255.0

    images_f32 = tf.clip_by_value(images_f32, 0.0, 255.0)
    return tf.cast(images_f32, orig_dtype)


def _gaussian_blur_batch(images, sigma_min, sigma_max):
    """Apply Gaussian blur using depthwise convolution with a random kernel."""
    sigma = tf.random.uniform([], sigma_min, sigma_max)
    kernel_size = 7  # Fixed kernel size, sigma controls actual blur strength
    ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.reshape(kernel, [kernel_size, kernel_size, 1, 1])
    kernel = tf.tile(kernel, [1, 1, 3, 1])  # One kernel per channel
    kernel = tf.cast(kernel, images.dtype)

    # Pad images for same-size output
    pad = kernel_size // 2
    padded = tf.pad(
        images, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="REFLECT"
    )
    blurred = tf.nn.depthwise_conv2d(
        padded, kernel, strides=[1, 1, 1, 1], padding="VALID"
    )
    return blurred


def _gaussian_noise_batch(images, sigma):
    """Add zero-mean Gaussian noise to a batch of images."""
    noise = tf.random.normal(
        tf.shape(images), mean=0.0, stddev=sigma * 255.0, dtype=images.dtype
    )
    return tf.clip_by_value(images + noise, 0.0, 255.0)


def _random_erasing_batch(images, scale_min, scale_max):
    """Apply random erasing (cutout) to each image in a batch independently."""
    batch_size = tf.shape(images)[0]
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]

    # Sample erase area as fraction of image
    scale = tf.random.uniform([batch_size], scale_min, scale_max)
    area = tf.cast(h * w, tf.float32) * scale
    erase_h = tf.cast(tf.math.sqrt(area), tf.int32)
    erase_w = tf.cast(tf.math.sqrt(area), tf.int32)
    erase_h = tf.minimum(erase_h, h - 1)
    erase_w = tf.minimum(erase_w, w - 1)

    # Random position for erase rectangle
    offset_h = tf.cast(
        tf.random.uniform([batch_size], 0.0, 1.0)
        * tf.cast(h - erase_h, tf.float32),
        tf.int32,
    )
    offset_w = tf.cast(
        tf.random.uniform([batch_size], 0.0, 1.0)
        * tf.cast(w - erase_w, tf.float32),
        tf.int32,
    )

    def _erase_single(args):
        img, oh, ow, eh, ew = args
        # Build a mask: 1 everywhere except the erase rectangle
        rows = tf.range(h)
        cols = tf.range(w)
        row_mask = tf.logical_and(rows >= oh, rows < oh + eh)
        col_mask = tf.logical_and(cols >= ow, cols < ow + ew)
        box_mask = tf.cast(
            tf.logical_and(row_mask[:, None], col_mask[None, :]), img.dtype
        )
        box_mask = box_mask[:, :, None]  # Broadcast over channels
        # Fill erased region with random noise (helps more than zero-fill)
        noise = tf.random.uniform(tf.shape(img), 0.0, 255.0, dtype=img.dtype)
        return img * (1.0 - box_mask) + noise * box_mask

    erased = tf.map_fn(
        _erase_single,
        (images, offset_h, offset_w, erase_h, erase_w),
        fn_output_signature=images.dtype,
    )
    return erased


def _randomize_background_batch_tf(images):
    """Dynamically segment the leaf and randomize the background color.

    This breaks any correlation between background features and class labels.
    """
    # images is (B, H, W, 3) in [0, 255]
    # Compute std deviation across RGB channels
    mean_val = tf.reduce_mean(images, axis=-1, keepdims=True)  # (B, H, W, 1)
    variance = tf.reduce_mean(
        tf.square(images - mean_val), axis=-1, keepdims=True
    )
    std_val = tf.sqrt(variance + 1e-8)  # (B, H, W, 1)

    # Segment leaf foreground
    leaf_mask = tf.cast(
        (std_val > 8.0) & (mean_val > 20.0), dtype=images.dtype
    )  # (B, H, W, 1)

    # Generate random background colors for each image in the batch
    batch_size = tf.shape(images)[0]
    random_colors = tf.random.uniform(
        [batch_size, 1, 1, 3],
        minval=0.0,
        maxval=255.0,
        dtype=images.dtype,
    )

    # Blend image leaf foreground with the random background color
    randomized_images = images * leaf_mask + random_colors * (1.0 - leaf_mask)
    return randomized_images


def _build_heavy_augmentation_layer(value_range=(0.0, 255.0)):
    """Build a comprehensive augmentation pipeline for background invariance.

    Returns a function that takes (images, training) and applies:
    RandomResizedCrop, BackgroundRandomization, Flip, Rotation, ColorJitter,
    GaussianBlur, GaussianNoise, and RandomErasing.
    """

    flip_layer = keras.layers.RandomFlip("horizontal")
    rotation_layer = keras.layers.RandomRotation(0.15)  # ~27 degrees

    def augment(images, training=True):
        if not training:
            return images

        x = tf.cast(images, tf.float32)

        # 1. RandomResizedCrop — destroys background layout
        if USE_RANDOM_RESIZED_CROP:
            x = _random_resized_crop_batch(
                x,
                target_size=IMG_SIZE,
                scale_min=RANDOM_CROP_SCALE_MIN,
                scale_max=RANDOM_CROP_SCALE_MAX,
                ratio_min=RANDOM_CROP_RATIO_MIN,
                ratio_max=RANDOM_CROP_RATIO_MAX,
            )

        # 1b. Background Randomization — prevents shortcut learning
        if USE_BACKGROUND_RANDOMIZATION:
            x = _randomize_background_batch_tf(x)

        # 2. Geometric augmentation (flip + rotation)
        x = flip_layer(x, training=True)
        x = rotation_layer(x, training=True)

        # 3. ColorJitter — prevents color shortcut learning
        if USE_COLOR_JITTER:
            x = _color_jitter_batch(
                x,
                brightness=COLOR_JITTER_BRIGHTNESS,
                contrast=COLOR_JITTER_CONTRAST,
                saturation=COLOR_JITTER_SATURATION,
                hue=COLOR_JITTER_HUE,
            )

        # 4. GaussianBlur (probabilistic)
        if USE_GAUSSIAN_BLUR:
            should_blur = tf.random.uniform([]) < GAUSSIAN_BLUR_PROB
            x = tf.cond(
                should_blur,
                lambda: _gaussian_blur_batch(
                    x, GAUSSIAN_BLUR_SIGMA_MIN, GAUSSIAN_BLUR_SIGMA_MAX
                ),
                lambda: x,
            )

        # 5. GaussianNoise (probabilistic)
        if USE_GAUSSIAN_NOISE:
            should_noise = tf.random.uniform([]) < GAUSSIAN_NOISE_PROB
            x = tf.cond(
                should_noise,
                lambda: _gaussian_noise_batch(x, GAUSSIAN_NOISE_SIGMA),
                lambda: x,
            )

        # 6. RandomErasing / Cutout (probabilistic)
        if USE_RANDOM_ERASING:
            should_erase = tf.random.uniform([]) < RANDOM_ERASING_PROB
            x = tf.cond(
                should_erase,
                lambda: _random_erasing_batch(
                    x, RANDOM_ERASING_SCALE_MIN, RANDOM_ERASING_SCALE_MAX
                ),
                lambda: x,
            )

        return tf.clip_by_value(x, value_range[0], value_range[1])

    return augment


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
            images, labels = mixup_numpy_batch(
                images, labels, alpha=mixup_alpha
            )
        elif route == "cutmix":
            images, labels = cutmix_numpy_batch(
                images, labels, alpha=cutmix_alpha
            )
        yield images, labels


def mixup_batch_tf(images, labels, alpha: float = 0.2):

    import tensorflow as tf

    batch_size = tf.shape(images)[0]
    if alpha <= 0:
        return images, labels

    gamma_1 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    gamma_2 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    lam = gamma_1 / (gamma_1 + gamma_2 + 1e-8)

    lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_y = tf.reshape(lam, [batch_size, 1])

    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = images * lam_x + tf.gather(images, indices) * (1.0 - lam_x)
    mixed_labels = labels * lam_y + tf.gather(labels, indices) * (1.0 - lam_y)
    return mixed_images, mixed_labels


# Optimiser


def _build_adamw_kwargs(learning_rate):

    optimizer_kwargs = {
        "learning_rate": learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "clipnorm": 1.0,
        "use_ema": USE_OPTIMIZER_EMA,
        "ema_momentum": EMA_MOMENTUM,
    }

    accumulation = int(ACCUMULATION_STEPS)
    if (
        accumulation > 1
        and "gradient_accumulation_steps"
        in inspect.signature(keras.optimizers.AdamW).parameters
    ):
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

    return optimizer_kwargs


def build_adamw_optimizer(learning_rate):
    configured = str(OPTIMIZER or "AdamW").strip().lower()
    if configured != "adamw":
        raise ValueError(
            f"Unsupported OPTIMIZER '{OPTIMIZER}'. Expected 'AdamW'."
        )

    optimizer_kwargs = _build_adamw_kwargs(learning_rate)

    if USE_OPTIMIZER_EMA:
        print(f"AdamW EMA enabled (momentum={EMA_MOMENTUM}).")
    print(f"AdamW weight_decay={WEIGHT_DECAY}.")
    return keras.optimizers.AdamW(**optimizer_kwargs)


def build_optimizer(learning_rate, optimizer_name: Optional[str] = None):
    name = str(optimizer_name or OPTIMIZER or "AdamW").strip().lower()

    if name == "adamw":
        optimizer_kwargs = _build_adamw_kwargs(learning_rate)
        if USE_OPTIMIZER_EMA:
            print(f"AdamW EMA enabled (momentum={EMA_MOMENTUM}).")
        print(f"AdamW weight_decay={WEIGHT_DECAY}.")
        return keras.optimizers.AdamW(**optimizer_kwargs)

    if name == "adam":
        if USE_OPTIMIZER_EMA:
            print(
                "Optimizer EMA is only applied for AdamW; skipping for Adam."
            )
        print("Using Adam optimizer.")
        return keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    if name == "sgd":
        if USE_OPTIMIZER_EMA:
            print("Optimizer EMA is only applied for AdamW; skipping for SGD.")
        print("Using SGD optimizer (momentum=0.9, nesterov=True).")
        return keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True,
            clipnorm=1.0,
        )

    if name == "rmsprop":
        if USE_OPTIMIZER_EMA:
            print(
                "Optimizer EMA is only applied for AdamW; skipping for RMSprop."
            )
        print("Using RMSprop optimizer (momentum=0.9).")
        return keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            momentum=0.9,
            clipnorm=1.0,
        )

    raise ValueError(
        "Unsupported optimizer "
        f"'{optimizer_name}'. Expected one of: AdamW, Adam, SGD, RMSprop."
    )


# Misc helpers


def resolve_step_count(
    config_steps: int, total_samples: int, batch_size: int
) -> int:

    total_batches = max(1, math.ceil(float(total_samples) / float(batch_size)))
    if int(config_steps) <= 0:
        return total_batches
    return min(int(config_steps), total_batches)


def tensorboard_available() -> bool:

    import importlib.util

    return importlib.util.find_spec("tensorboard") is not None


def parse_class_structure(class_names: list[str]) -> list[int]:
    """Identify healthy baseline class mapping for family-based learning.

    For each class index, returns the index of its family's healthy baseline
    class, or -1 if the family has no healthy baseline or if the class
    itself is healthy.
    """
    family_of_class = []
    healthy_class_of_family = {}

    # First pass: map classes to families and find healthy class for family
    for idx, name in enumerate(class_names):
        if "___" in name:
            family, subclass = name.split("___", 1)
        else:
            family = name.split()[0]
            subclass = name

        family_of_class.append(family)
        if "healthy" in subclass.lower():
            healthy_class_of_family[family] = idx

    # Second pass: map each class to its healthy partner index
    healthy_partner_indices = []
    for idx, name in enumerate(class_names):
        family = family_of_class[idx]
        partner_idx = healthy_class_of_family.get(family, -1)
        if partner_idx == idx:
            # The class itself is the healthy baseline
            healthy_partner_indices.append(-1)
        else:
            healthy_partner_indices.append(partner_idx)

    return healthy_partner_indices


@register_keras_serializable(package="training_utils")
class FamilyDeviationClassifier(keras.layers.Layer):
    """Custom classifier head that models disease classes as deviations.

    Logits of diseased classes are calculated as: healthy baseline logit +
    learned deviation score.
    """

    def __init__(
        self, num_classes: int, healthy_partners: list[int], **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.healthy_partners = list(healthy_partners)

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(feature_dim, self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.num_classes,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, feature_dim)
        raw_logits = keras.ops.matmul(inputs, self.kernel) + self.bias

        # Vectorized mapping: gather healthy partner logits for each class
        gather_indices = [
            idx if idx != -1 else 0 for idx in self.healthy_partners
        ]
        mask = [1.0 if idx != -1 else 0.0 for idx in self.healthy_partners]

        # Convert to tensors
        gather_indices = keras.ops.convert_to_tensor(
            gather_indices, dtype="int32"
        )
        mask = keras.ops.convert_to_tensor(mask, dtype=raw_logits.dtype)

        # Gather partner logits: shape (batch_size, num_classes)
        partner_logits = keras.ops.take(raw_logits, gather_indices, axis=-1)

        # Add partner logits to raw logits for disease classes
        logits = raw_logits + partner_logits * mask
        return logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "healthy_partners": self.healthy_partners,
            }
        )
        return config


class GradCamEpochCollageCallback(keras.callbacks.Callback):
    """Custom Keras callback to generate and save a Grad-CAM collage after each epoch."""

    def __init__(
        self,
        val_dir: str,
        class_names: list[str],
        output_dir: str = "plots/gradcam_epochs",
        backbone_name: str = "DINOv3",
    ):
        super().__init__()
        self.val_dir = val_dir
        self.class_names = class_names
        self.output_dir = output_dir
        self.backbone_name = backbone_name
        self.representative_samples = []

    def on_train_begin(self, logs=None):
        import os
        import random

        from tensorflow.keras.utils import img_to_array, load_img

        from src.core.preprocessing import preprocess_array_for_model

        print(
            "\n[GradCamEpochCollageCallback] Preparing representative "
            "validation images for Grad-CAM collage..."
        )

        self.representative_samples = []
        for class_name in self.class_names:
            class_path = os.path.join(self.val_dir, class_name)
            if not os.path.isdir(class_path):
                # Fallback to case insensitive match
                found_dir = None
                if os.path.exists(self.val_dir):
                    for d in os.listdir(self.val_dir):
                        cleaned_d = d.lower().replace("_", "").replace(",", "")
                        cleaned_cls = (
                            class_name.lower()
                            .replace("_", "")
                            .replace(",", "")
                        )
                        if cleaned_d == cleaned_cls:
                            found_dir = os.path.join(self.val_dir, d)
                            break
                if found_dir:
                    class_path = found_dir
                else:
                    print(
                        f"Warning: Directory not found for class: {class_name}"
                    )
                    continue

            # Get all images in directory
            fnames = [
                f
                for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
            if not fnames:
                continue

            # Select one image randomly
            fname = random.choice(fnames)
            img_path = os.path.join(class_path, fname)

            try:
                # Load and preprocess
                img = load_img(img_path, target_size=(224, 224))
                img_array = img_to_array(img)
                preprocessed = preprocess_array_for_model(
                    img_array[np.newaxis, ...],
                    backbone_name=self.backbone_name,
                )
                self.representative_samples.append(
                    {
                        "class_name": class_name,
                        "img_path": img_path,
                        "original_img": img_array,
                        "preprocessed": preprocessed,
                    }
                )
            except Exception as e:
                print(
                    f"Warning: Failed to load image {img_path} for "
                    f"collage: {e}"
                )

        print(
            f"[GradCamEpochCollageCallback] Loaded "
            f"{len(self.representative_samples)} validation samples "
            f"for collage."
        )

    def on_epoch_end(self, epoch, logs=None):
        if not self.representative_samples:
            return

        import os

        import matplotlib.pyplot as plt
        import numpy as np

        from scripts.gradcam_check import (
            _make_gradcam_heatmap,
            _overlay_heatmap,
        )

        os.makedirs(self.output_dir, exist_ok=True)

        # Dynamically size the grid based on number of samples
        cols = 8
        rows = int(np.ceil(len(self.representative_samples) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        fig.suptitle(
            f"Grad-CAM Epoch Collage - Epoch {epoch + 1}",
            fontsize=24,
            y=0.98,
        )

        target_layer_name = None

        # Generate overlays
        for idx, sample in enumerate(self.representative_samples):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]

            class_name = sample["class_name"]
            preprocessed = sample["preprocessed"]
            original_img = sample["original_img"]

            try:
                # Get predictions
                preds = self.model.predict(preprocessed, verbose=0)
                if isinstance(preds, dict):
                    disease_preds = preds["disease_output"]
                else:
                    disease_preds = preds
                    
                pred_idx = int(np.argmax(disease_preds[0]))
                pred_label = self.class_names[pred_idx]
                pred_conf = float(disease_preds[0][pred_idx])

                # Generate Grad-CAM for the predicted class
                crop_heatmap, disease_heatmap = _make_gradcam_heatmap(
                    model=self.model,
                    img_array=preprocessed,
                    target_layer_name=target_layer_name,
                    pred_index=pred_idx,
                    backbone_name=self.backbone_name,
                    vit_block_idx=6,
                )

                # Overlay crop and disease separately
                overlay1 = _overlay_heatmap(original_img, crop_heatmap, alpha=0.3, colormap="viridis")
                overlay = _overlay_heatmap(overlay1, disease_heatmap, alpha=0.5, colormap="jet")

                # Display image
                ax.imshow(overlay)

                # Add title
                lbl_pred = pred_label.lower().replace("_", "").replace(",", "")
                lbl_true = class_name.lower().replace("_", "").replace(",", "")
                is_correct = lbl_pred == lbl_true
                color = "green" if is_correct else "red"
                short_true = class_name.split("___")[-1][:12]
                short_pred = pred_label.split("___")[-1][:12]
                ax.set_title(
                    f"T: {short_true}\nP: {short_pred} ({pred_conf:.2f})",
                    fontsize=8,
                    color=color,
                )
            except Exception:
                ax.text(
                    0.5,
                    0.5,
                    "Failed",
                    ha="center",
                    va="center",
                    color="red",
                )
                ax.set_title(f"Err: {class_name[:12]}", fontsize=8)

            ax.axis("off")

        # Hide any unused subplots
        for idx in range(len(self.representative_samples), rows * cols):
            r = idx // cols
            c = idx % cols
            axes[r, c].axis("off")

        plt.tight_layout()
        collage_path = os.path.join(
            self.output_dir, f"epoch_{epoch + 1:03d}.png"
        )
        plt.savefig(collage_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        print(
            f"\n[GradCamEpochCollageCallback] Saved Grad-CAM collage "
            f"to: {collage_path}"
        )
