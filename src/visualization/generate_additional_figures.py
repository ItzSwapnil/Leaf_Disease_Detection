#!/usr/bin/env python3
"""Generate additional figures from the trained model and dataset.

This script produces PNG-only outputs with no synthetic fallback data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import auc, roc_curve
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from src.core.preprocessing import preprocess_batch_for_model_tf
from src.training.training_utils import (
    WarmupCosineSchedule,
    cutmix_numpy_batch,
    mixup_cutmix_generator,
    mixup_numpy_batch,
)
from src.utils.config import (
    CUTMIX_ALPHA,
    FINAL_MODEL_PATH,
    IMG_SIZE,
    LABEL_SMOOTHING,
    MIXUP_ALPHA,
    TRAIN_DIR,
    USE_CUTMIX,
    USE_MIXUP,
    VAL_DIR,
)
from src.utils.model_paths import resolve_keras_model_path

PLOTS_DIR = ROOT / "plots"
DATASET_TRAIN = ROOT / "dataset" / "train"
BASE_DPI = 300
MAX_DPI = 600
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
RNG = random.Random(42)
MODEL_PATH_OVERRIDE: str | None = None


def _save(fig: plt.Figure, name: str) -> None:
    out_path = PLOTS_DIR / name
    min_w, min_h = 7680, 4320
    max_w, max_h = 15000, 11000
    w_in, h_in = fig.get_size_inches()
    if w_in <= 0.0 or h_in <= 0.0:
        raise ValueError("Figure has invalid size.")

    min_needed_dpi = max(min_w / w_in, min_h / h_in)
    max_allowed_dpi = min(MAX_DPI, max_w / w_in, max_h / h_in)
    target_dpi = max(float(BASE_DPI), float(min_needed_dpi))
    target_dpi = min(target_dpi, float(max_allowed_dpi))

    if target_dpi <= 0.0:
        raise ValueError("Unable to derive a valid DPI for figure save.")

    pixel_w = int(round(w_in * target_dpi))
    pixel_h = int(round(h_in * target_dpi))
    if pixel_w < min_w or pixel_h < min_h:
        scale = max(min_w / max(pixel_w, 1), min_h / max(pixel_h, 1))
        fig.set_size_inches(w_in * scale, h_in * scale, forward=True)
        w_in, h_in = fig.get_size_inches()
        max_allowed_dpi = min(MAX_DPI, max_w / w_in, max_h / h_in)
        min_needed_dpi = max(min_w / w_in, min_h / h_in)
        target_dpi = min(max_allowed_dpi, max(float(BASE_DPI), min_needed_dpi))

    if fig.get_constrained_layout():
        fig.set_constrained_layout(True)
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=target_dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT)}")


def _load_train_batch(batch_size: int = 8):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_TRAIN,
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    return next(iter(ds))


def _train_flow_for_labels(batch_size: int = 256):
    """Build a non-augmented training flow for full-label statistics."""
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )
    return image_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )


def _load_training_flow_batch(
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one training-style batch from the exact datagen pipeline."""
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.25,
        brightness_range=(0.7, 1.3),
        shear_range=0.15,
        channel_shift_range=20.0,
        fill_mode="reflect",
    )
    flow = image_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    images_np, labels_np = next(flow)
    idx_to_class = {idx: name for name, idx in flow.class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    return (
        images_np.astype(np.float32),
        labels_np.astype(np.float32),
        class_names,
    )


def _load_augmented_training_batch(
    batch_size: int = 8,
    force_mixup: bool = False,
    force_cutmix: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one batch using the same augmentation generator path as training."""
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.25,
        brightness_range=(0.7, 1.3),
        shear_range=0.15,
        channel_shift_range=20.0,
        fill_mode="reflect",
    )
    flow = image_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    idx_to_class = {idx: name for name, idx in flow.class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    use_mixup = USE_MIXUP
    use_cutmix = USE_CUTMIX
    if force_mixup:
        use_mixup = True
        use_cutmix = False
    if force_cutmix:
        use_mixup = False
        use_cutmix = True

    if not use_mixup and not use_cutmix:
        images_np, labels_np = next(flow)
    else:
        aug_source = mixup_cutmix_generator(
            flow,
            mixup_alpha=float(MIXUP_ALPHA),
            cutmix_alpha=float(CUTMIX_ALPHA),
            use_mixup=use_mixup,
            use_cutmix=use_cutmix,
        )
        images_np, labels_np = next(aug_source)

    return (
        images_np.astype(np.float32),
        labels_np.astype(np.float32),
        class_names,
    )


def _to_display_image(image: np.ndarray) -> np.ndarray:
    """Convert preprocessed tensor to [0, 1] RGB for plotting."""
    img = image.astype(np.float32)
    if img.min() < 0.0:
        img = (img + 1.0) / 2.0
    elif img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _class_names() -> list[str]:
    return [p.name for p in sorted(DATASET_TRAIN.iterdir()) if p.is_dir()]


def _crop_name(label: str) -> str:
    if "___" in label:
        return label.split("___", 1)[0].replace(",", "").replace("_", " ")
    return label.replace("_", " ")


def _disease_name(label: str) -> str:
    if "___" in label:
        return label.split("___", 1)[1].replace("_", " ")
    return label.replace("_", " ")


def _slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip(
        "_"
    )


def _load_same_class_batch(
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a batch from a single class so shape/style remain comparable."""
    class_dirs = [p for p in sorted(DATASET_TRAIN.iterdir()) if p.is_dir()]
    eligible = []
    for class_dir in class_dirs:
        files = [
            f
            for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        if len(files) >= batch_size:
            eligible.append((class_dir, files))
    if not eligible:
        raise SystemExit(
            "No class has enough images for same-class augmentation preview."
        )

    class_dir, files = RNG.choice(eligible)
    chosen = RNG.sample(files, batch_size)

    images = []
    for path in chosen:
        img = (
            Image.open(path)
            .convert("RGB")
            .resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
        )
        images.append(np.asarray(img, dtype=np.float32))
    images_np = np.stack(images, axis=0)

    names = _class_names()
    class_index = names.index(class_dir.name)
    labels_np = np.zeros((batch_size, len(names)), dtype=np.float32)
    labels_np[:, class_index] = 1.0
    return images_np, labels_np, names


def _load_eval_data(split_dir: Path):
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        shuffle=False,
    )
    images = []
    labels = []
    for batch_images, batch_labels in ds:
        images.append(batch_images)
        labels.append(batch_labels)
    return tf.concat(images, axis=0), tf.concat(labels, axis=0), ds.class_names


def _predict_eval_split(
    model, split_dir: Path, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run evaluation prediction in batches to avoid large host/GPU allocations."""
    backbone_name = _infer_backbone_from_model(model)
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=False,
    )

    pred_batches = []
    label_batches = []
    for batch_images, batch_labels in ds:
        batch_preds = model.predict(
            preprocess_batch_for_model_tf(
                batch_images, backbone_name=backbone_name
            ),
            verbose=0,
        )
        pred_batches.append(batch_preds.astype(np.float32))
        label_batches.append(batch_labels.numpy().astype(np.float32))

    predictions = np.concatenate(pred_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    return predictions, labels, ds.class_names


def _patch_vit_layer_init_for_compat() -> bool:
    """Patch keras-hub ViT layer init to ignore legacy serialized kwargs."""
    try:
        from keras_hub.src.models.vit import vit_layers

        layer_cls = vit_layers.ViTPatchingAndEmbedding
    except Exception:
        return False

    if getattr(layer_cls, "_leaf_compat_patched", False):
        return True

    original_init = layer_cls.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.pop("num_patches", None)
        kwargs.pop("num_positions", None)
        return original_init(self, *args, **kwargs)

    layer_cls.__init__ = _patched_init
    layer_cls._leaf_compat_patched = True
    return True


def _load_model():
    path_candidates = (
        [MODEL_PATH_OVERRIDE] if MODEL_PATH_OVERRIDE else [FINAL_MODEL_PATH]
    )
    model_path = resolve_keras_model_path(path_candidates)
    custom_objects = {"WarmupCosineSchedule": WarmupCosineSchedule}
    try:
        return load_model(model_path, custom_objects=custom_objects)
    except TypeError as exc:
        if "ViTPatchingAndEmbedding" not in str(exc):
            raise
        if not _patch_vit_layer_init_for_compat():
            raise RuntimeError(
                "Failed to load ViT/DINO checkpoint due to keras-hub version mismatch."
            ) from exc
        print(
            "Detected KerasHub ViT compatibility mismatch; retrying with shim."
        )
        return load_model(model_path, custom_objects=custom_objects)


def _infer_backbone_from_model(model) -> str:
    name = str(getattr(model, "name", "") or "").lower()
    if "dino" in name or "vit" in name:
        return "DINOv3"

    for layer in model.layers:
        layer_name = str(getattr(layer, "name", "") or "").lower()
        class_name = layer.__class__.__name__.lower()
        if "dino" in layer_name or "vit" in layer_name:
            return "DINOv3"
        if "vit" in class_name:
            return "DINOv3"

    return "EfficientNetV2S"


def _make_cutout(images: np.ndarray, size_ratio: float = 0.35) -> np.ndarray:
    output = images.copy()
    batch_size, height, width = output.shape[:3]
    cut_h = max(1, int(height * size_ratio))
    cut_w = max(1, int(width * size_ratio))
    rng = np.random.default_rng(123)
    for idx in range(batch_size):
        cy = rng.integers(0, height)
        cx = rng.integers(0, width)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(height, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(width, cx + cut_w // 2)
        fill = output[idx].mean(axis=(0, 1), keepdims=True)
        output[idx, y1:y2, x1:x2, :] = fill
    return output


def _class_name(class_names: list[str], index: int) -> str:
    return class_names[int(index)].replace("___", " / ").replace("_", " ")


def _leaf_mask(image: np.ndarray) -> np.ndarray:
    """Estimate a simple leaf foreground mask from RGB image values."""
    img = image.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    # Green-dominant foreground heuristic works well for this dataset.
    mask = (g > 0.12) & (g > r * 0.88) & (g > b * 0.88)
    return mask


def _major_axis_angle_deg(mask: np.ndarray) -> float:
    """Return major-axis angle in degrees relative to x-axis."""
    ys, xs = np.where(mask)
    if ys.size < 50:
        return 90.0
    coords = np.stack([ys.astype(np.float32), xs.astype(np.float32)], axis=1)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, np.argmax(eigvals)]
    # vec = [vy, vx]
    return float(np.degrees(np.arctan2(vec[0], vec[1])))


def _normalized_axis_angle(image: np.ndarray) -> float:
    """Return midrib/major-axis angle in [0, 180)."""
    mask = _leaf_mask(image)
    angle = _major_axis_angle_deg(mask)
    while angle < 0.0:
        angle += 180.0
    while angle >= 180.0:
        angle -= 180.0
    return angle


def _shape_descriptor(image: np.ndarray) -> tuple[float, float, float]:
    """Return (foreground_area_ratio, bbox_aspect_ratio, axis_angle_deg)."""
    mask = _leaf_mask(image)
    area = float(mask.mean())
    ys, xs = np.where(mask)
    if ys.size < 50:
        return area, 1.0, 90.0
    h = float(ys.max() - ys.min() + 1)
    w = float(xs.max() - xs.min() + 1)
    aspect = w / max(h, 1.0)
    angle = _normalized_axis_angle(image)
    return area, aspect, angle


def _best_partner_indices(images: np.ndarray) -> np.ndarray:
    """Find nearest partner by shape and midrib angle (no self-pairs)."""
    desc = np.array(
        [_shape_descriptor(img) for img in images], dtype=np.float32
    )
    n = desc.shape[0]
    partners = np.zeros(n, dtype=np.int32)
    for i in range(n):
        area_diff = np.abs(desc[:, 0] - desc[i, 0])
        aspect_diff = np.abs(desc[:, 1] - desc[i, 1])
        raw_angle = np.abs(desc[:, 2] - desc[i, 2])
        angle_diff = np.minimum(raw_angle, 180.0 - raw_angle) / 180.0
        d = (2.0 * area_diff) + (1.5 * aspect_diff) + (3.0 * angle_diff)
        d[i] = np.inf
        partners[i] = int(np.argmin(d))
    return partners


def _mixup_with_partners(
    images: np.ndarray,
    labels: np.ndarray,
    partners: np.ndarray,
    alpha: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    if alpha <= 0.0 or images.shape[0] < 2:
        return images, labels
    lam = np.random.beta(alpha, alpha, size=images.shape[0]).astype(np.float32)
    lam_x = lam.reshape((-1, 1, 1, 1))
    lam_y = lam.reshape((-1, 1))
    mixed_images = images * lam_x + images[partners] * (1.0 - lam_x)
    mixed_labels = labels * lam_y + labels[partners] * (1.0 - lam_y)
    return mixed_images, mixed_labels


def _cutmix_with_partners(
    images: np.ndarray,
    labels: np.ndarray,
    partners: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if alpha <= 0.0 or images.shape[0] < 2:
        return images, labels

    lam = np.random.beta(alpha, alpha)
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
    mixed_images[:, y1:y2, x1:x2, :] = images[partners, y1:y2, x1:x2, :]
    patch_area = float((y2 - y1) * (x2 - x1))
    lam_adj = 1.0 - patch_area / float(h * w)
    mixed_labels = labels * lam_adj + labels[partners] * (1.0 - lam_adj)
    return mixed_images, mixed_labels


def _load_aligned_train_batch(batch_size: int = 8):
    images_np, labels_np, class_names = _load_same_class_batch(
        batch_size=batch_size
    )
    return images_np, labels_np, class_names


def plot_augmentation_overview() -> None:
    images_np, labels_np, class_names = _load_training_flow_batch(batch_size=8)
    cut_images, cut_labels = cutmix_numpy_batch(
        images_np.copy(), labels_np.copy(), alpha=float(CUTMIX_ALPHA)
    )
    mix_images, mix_labels = mixup_numpy_batch(
        images_np.copy(), labels_np.copy(), alpha=float(MIXUP_ALPHA)
    )

    show_idx = [0, 1]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    col_titles = ["Original", "CutMix", "MixUp"]
    for col_idx, col_name in enumerate(col_titles):
        axes[0, col_idx].set_title(col_name, fontsize=14)

    for row, idx in enumerate(show_idx):
        panels = [
            (images_np[idx], labels_np[idx]),
            (cut_images[idx], cut_labels[idx]),
            (mix_images[idx], mix_labels[idx]),
        ]
        for col, (image, label_vec) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(_to_display_image(image))
            class_idx = int(np.argmax(label_vec))
            ax.text(
                0.02,
                0.98,
                _class_name(class_names, class_idx),
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                ha="left",
                color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 2.0},
            )
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"Sample {row + 1}", fontsize=12)

    fig.suptitle(
        "Augmentation Overview: 2 Originals with their CutMix and MixUp versions",
        fontsize=16,
    )
    _save(fig, "augmentations_overview.png")


def plot_mixup_examples() -> None:
    if not USE_MIXUP:
        print("Skipping mixup figure: USE_MIXUP is disabled in config.")
        return
    images_np, labels_np, class_names = _load_training_flow_batch(batch_size=8)
    mix_images, mix_labels = mixup_numpy_batch(
        images_np.copy(), labels_np.copy(), alpha=float(MIXUP_ALPHA)
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, image, target in zip(axes.flat, mix_images, mix_labels):
        ax.imshow(_to_display_image(image))
        pred_idx = int(np.argmax(target))
        ax.set_title(
            f"MixUp target: {_class_name(class_names, pred_idx)}", fontsize=11
        )
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, "mixup.png")


def plot_cutmix_examples() -> None:
    if not USE_CUTMIX:
        print("Skipping cutmix figure: USE_CUTMIX is disabled in config.")
        return
    images_np, labels_np, class_names = _load_training_flow_batch(batch_size=8)
    cut_images, cut_labels = cutmix_numpy_batch(
        images_np.copy(), labels_np.copy(), alpha=float(CUTMIX_ALPHA)
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, image, target in zip(axes.flat, cut_images, cut_labels):
        ax.imshow(_to_display_image(image))
        pred_idx = int(np.argmax(target))
        ax.set_title(
            f"CutMix target: {_class_name(class_names, pred_idx)}", fontsize=11
        )
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, "cutmix.png")


def plot_cutout_examples() -> None:
    images_np, labels_np, class_names = _load_training_flow_batch(batch_size=8)
    cut_images = _make_cutout(images_np)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, image, target in zip(axes.flat, cut_images, labels_np):
        ax.imshow(_to_display_image(image))
        ax.set_title(
            f"Cutout (not in training): {_class_name(class_names, int(np.argmax(target)))}",
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, "cutout.png")


def plot_label_smoothing() -> None:
    flow = _train_flow_for_labels(batch_size=256)
    class_names = [
        name
        for name, _ in sorted(
            flow.class_indices.items(), key=lambda item: item[1]
        )
    ]
    class_ids = np.asarray(flow.classes, dtype=np.int32)
    num_classes = len(class_names)
    epsilon = float(LABEL_SMOOTHING)

    counts = np.bincount(class_ids, minlength=num_classes).astype(np.float32)
    total = float(np.sum(counts))
    if total <= 0.0:
        raise SystemExit("No training labels found for label smoothing plot.")

    hard_mean = counts / total
    smooth_mean = hard_mean * (1.0 - epsilon) + (epsilon / float(num_classes))

    fig, ax = plt.subplots(figsize=(20, 8))
    indices = np.arange(num_classes)
    ax.bar(
        indices - 0.2,
        hard_mean,
        width=0.4,
        label="One-hot targets (dataset mean)",
        color="#2a9d8f",
    )
    ax.bar(
        indices + 0.2,
        smooth_mean,
        width=0.4,
        label="Smoothed targets (dataset mean)",
        color="#e76f51",
    )
    ax.set_xlim(-1, num_classes)
    ax.set_xticks(indices)
    ax.set_xticklabels(
        [c.replace("___", " / ").replace("_", " ") for c in class_names],
        rotation=75,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Mean target probability")
    ax.set_title(
        f"Label smoothing on full training labels (epsilon={epsilon:.3f})"
    )
    ax.legend(frameon=False)
    _save(fig, "label_smoothing.png")


def plot_calibration_curve() -> None:
    model = _load_model()
    predictions, labels, _ = _predict_eval_split(model, VAL_DIR, batch_size=32)
    confidences = predictions.max(axis=1)
    correct = (predictions.argmax(axis=1) == labels.argmax(axis=1)).astype(
        np.float32
    )

    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(confidences, bins, right=True)
    avg_conf = []
    avg_acc = []
    centers = []
    for idx in range(1, len(bins)):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        avg_conf.append(float(confidences[mask].mean()))
        avg_acc.append(float(correct[mask].mean()))
        centers.append(float((bins[idx - 1] + bins[idx]) / 2.0))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.5)
    ax.plot(
        centers,
        avg_acc,
        marker="o",
        label="Empirical accuracy",
        color="#2a9d8f",
    )
    ax.plot(
        centers, avg_conf, marker="s", label="Mean confidence", color="#e76f51"
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "Validation calibration curve: confidence vs empirical accuracy"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.legend(frameon=False)
    _save(fig, "calibration_curves.png")


def plot_roc_pr_per_class() -> None:
    model = _load_model()
    predictions, y_true, class_names = _predict_eval_split(
        model, VAL_DIR, batch_size=32
    )

    crop_names = sorted({_crop_name(name) for name in class_names})
    per_crop_curves: list[dict] = []

    for crop in crop_names:
        class_indices = [
            index
            for index, name in enumerate(class_names)
            if _crop_name(name) == crop
        ]
        if not class_indices:
            continue

        palette = sns.color_palette(
            "tab20", n_colors=max(3, len(class_indices))
        )
        curves = []
        for color_index, class_index in enumerate(class_indices):
            truth = y_true[:, class_index]
            scores = predictions[:, class_index]
            if np.unique(truth).size < 2:
                continue

            fpr, tpr, _ = roc_curve(truth, scores)
            roc_auc = auc(fpr, tpr)
            class_label = _disease_name(class_names[class_index])
            curves.append(
                {
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": float(roc_auc),
                    "label": class_label,
                    "color": palette[color_index % len(palette)],
                }
            )

        if not curves:
            continue

        per_crop_curves.append({"crop": crop, "curves": curves})

    for crop_data in per_crop_curves:
        crop = crop_data["crop"]
        curves = crop_data["curves"]
        auc_values = [curve["auc"] for curve in curves]

        fig_width = 11 + max(0, len(curves) - 4) * 0.4
        fig, ax = plt.subplots(figsize=(fig_width, 8.5))
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.2,
            color="#8e8e8e",
            alpha=0.8,
        )

        for curve in curves:
            ax.plot(
                curve["fpr"],
                curve["tpr"],
                color=curve["color"],
                linewidth=2.0,
                label=f"{curve['label']} (AUC={curve['auc']:.4f})",
            )

        ax.set_title(
            f"ROC curves for {crop} classes", fontsize=15, fontweight="bold"
        )
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9,
            title="Class (AUC)",
            title_fontsize=10,
        )
        ax.text(
            0.02,
            0.02,
            f"Classes shown: {len(curves)}\nAUC min={min(auc_values):.4f}  mean={float(np.mean(auc_values)):.4f}  max={max(auc_values):.4f}",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.85, "pad": 2.0},
        )
        _save(fig, f"roc_{_slugify(crop)}.png")

    if not per_crop_curves:
        return

    n_crops = len(per_crop_curves)
    n_cols = 3
    n_rows = int(math.ceil(n_crops / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 9, n_rows * 7),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for ax_idx, crop_data in enumerate(per_crop_curves):
        crop = crop_data["crop"]
        curves = crop_data["curves"]
        auc_values = [curve["auc"] for curve in curves]
        axis = axes[ax_idx]

        axis.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.0,
            color="#8e8e8e",
            alpha=0.8,
        )
        for curve in curves:
            axis.plot(
                curve["fpr"],
                curve["tpr"],
                color=curve["color"],
                linewidth=1.6,
                label=f"{curve['label']} ({curve['auc']:.3f})",
            )

        axis.set_title(f"{crop}", fontsize=12, fontweight="bold")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.set_xlabel("FPR", fontsize=9)
        axis.set_ylabel("TPR", fontsize=9)
        axis.tick_params(axis="both", labelsize=8)
        axis.legend(
            loc="lower right",
            frameon=False,
            fontsize=6,
            title="Class (AUC)",
            title_fontsize=7,
        )
        axis.text(
            0.02,
            0.02,
            f"n={len(curves)}  min={min(auc_values):.3f}  mean={float(np.mean(auc_values)):.3f}",
            transform=axis.transAxes,
            fontsize=7,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 1.2},
        )

    for axis in axes[n_crops:]:
        axis.axis("off")

    fig.suptitle(
        "ROC curves grouped by crop (all classes together)",
        fontsize=15,
        fontweight="bold",
    )
    _save(fig, "roc_all_crops_compiled.png")


def plot_training_phase_timeline() -> None:
    logs_dir = ROOT / "models" / "logs"
    latest_runs_path = logs_dir / "latest_runs.json"

    latest_runs = {}
    if latest_runs_path.exists():
        try:
            latest_runs = json.loads(
                latest_runs_path.read_text(encoding="utf-8")
            )
        except Exception:
            latest_runs = {}

    phase_defs = [
        (
            "train",
            "Training",
            logs_dir / "train_history.csv",
            int((latest_runs.get("train") or {}).get("epochs_phase1") or 0)
            + int((latest_runs.get("train") or {}).get("epochs_phase2") or 0),
            1,
        ),
        (
            "fine_tune",
            "Fine-tuning",
            logs_dir / "fine_tune_history.csv",
            int(
                (latest_runs.get("fine_tune") or {}).get("fine_tune_epochs")
                or 0
            ),
            1,
        ),
        (
            "refine",
            "Refining",
            logs_dir / "refine_history.csv",
            int((latest_runs.get("refine") or {}).get("epochs") or 0),
            1,
        ),
    ]

    rows = []
    phase_offsets = {}
    phase_lengths = {}
    cursor = 0

    for phase_key, phase_label, path, _, _ in phase_defs:
        phase_offsets[phase_key] = cursor
        local_rows = []
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    try:
                        local_epoch = int(row["epoch"]) + 1
                        local_rows.append(
                            {
                                "epoch": cursor + local_epoch,
                                "local_epoch": local_epoch,
                                "phase": phase_key,
                                "phase_label": phase_label,
                                "accuracy": float(row.get("accuracy", "nan")),
                                "val_accuracy": float(
                                    row.get("val_accuracy", "nan")
                                ),
                                "loss": float(row.get("loss", "nan")),
                                "val_loss": float(row.get("val_loss", "nan")),
                            }
                        )
                    except Exception:
                        continue
        phase_lengths[phase_key] = len(local_rows)
        rows.extend(local_rows)
        cursor += len(local_rows)

    if not rows:
        return

    epochs = np.array([row["epoch"] for row in rows], dtype=np.float64)
    val_acc = np.array([row["val_accuracy"] for row in rows], dtype=np.float64)
    val_loss = np.array([row["val_loss"] for row in rows], dtype=np.float64)

    global_best_epoch = (
        int(rows[int(np.nanargmax(val_acc))]["epoch"])
        if np.any(np.isfinite(val_acc))
        else None
    )

    phase_best_epochs = {}
    for phase_key, phase_label, _, _, _ in phase_defs:
        phase_rows = [row for row in rows if row["phase"] == phase_key]
        if not phase_rows:
            continue
        best = max(phase_rows, key=lambda row: row["val_accuracy"])
        phase_best_epochs[phase_key] = int(best["epoch"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(
        epochs,
        val_acc,
        color="#d62828",
        linewidth=2.2,
        label="Validation Accuracy",
    )
    ax2.plot(
        epochs,
        val_loss,
        color="#264653",
        linewidth=2.2,
        label="Validation Loss",
    )

    phase_start_colors = {
        "train": "#577590",
        "fine_tune": "#f8961e",
        "refine": "#9c89b8",
    }

    for phase_key, phase_label, _, cfg_epochs, warmup_epochs in phase_defs:
        length = phase_lengths.get(phase_key, 0)
        if length <= 0:
            continue
        start = phase_offsets[phase_key] + 1
        warmup_end = phase_offsets[phase_key] + min(int(warmup_epochs), length)
        phase_best = phase_best_epochs.get(phase_key)

        for axis in (ax1, ax2):
            axis.axvline(
                start,
                color=phase_start_colors.get(phase_key, "#6c757d"),
                linestyle="--",
                alpha=0.7,
                linewidth=1.2,
                label=f"{phase_label} Start",
            )
            axis.axvline(
                warmup_end,
                color="#7b2cbf",
                linestyle="-.",
                alpha=0.7,
                linewidth=1.1,
                label=f"{phase_label} Warmup End",
            )
            if phase_best is not None:
                axis.axvline(
                    phase_best,
                    color="#2a9d8f",
                    linestyle=":",
                    alpha=0.85,
                    linewidth=1.2,
                    label=f"{phase_label} Best",
                )
            if cfg_epochs > 0 and length < cfg_epochs:
                axis.axvline(
                    phase_offsets[phase_key] + length,
                    color="#c1121f",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=1.3,
                    label=f"{phase_label} Restore Event",
                )

    if global_best_epoch is not None:
        for axis in (ax1, ax2):
            axis.axvline(
                global_best_epoch,
                color="#ff6d00",
                linestyle="-",
                alpha=0.9,
                linewidth=1.5,
                label=f"Global Best (epoch {global_best_epoch})",
            )

    ax1.set_title("Validation Accuracy Across All Training Phases")
    ax1.set_xlabel("Global Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Validation Loss Across All Training Phases")
    ax2.set_xlabel("Global Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    dedup = {}
    for handle, label in zip(handles, labels):
        dedup[label] = handle
    ax1.legend(dedup.values(), dedup.keys(), fontsize=8, loc="best")

    _save(fig, "training_phase_timeline.png")


def main() -> None:
    global MODEL_PATH_OVERRIDE
    parser = argparse.ArgumentParser(
        description="Generate additional figures."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit model path for calibration/ROC plots.",
    )
    args = parser.parse_args()
    MODEL_PATH_OVERRIDE = args.model_path

    plot_augmentation_overview()
    plot_mixup_examples()
    plot_cutmix_examples()
    plot_label_smoothing()
    plot_calibration_curve()
    plot_roc_pr_per_class()
    plot_training_phase_timeline()


if __name__ == "__main__":
    main()
