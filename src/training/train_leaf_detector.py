"""
Train a binary leaf/non-leaf detector using transfer learning.

Uses the same backbone as your disease classifier and trains on your
existing dataset (positives from disease classes, negatives from
background/non-plant images if available).
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from src.utils.config import (
    BATCH_SIZE,
    EPOCHS_PHASE1,
    IMG_SIZE,
    LABEL_SMOOTHING,
    LEARNING_RATE_PHASE1,
    LEARNING_RATE_PHASE2,
    MODELS_DIR,
    TRAIN_DIR,
    VAL_DIR,
)

BINARY_DETECTOR_CHECKPOINT = (
    Path(MODELS_DIR) / "leaf_detector_checkpoint.keras"
)
BINARY_DETECTOR_FINAL = Path(MODELS_DIR) / "leaf_detector_final.keras"


def _build_binary_leaf_detector() -> keras.Model:
    """Build binary classifier (leaf/non-leaf) using EffNetV2 backbone."""
    backbone = keras.applications.EfficientNetV2B0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = False  # Freeze for phase 1

    return keras.Sequential(
        [
            backbone,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),  # Binary output
        ]
    )


def _build_dataset(
    directory: str | Path,
    label: int,
    image_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset | None:
    """
    Build dataset from directory structure.

    Assumes directory contains subdirectories with images.
    For leaf detector: all files are labeled as leaf (1) or non-leaf (0).
    """
    if not Path(directory).exists():
        print(f"[WARNING] Directory does not exist: {directory}")
        return None

    image_paths = list(Path(directory).rglob("*.jpg")) + list(
        Path(directory).rglob("*.png")
    )

    if not image_paths:
        print(f"[WARNING] No images found in {directory}")
        return None

    print(
        f"[*] Found {len(image_paths)} images in {directory} (label={label})"
    )

    def load_and_preprocess(path):
        """Load image, preprocess, and assign label."""
        img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (image_size, image_size))
        img = img / 255.0
        return img, tf.cast(label, tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        [str(p) for p in image_paths]
    ).map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(len(image_paths)).batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def train_leaf_detector(
    epochs_phase1: int = EPOCHS_PHASE1,
    epochs_phase2: int = 5,
    lr_phase1: float = LEARNING_RATE_PHASE1,
    lr_phase2: float = LEARNING_RATE_PHASE2,
) -> keras.Model:
    """
    Train binary leaf detector in two phases.

    Phase 1: Frozen backbone, train head only (fast convergence)
    Phase 2: Unfreeze backbone layers, fine-tune (better accuracy)
    """
    print("=" * 70)
    print("TRAINING BINARY LEAF DETECTOR")
    print("=" * 70)

    # Build model
    print("\n[1/4] Building model...")
    model = _build_binary_leaf_detector()
    model.summary()

    # Load training data (all images in TRAIN_DIR are leaves)
    print(f"\n[2/4] Loading training data from {TRAIN_DIR}...")
    train_ds = _build_dataset(TRAIN_DIR, label=1, batch_size=BATCH_SIZE)

    if train_ds is None:
        print("[ERROR] Could not load training data")
        return None

    # Load validation data (all images in VAL_DIR are leaves)
    print(f"\n[3/4] Loading validation data from {VAL_DIR}...")
    val_ds = _build_dataset(VAL_DIR, label=1, batch_size=BATCH_SIZE)

    if val_ds is None:
        print("[WARNING] Could not load validation data")

    # PHASE 1: Frozen backbone
    print(
        f"\n[4/4] PHASE 1: Training head only (frozen backbone) "
        f"for {epochs_phase1} epochs..."
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_phase1),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["binary_accuracy"],
    )

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        verbose=1,
    )

    # Save checkpoint
    model.save(str(BINARY_DETECTOR_CHECKPOINT))
    print(f"✓ Checkpoint saved: {BINARY_DETECTOR_CHECKPOINT}")

    # PHASE 2: Unfreeze backbone
    print(f"\nPHASE 2: Fine-tuning backbone for {epochs_phase2} epochs...")
    model.layers[0].trainable = True  # Unfreeze backbone

    # Lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_phase2),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["binary_accuracy"],
    )

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase2,
        verbose=1,
    )

    # Save final model
    model.save(str(BINARY_DETECTOR_FINAL))
    print(f"✓ Final model saved: {BINARY_DETECTOR_FINAL}")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(
        f"Phase 1 final accuracy: "
        f"{history_phase1.history['binary_accuracy'][-1]:.3f}"
    )
    if history_phase1.history.get("val_binary_accuracy"):
        print(
            f"Phase 1 val accuracy: "
            f"{history_phase1.history['val_binary_accuracy'][-1]:.3f}"
        )

    print(
        f"Phase 2 final accuracy: "
        f"{history_phase2.history['binary_accuracy'][-1]:.3f}"
    )
    if history_phase2.history.get("val_binary_accuracy"):
        print(
            f"Phase 2 val accuracy: "
            f"{history_phase2.history['val_binary_accuracy'][-1]:.3f}"
        )

    return model


if __name__ == "__main__":
    train_leaf_detector()
