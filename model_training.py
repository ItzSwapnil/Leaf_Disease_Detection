import os
import json
import random
import time
import math
import keras
import tensorflow as tf
import numpy as np
from datetime import datetime

from keras.applications import (
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2S,
    EfficientNetV2M,
    EfficientNetV2L,
)
from keras.applications.efficientnet_v2 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger

from hardware import configure_tensorflow, get_training_strategy
from training_progress import ProgressEmitter, IntervalMetricsLogger
from training_utils import (
    BestModelSaver,
    WarmupCosineSchedule,
    build_adamw_optimizer,
    build_loss,
    compute_class_weights_from_flow,
    mixup_cutmix_generator,
    resolve_step_count,
    tensorboard_available,
)
from config import (
    IMG_SIZE, BATCH_SIZE, STEPS_PER_EPOCH, VALIDATION_STEPS,
    EPOCHS_PHASE1, EPOCHS_PHASE2, TRAIN_DIR, VAL_DIR,
    CHECKPOINT_PATH, INTRA_OP_THREADS, INTER_OP_THREADS,
    DENSE_UNITS, DROPOUT_RATE, NUM_CLASSES, LABEL_SMOOTHING,
    LEARNING_RATE_PHASE1, LEARNING_RATE_PHASE2,
    EARLY_STOPPING_PATIENCE, UNFREEZE_LAYERS,
    CLASS_INDICES_PATH, BASE_MODEL,
    SAVE_LOG_ARCHIVE, SAVE_RUN_MANIFESTS,
    USE_MIXUP, MIXUP_ALPHA,
    WARMUP_EPOCHS,
)

# Import optional sota augmentation flags with safe fallbacks
try:
    from config import USE_CUTMIX, CUTMIX_ALPHA
except ImportError:
    USE_CUTMIX = False
    CUTMIX_ALPHA = 1.0

# Backbone registry

_BACKBONE_REGISTRY = {
    "EfficientNetV2B0": EfficientNetV2B0,
    "EfficientNetV2B1": EfficientNetV2B1,
    "EfficientNetV2B2": EfficientNetV2B2,
    "EfficientNetV2B3": EfficientNetV2B3,
    "EfficientNetV2S": EfficientNetV2S,
    "EfficientNetV2M": EfficientNetV2M,
    "EfficientNetV2L": EfficientNetV2L,
}

def _resolve_backbone_factory(name: str):
    
    if name in _BACKBONE_REGISTRY:
        return _BACKBONE_REGISTRY[name]
    supported = ", ".join(sorted(_BACKBONE_REGISTRY.keys()))
    raise ValueError(
        f"Supported backbones: {supported}. "
    )

# Main training entrypoint

def main():
    
        print("EfficientNetV2-S  |  SOTA Training Pipeline")
    print("Target: 99%+ top-1 accuracy on PlantVillage-46")
    
    # Reproducibility
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)
    keras.utils.set_random_seed(42)
    configure_tensorflow()



    # Mixed precision: halves gpu memory footprint with negligible accuracy loss
    if tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: mixed_float16 (2x memory savings)")
    else:
        keras.mixed_precision.set_global_policy("float32")

    tf.config.threading.set_intra_op_parallelism_threads(INTRA_OP_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    logs_dir = os.path.join(os.path.dirname(CHECKPOINT_PATH), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # ── data loading ──────────────────────────────────────────────────────
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

    train_datagen = ImageDataGenerator(
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
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    print(f"\nLoading training data from: {TRAIN_DIR}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}  |  Batch size: {BATCH_SIZE}")

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Number of classes: {NUM_CLASSES}")

    # ── model construction ────────────────────────────────────────────────
    strategy = get_training_strategy()
    with strategy.scope():
        backbone_factory = _resolve_backbone_factory(BASE_MODEL)
        base_model = backbone_factory(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization(dtype="float32")(x)
        x = Dense(DENSE_UNITS, activation="swish")(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(DENSE_UNITS // 2, activation="swish")(x)
        x = Dropout(DROPOUT_RATE * 0.5)(x)
        outputs = Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
        model = Model(inputs=base_model.input, outputs=outputs)

    print(f"\nBackbone: {BASE_MODEL}")
    print(f"Total parameters: {model.count_params():,}")
    trainable_params = sum(p.numpy().size for p in model.trainable_weights)
    print(f"Trainable parameters (phase 1): {trainable_params:,}")

    steps_per_epoch = resolve_step_count(STEPS_PER_EPOCH, train_gen.samples, BATCH_SIZE)
    validation_steps = resolve_step_count(VALIDATION_STEPS, val_gen.samples, BATCH_SIZE)

    phase1_epochs = max(0, int(EPOCHS_PHASE1))
    phase2_epochs = max(0, int(EPOCHS_PHASE2))
    total_epochs = phase1_epochs + phase2_epochs

    # Persist class-to-index mapping for inference
    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as class_file:
        json.dump(train_gen.class_indices, class_file, indent=2)

    # ── class weighting and loss ──────────────────────────────────────────
    class_weight = compute_class_weights_from_flow(train_gen)
    if class_weight:
        print("Class-balanced weighting enabled (inverse-sqrt frequency).")
        class_names_by_idx = {idx: name for name, idx in train_gen.class_indices.items()}
        top = sorted(class_weight.items(), key=lambda item: item[1], reverse=True)[:6]
        print(
            "Highest class weights: "
            + ", ".join(f"{class_names_by_idx.get(idx, idx)}={w:.2f}" for idx, w in top)
        )

    selected_loss, fit_class_weight = build_loss(class_weight)

    # ── mixup / cutmix ───────────────────────────────────────────────────
    train_source = train_gen
    if USE_MIXUP or USE_CUTMIX:
        augmentation_desc = []
        if USE_MIXUP:
            augmentation_desc.append(f"MixUp(alpha={MIXUP_ALPHA})")
        if USE_CUTMIX:
            augmentation_desc.append(f"CutMix(alpha={CUTMIX_ALPHA})")
        print(f"Augmentation: {' + '.join(augmentation_desc)}")
        train_source = mixup_cutmix_generator(
            train_gen,
            mixup_alpha=float(MIXUP_ALPHA),
            cutmix_alpha=float(CUTMIX_ALPHA),
            use_mixup=USE_MIXUP,
            use_cutmix=USE_CUTMIX,
        )
        if fit_class_weight:
            print("MixUp/CutMix active: disabling class_weight to avoid label-mix conflicts.")
            fit_class_weight = None

    # ── callbacks ─────────────────────────────────────────────────────────
    checkpoint = BestModelSaver(
        CHECKPOINT_PATH, monitor="val_accuracy", mode="max", verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=EARLY_STOPPING_PATIENCE,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"train_{run_stamp}"
    train_history_latest_path = os.path.join(logs_dir, "train_history.csv")
    train_history_archive_path = os.path.join(logs_dir, f"train_history_{run_stamp}.csv")
    train_interval_latest_path = os.path.join(logs_dir, "train_interval_history.csv")
    train_interval_archive_path = os.path.join(logs_dir, f"train_interval_history_{run_stamp}.csv")
    latest_runs_path = os.path.join(logs_dir, "latest_runs.json")

    csv_loggers_phase1 = [CSVLogger(train_history_latest_path, append=False)]
    if SAVE_LOG_ARCHIVE:
        csv_loggers_phase1.append(CSVLogger(train_history_archive_path, append=False))
    csv_loggers_phase2 = [CSVLogger(train_history_latest_path, append=True)]
    if SAVE_LOG_ARCHIVE:
        csv_loggers_phase2.append(CSVLogger(train_history_archive_path, append=True))

    interval_loggers_phase1 = [
        IntervalMetricsLogger(
            train_interval_latest_path, points_per_epoch=12,
            stage="train_full", append=False, run_id=run_id,
        ),
    ]
    if SAVE_LOG_ARCHIVE:
        interval_loggers_phase1.append(
            IntervalMetricsLogger(
                train_interval_archive_path, points_per_epoch=12,
                stage="train_full", append=False, run_id=run_id,
            )
        )
    interval_loggers_phase2 = [
        IntervalMetricsLogger(
            train_interval_latest_path, points_per_epoch=12,
            stage="train_full", append=True, run_id=run_id,
        ),
    ]
    if SAVE_LOG_ARCHIVE:
        interval_loggers_phase2.append(
            IntervalMetricsLogger(
                train_interval_archive_path, points_per_epoch=12,
                stage="train_full", append=True, run_id=run_id,
            )
        )

    print(f"Training history log: {train_history_latest_path}")
    print(f"Interval history log: {train_interval_latest_path}")
    if SAVE_LOG_ARCHIVE:
        print(f"Archive training log: {train_history_archive_path}")
        print(f"Archive interval log: {train_interval_archive_path}")

    tensorboard = None
    if tensorboard_available():
        tensorboard = TensorBoard(
            log_dir=os.path.join(logs_dir, "tensorboard"),
            histogram_freq=1, update_freq="epoch", write_graph=False,
        )
    else:
        print("TensorBoard not installed; skipping TensorBoard callback.")

    print(f"\n{'=' * 70}")
    print(f"Phase 1 (head-only): {phase1_epochs} epochs")
    print(f"Phase 2 (full fine-tune): {phase2_epochs} epochs")
    print(f"Total epochs: {total_epochs}  |  Steps/epoch: {steps_per_epoch}")
    print(f"Optimiser: AdamW  |  Label smoothing: {LABEL_SMOOTHING}")
    print(f"{'=' * 70}\n")

    run_start_time = time.time()

    # ── phase 1: train classification head only ───────────────────────────
    if phase1_epochs > 0:
        phase1_total_steps = max(1, steps_per_epoch * phase1_epochs)
        phase1_warmup_steps = max(0, steps_per_epoch * min(int(WARMUP_EPOCHS), phase1_epochs))
        phase1_lr = WarmupCosineSchedule(
            peak_lr=LEARNING_RATE_PHASE1,
            min_lr=max(LEARNING_RATE_PHASE1 * 0.01, 1e-7),
            warmup_steps=phase1_warmup_steps,
            total_steps=phase1_total_steps,
        )

        with strategy.scope():
            model.compile(
                optimizer=build_adamw_optimizer(phase1_lr),
                loss=selected_loss,
                metrics=["accuracy"],
            )

        progress_phase1 = ProgressEmitter(
            stage="phase1_warmup",
            total_epochs=total_epochs,
            completed_epochs_before=0,
            run_start_time=run_start_time,
        )

        print(f"\n--- Phase 1: warm-up on {train_gen.samples} training images ---")
        phase1_callbacks: list[keras.callbacks.Callback] = [
            checkpoint, early_stopping,
            *csv_loggers_phase1, *interval_loggers_phase1,
            progress_phase1,
        ]
        if tensorboard is not None:
            phase1_callbacks.append(tensorboard)

        phase1_history = model.fit(
            train_source,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=phase1_epochs,
            callbacks=phase1_callbacks,
            class_weight=fit_class_weight,
            verbose=1,
        )
        completed_phase1_epochs = len(phase1_history.history.get("loss", []))
    else:
        completed_phase1_epochs = 0

    # ── phase 2: unfreeze backbone and fine-tune ──────────────────────────
    if phase2_epochs > 0:
        base_model.trainable = True
        if int(UNFREEZE_LAYERS) > 0:
            for layer in base_model.layers[: -int(UNFREEZE_LAYERS)]:
                layer.trainable = False
            print(f"Unfroze top {UNFREEZE_LAYERS} backbone layers.")
        else:
            print("Unfroze entire backbone for fine-tuning.")

        # Keep batchnormalization layers frozen for training stability
        bn_frozen = 0
        for layer in base_model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
                bn_frozen += 1
        if bn_frozen > 0:
            print(f"Froze {bn_frozen} BatchNormalization layers for stability.")

        phase2_total_steps = max(1, steps_per_epoch * phase2_epochs)
        phase2_warmup_steps = max(0, steps_per_epoch * min(int(WARMUP_EPOCHS), phase2_epochs))
        phase2_lr = WarmupCosineSchedule(
            peak_lr=LEARNING_RATE_PHASE2,
            min_lr=max(LEARNING_RATE_PHASE2 * 0.01, 1e-7),
            warmup_steps=phase2_warmup_steps,
            total_steps=phase2_total_steps,
        )

        with strategy.scope():
            model.compile(
                optimizer=build_adamw_optimizer(phase2_lr),
                loss=selected_loss,
                metrics=["accuracy"],
            )

        total_epochs_phase2_view = max(1, completed_phase1_epochs + phase2_epochs)
        progress_phase2 = ProgressEmitter(
            stage="phase2_finetune",
            total_epochs=total_epochs_phase2_view,
            completed_epochs_before=completed_phase1_epochs,
            run_start_time=run_start_time,
        )

        print("\n--- Phase 2: fine-tuning entire network ---")
        phase2_callbacks: list[keras.callbacks.Callback] = [
            checkpoint, early_stopping,
            *csv_loggers_phase2, *interval_loggers_phase2,
            progress_phase2,
        ]
        if tensorboard is not None:
            phase2_callbacks.append(tensorboard)

        model.fit(
            train_source,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            initial_epoch=completed_phase1_epochs,
            epochs=completed_phase1_epochs + phase2_epochs,
            callbacks=phase2_callbacks,
            class_weight=fit_class_weight,
            verbose=1,
        )

    elapsed = time.time() - run_start_time
    print(f"\nTraining completed in {elapsed / 3600:.2f} hours")

    # ── run manifest ──────────────────────────────────────────────────────
    latest_runs = {}
    if os.path.exists(latest_runs_path):
        try:
            with open(latest_runs_path, "r", encoding="utf-8") as in_file:
                latest_runs = json.load(in_file)
        except Exception:
            latest_runs = {}

    train_manifest = {
        "run_id": run_id,
        "run_stamp": run_stamp,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_history_latest": train_history_latest_path,
        "train_history_archive": train_history_archive_path if SAVE_LOG_ARCHIVE else None,
        "train_interval_latest": train_interval_latest_path,
        "train_interval_archive": train_interval_archive_path if SAVE_LOG_ARCHIVE else None,
        "base_model": BASE_MODEL,
        "use_mixup": USE_MIXUP,
        "use_cutmix": USE_CUTMIX,
        "epochs_phase1": phase1_epochs,
        "epochs_phase2": phase2_epochs,
    }
    latest_runs["train"] = train_manifest

    if SAVE_RUN_MANIFESTS or SAVE_LOG_ARCHIVE:
        with open(
            os.path.join(logs_dir, f"train_run_manifest_{run_stamp}.json"),
            "w", encoding="utf-8",
        ) as out_file:
            json.dump(train_manifest, out_file, indent=2)
    with open(latest_runs_path, "w", encoding="utf-8") as out_file:
        json.dump(latest_runs, out_file, indent=2)

    return model

if __name__ == "__main__":
    main()