from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import CSVLogger, TensorBoard

from config import (
    FINAL_MODEL_PATH,
    IMG_SIZE,
    INTER_OP_THREADS,
    INTRA_OP_THREADS,
    MIXUP_ALPHA,
    SAVE_LOG_ARCHIVE,
    SAVE_RUN_MANIFESTS,
    TRAIN_DIR,
    USE_MIXUP,
    VAL_DIR,
)
from fine_tune_model import (
    _infer_backbone_from_model,
    _load_model_robust,
    _unfreeze_top_layers,
)
from hardware import configure_tensorflow, get_training_strategy
from preprocessing import preprocess_batch_for_model_tf
from training_progress import IntervalMetricsLogger, ProgressEmitter
from training_utils import (
    RollingPreOverfitRestorer,
    WarmupCosineSchedule,
    build_adamw_optimizer,
    build_loss,
    compute_class_weights_from_directory,
    count_class_samples_from_directory,
    mixup_batch_tf,
    resolve_step_count,
    tensorboard_available,
)

# We deliberately start from the saved classifier and refine that exact model.
REFINED_MODEL_NAME = "leaf_disease_refined.keras"


def _parse_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return int(default)
    try:
        return int(raw_value.strip())
    except Exception:
        return int(default)


def _parse_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return float(default)
    try:
        return float(raw_value.strip())
    except Exception:
        return float(default)


def _parse_path_env(name: str, default: str) -> str:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip()


def _resolve_output_path(default_path: str) -> str:
    output_path = Path(default_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Refine the saved classifier into a new model with rolling pre-overfit restoration."
    )
    parser.add_argument(
        "--model-path",
        default=_parse_path_env("LEAF_REFINE_MODEL_PATH", FINAL_MODEL_PATH),
        help="Source model to refine (defaults to models/leaf_disease_classifier.keras).",
    )
    parser.add_argument(
        "--output-path",
        default=_resolve_output_path(
            _parse_path_env(
                "LEAF_REFINE_OUTPUT_PATH",
                str(Path(FINAL_MODEL_PATH).with_name(REFINED_MODEL_NAME)),
            )
        ),
        help="Where to save the refined model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_parse_int_env("LEAF_REFINE_EPOCHS", 30),
        help="Maximum refinement epochs before early stop or overfit restore.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_parse_int_env("LEAF_REFINE_BATCH_SIZE", 16),
        help="Training batch size for refinement.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=_parse_float_env("LEAF_REFINE_LEARNING_RATE", 2e-5),
        help="Learning rate for refinement.",
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=_parse_float_env("LEAF_REFINE_DATA_FRACTION", 1.0),
        help="Fraction of shuffled training batches to use each epoch.",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=_parse_int_env("LEAF_REFINE_MAX_STEPS_PER_EPOCH", 0),
        help="Optional cap on training steps per epoch.",
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=_parse_int_env("LEAF_REFINE_VAL_MAX_STEPS", 0),
        help="Optional cap on validation steps per epoch.",
    )
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=_parse_int_env("LEAF_REFINE_UNFREEZE_LAYERS", -1),
        help="How many top backbone layers to unfreeze (-1 = full backbone).",
    )
    parser.add_argument(
        "--overfit-gap",
        type=float,
        default=_parse_float_env("LEAF_REFINE_OVERFIT_GAP", 0.0),
        help="Minimum train-accuracy minus val-accuracy gap that counts as overfitting.",
    )
    parser.add_argument(
        "--overfit-patience",
        type=int,
        default=_parse_int_env("LEAF_REFINE_OVERFIT_PATIENCE", 1),
        help="How many consecutive overfitting epochs trigger a restore and stop.",
    )
    parser.add_argument(
        "--snapshot-count",
        type=int,
        default=_parse_int_env("LEAF_REFINE_SNAPSHOT_COUNT", 30),
        help="How many recent safe checkpoints to keep in the rolling restore window.",
    )
    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    seed_env = os.environ.get("RUN_SEED")
    try:
        seed = int(seed_env) if seed_env is not None else 42
    except Exception:
        seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    keras.utils.set_random_seed(seed)
    print(f"Using run seed: {seed}")

    configure_tensorflow()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(INTRA_OP_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)
    except RuntimeError as exc:
        print(f"TensorFlow threading config skipped: {exc}")

    if tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: mixed_float16 (2x memory savings)")
    else:
        keras.mixed_precision.set_global_policy("float32")

    logs_dir = Path(FINAL_MODEL_PATH).parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("Refinement Pipeline")
    print(f"Source model: {args.model_path}")
    print(f"Output model: {args.output_path}")

    strategy = get_training_strategy()
    with strategy.scope():
        model = _load_model_robust(args.model_path)

    detected_backbone = _infer_backbone_from_model(model)
    if detected_backbone == "Unknown":
        raise ValueError(
            "Unable to infer backbone from the loaded model. "
            "Refinement aborted to avoid using incorrect preprocessing."
        )

    latest_runs_path_obj = logs_dir / "latest_runs.json"
    recorded_backbone = None
    if latest_runs_path_obj.exists():
        try:
            with open(latest_runs_path_obj, "r", encoding="utf-8") as in_file:
                latest_runs_data = json.load(in_file)
            recorded_backbone = (latest_runs_data.get("train") or {}).get("base_model")
        except Exception:
            recorded_backbone = None

    print(f"Refine backbone: {detected_backbone}")
    if recorded_backbone:
        print(f"Recorded train backbone: {recorded_backbone}")
        if str(recorded_backbone) != str(detected_backbone):
            print(
                "Backbone mismatch detected; using the loaded model's detected backbone "
                "for preprocessing to ensure correctness."
            )
    print(f"Backbone-locked preprocessing active: {detected_backbone}")

    gpu_count = len(tf.config.list_physical_devices("GPU"))
    batch_env = os.getenv("LEAF_REFINE_BATCH_SIZE")
    batch_size = max(1, int(args.batch_size))
    if batch_env is None and detected_backbone == "DINOv3" and batch_size > 8:
        target_batch = 32 if gpu_count > 1 else 16
        if batch_size > target_batch:
            batch_size = target_batch
            print(
                "Auto-adjusted refine batch size to "
                f"{batch_size} for DINOv3 on {gpu_count} GPU(s). "
                "Override via LEAF_REFINE_BATCH_SIZE if needed."
            )

    if int(args.unfreeze_layers) < 0:
        unfreeze_target = -1
    else:
        unfreeze_target = int(args.unfreeze_layers)
    trainable_count = _unfreeze_top_layers(model, unfreeze_target)
    if unfreeze_target < 0:
        print(f"Unfroze full model ({trainable_count} trainable layers).")
    else:
        print(f"Unfroze {trainable_count} trainable layers (BN kept frozen).")

    autotune = tf.data.AUTOTUNE
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=False,
    )

    train_class_names = (
        list(train_ds.class_names) if getattr(train_ds, "class_names", None) else []
    )
    if not train_class_names:
        train_class_names = sorted(
            entry.name for entry in os.scandir(TRAIN_DIR) if entry.is_dir()
        )

    _, train_samples = count_class_samples_from_directory(
        str(TRAIN_DIR), train_class_names
    )
    _, val_samples = count_class_samples_from_directory(str(VAL_DIR), train_class_names)

    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")

    train_options = tf.data.Options()
    train_options.experimental_deterministic = False
    train_ds = train_ds.with_options(train_options)

    train_ds = train_ds.map(
        lambda images, labels: (
            preprocess_batch_for_model_tf(images, backbone_name=detected_backbone),
            labels,
        ),
        num_parallel_calls=autotune,
    )
    val_ds = val_ds.map(
        lambda images, labels: (
            preprocess_batch_for_model_tf(images, backbone_name=detected_backbone),
            labels,
        ),
        num_parallel_calls=autotune,
    )
    val_ds = val_ds.prefetch(autotune)

    class_weight = compute_class_weights_from_directory(
        str(TRAIN_DIR), train_class_names
    )
    if class_weight:
        print("Class-balanced weighting enabled for refinement.")

    selected_loss, fit_class_weight = build_loss(class_weight)
    if USE_MIXUP and fit_class_weight:
        print("MixUp active: disabling class_weight to avoid label-mix conflicts.")
        fit_class_weight = None
    if USE_MIXUP:
        print(f"MixUp augmentation enabled (alpha={MIXUP_ALPHA}).")

    train_data_fraction = float(args.data_fraction)
    full_steps_per_epoch = resolve_step_count(
        args.max_steps_per_epoch, train_samples, batch_size
    )
    if 0.0 < train_data_fraction < 1.0:
        steps_per_epoch = max(1, int(round(full_steps_per_epoch * train_data_fraction)))
        print(
            "Refinement subset per epoch: "
            f"{steps_per_epoch}/{full_steps_per_epoch} batches "
            f"(fraction={train_data_fraction:.2f}, reshuffled each epoch)"
        )
    else:
        steps_per_epoch = full_steps_per_epoch
    validation_steps = resolve_step_count(
        args.validation_steps, val_samples, batch_size
    )

    if USE_MIXUP:
        train_ds = train_ds.map(
            lambda images, labels: mixup_batch_tf(
                images, labels, alpha=float(MIXUP_ALPHA)
            ),
            num_parallel_calls=autotune,
        )

    train_ds = train_ds.prefetch(autotune)

    total_epochs = max(1, int(args.epochs))
    warmup_steps = max(0, steps_per_epoch * min(1, total_epochs))
    total_steps = max(1, steps_per_epoch * total_epochs)
    lr_schedule = WarmupCosineSchedule(
        peak_lr=float(args.learning_rate),
        min_lr=max(float(args.learning_rate) * 0.1, 1e-7),
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    with strategy.scope():
        model.compile(
            optimizer=build_adamw_optimizer(lr_schedule),
            loss=selected_loss,
            metrics=["accuracy"],
        )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"refine_{run_stamp}"
    snapshot_dir = logs_dir / f"refine_snapshots_{run_stamp}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refine_history_latest = str(logs_dir / "refine_history.csv")
    refine_history_archive = str(logs_dir / f"refine_history_{run_stamp}.csv")
    refine_interval_latest = str(logs_dir / "refine_interval_history.csv")
    refine_interval_archive = str(logs_dir / f"refine_interval_history_{run_stamp}.csv")
    latest_runs_path = str(logs_dir / "latest_runs.json")

    csv_loggers = [CSVLogger(refine_history_latest, append=False)]
    if SAVE_LOG_ARCHIVE:
        csv_loggers.append(CSVLogger(refine_history_archive, append=False))

    interval_loggers = [
        IntervalMetricsLogger(
            refine_interval_latest,
            points_per_epoch=12,
            stage="refine",
            append=False,
            run_id=run_id,
        ),
    ]
    if SAVE_LOG_ARCHIVE:
        interval_loggers.append(
            IntervalMetricsLogger(
                refine_interval_archive,
                points_per_epoch=12,
                stage="refine",
                append=False,
                run_id=run_id,
            ),
        )

    tensorboard = None
    if tensorboard_available():
        tensorboard = TensorBoard(
            log_dir=str(logs_dir / "refine_tensorboard"),
            histogram_freq=1,
            update_freq="epoch",
            write_graph=False,
        )
    else:
        print("TensorBoard not installed; skipping TensorBoard callback.")

    print(f"\n{'=' * 70}")
    print(f"Refinement epochs: {total_epochs}")
    print(
        f"Batch size: {batch_size}  |  Steps/epoch: {steps_per_epoch}  |  Validation steps: {validation_steps}"
    )
    print(
        "Rolling restore: "
        f"last {max(1, int(args.snapshot_count))} safe snapshots, "
        f"gap={float(args.overfit_gap):.3f}, patience={int(args.overfit_patience)} "
        "(strict any-overfit policy)"
    )
    print(f"{'=' * 70}\n")

    run_start_time = time.time()

    progress = ProgressEmitter(
        stage="refinement",
        total_epochs=total_epochs,
        completed_epochs_before=0,
        run_start_time=run_start_time,
    )

    callbacks: list[keras.callbacks.Callback] = [
        *csv_loggers,
        *interval_loggers,
        progress,
        RollingPreOverfitRestorer(
            min_gap=float(args.overfit_gap),
            patience=int(args.overfit_patience),
            snapshot_count=max(1, int(args.snapshot_count)),
            snapshot_dir=str(snapshot_dir),
            strict=True,
            verbose=1,
        ),
    ]
    if tensorboard is not None:
        callbacks.append(tensorboard)

    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=total_epochs,
        callbacks=callbacks,
        class_weight=fit_class_weight,
        verbose=1,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    elapsed = time.time() - run_start_time
    print(f"\nRefinement completed in {elapsed / 3600:.2f} hours")
    print(f"Refined model saved to: {output_path}")

    latest_runs = {}
    if os.path.exists(latest_runs_path):
        try:
            with open(latest_runs_path, "r", encoding="utf-8") as in_file:
                latest_runs = json.load(in_file)
        except Exception:
            latest_runs = {}

    latest_runs["refine"] = {
        "run_id": run_id,
        "run_stamp": run_stamp,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_model": args.model_path,
        "output_model": str(output_path),
        "batch_size": int(batch_size),
        "epochs": total_epochs,
        "steps_per_epoch": int(steps_per_epoch),
        "validation_steps": int(validation_steps),
        "data_fraction": float(train_data_fraction),
        "snapshot_count": int(args.snapshot_count),
        "overfit_gap": float(args.overfit_gap),
        "overfit_patience": int(args.overfit_patience),
        "detected_backbone": detected_backbone,
    }
    if SAVE_RUN_MANIFESTS or SAVE_LOG_ARCHIVE:
        manifest_path = logs_dir / f"refine_run_manifest_{run_stamp}.json"
        with open(manifest_path, "w", encoding="utf-8") as out_file:
            json.dump(latest_runs["refine"], out_file, indent=2)
    with open(latest_runs_path, "w", encoding="utf-8") as out_file:
        json.dump(latest_runs, out_file, indent=2)

    gc.collect()
    keras.backend.clear_session()


if __name__ == "__main__":
    main()
