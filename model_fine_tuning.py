import gc
import random
import json
import os
import time
from datetime import datetime

import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model

from config import (
    ACCUMULATION_STEPS,
    CHECKPOINT_PATH,
    EMA_MOMENTUM,
    FINAL_MODEL_PATH,
    FINE_TUNE_BATCH_SIZE,
    FINE_TUNE_DATA_FRACTION,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    FINE_TUNE_MAX_STEPS_PER_EPOCH,
    FINE_TUNE_UNFREEZE_LAYERS,
    FINE_TUNE_VAL_MAX_STEPS,
    IMG_SIZE,
    INTER_OP_THREADS,
    INTRA_OP_THREADS,
    LR_SCHEDULER,
    SAVE_LOG_ARCHIVE,
    SAVE_RUN_MANIFESTS,
    TRAIN_DIR,
    UNFREEZE_LAYERS,
    USE_MIXUP,
    MIXUP_ALPHA,
    USE_OPTIMIZER_EMA,
    VAL_DIR,
    WEIGHT_DECAY,
)
from hardware import configure_tensorflow, get_training_strategy
from preprocessing import preprocess_batch_for_model
from training_progress import IntervalMetricsLogger, ProgressEmitter
from training_utils import (
    BestModelSaver,
    OverfittingStopper,
    WarmupCosineSchedule,
    build_loss,
    compute_class_weights_from_directory,
    mixup_batch_tf,
    tensorboard_available,
)

# Internal helpers

def _resolve_model_path(*candidates: str) -> str:
    
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"No model file found in candidates: {candidates}")

def _evaluate_val_accuracy(model_path, val_ds, validation_steps, strategy):
    
    with strategy.scope():
        eval_model = load_model(model_path)
    metrics = eval_model.evaluate(
        val_ds, steps=validation_steps, verbose=0, return_dict=True
    )
    acc = float(metrics.get("accuracy", 0.0))
    keras.backend.clear_session()
    gc.collect()
    return acc

def _cardinality_int(ds):
    
    card = tf.data.experimental.cardinality(ds)
    if card in (
        tf.data.experimental.UNKNOWN_CARDINALITY,
        tf.data.experimental.INFINITE_CARDINALITY,
    ):
        return None
    return int(card.numpy())

def _prepare_subset(ds, fraction: float, max_batches: int, dataset_name: str):
    
    total_batches = _cardinality_int(ds)
    max_batches = int(max_batches)

    if total_batches is None or total_batches <= 0:
        chosen_batches = max(1, max_batches) if max_batches > 0 else 1
    else:
        if max_batches <= 0:
            chosen_batches = total_batches
        else:
            chosen_batches = max(1, int(round(total_batches * float(fraction))))
            chosen_batches = min(chosen_batches, max_batches, total_batches)

    print(
        f"  {dataset_name} subset: {chosen_batches} batch(es) "
        f"(fraction={fraction:.2f}, max={max_batches})"
    )
    return ds.take(chosen_batches).repeat(), chosen_batches

def _unfreeze_top_layers(model, target_count: int) -> int:
    
    if int(target_count) < 0:
        for layer in model.layers:
            layer.trainable = True
        return sum(1 for layer in model.layers if layer.weights)

    for layer in model.layers:
        layer.trainable = False

    desired = max(1, int(target_count))
    changed = 0
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.BatchNormalization):
            continue
        if not layer.weights:
            continue
        layer.trainable = True
        changed += 1
        if changed >= desired:
            break

    return changed

# Main fine-tuning entrypoint

def main():
    
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # Allow overriding the run seed via RUN_SEED env var for multi-seed experiments
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

    # Setting precision directly for standard backbones
    if tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: mixed_float16 (2x memory savings)")
    else:
        keras.mixed_precision.set_global_policy("float32")

    tf.config.threading.set_intra_op_parallelism_threads(INTRA_OP_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)
    logs_dir = os.path.join(os.path.dirname(FINAL_MODEL_PATH), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    gc.collect()
    keras.backend.clear_session()

        print("SOTA Fine-Tuning Pipeline")
    
    # ── load checkpoint ───────────────────────────────────────────────────
    checkpoint_source = _resolve_model_path(CHECKPOINT_PATH, FINAL_MODEL_PATH)
    print(f"Loading checkpoint: {checkpoint_source}")
    strategy = get_training_strategy()
    with strategy.scope():
        model = load_model(checkpoint_source)

    # ── layer unfreezing ──────────────────────────────────────────────────
    if int(FINE_TUNE_UNFREEZE_LAYERS) < 0 or int(UNFREEZE_LAYERS) < 0:
        unfreeze_target = -1
    else:
        unfreeze_target = min(int(FINE_TUNE_UNFREEZE_LAYERS), int(UNFREEZE_LAYERS))
    trainable_count = _unfreeze_top_layers(model, unfreeze_target)
    if unfreeze_target < 0:
        print(f"Unfroze full model ({trainable_count} trainable layers).")
    else:
        print(f"Unfroze {trainable_count} trainable layers (BN kept frozen).")

    # ── data pipeline ─────────────────────────────────────────────────────
    batch_size = int(FINE_TUNE_BATCH_SIZE)
    autotune = tf.data.AUTOTUNE

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=False,
    )

    def _prep(ds, training=False):
        mapped = ds.map(
            lambda x, y: (preprocess_batch_for_model(x), y),
            num_parallel_calls=autotune,
        )
        if training and USE_MIXUP:
            mapped = mapped.map(
                lambda x, y: mixup_batch_tf(x, y, alpha=MIXUP_ALPHA),
                num_parallel_calls=autotune,
            )
        return mapped.prefetch(autotune)

    train_gen = _prep(train_ds, training=True)
    val_gen = _prep(val_ds, training=False)

    train_gen, steps_per_epoch = _prepare_subset(
        train_gen,
        fraction=float(FINE_TUNE_DATA_FRACTION),
        max_batches=int(FINE_TUNE_MAX_STEPS_PER_EPOCH),
        dataset_name="train",
    )
    val_gen, validation_steps = _prepare_subset(
        val_gen,
        fraction=1.0,
        max_batches=int(FINE_TUNE_VAL_MAX_STEPS),
        dataset_name="validation",
    )

    # ── class weighting and loss ──────────────────────────────────────────
    class_weight = compute_class_weights_from_directory(TRAIN_DIR, train_ds.class_names)
    if class_weight:
        print("Class-balanced weighting enabled for fine-tuning.")

    selected_loss, fit_class_weight = build_loss(class_weight)
    if USE_MIXUP and fit_class_weight:
        print("MixUp active: disabling class_weight to avoid label-mix conflicts.")
        fit_class_weight = None

    if USE_MIXUP:
        print(f"MixUp augmentation enabled (alpha={MIXUP_ALPHA}).")

    # ── learning rate schedule ────────────────────────────────────────────
    total_epochs = int(FINE_TUNE_EPOCHS)
    total_steps = max(1, int(steps_per_epoch) * total_epochs)
    from config import WARMUP_EPOCHS

    warmup_steps = max(0, int(steps_per_epoch) * min(int(WARMUP_EPOCHS), total_epochs))

    if LR_SCHEDULER.lower() == "cosine":
        lr_schedule = WarmupCosineSchedule(
            peak_lr=float(FINE_TUNE_LEARNING_RATE),
            min_lr=max(float(FINE_TUNE_LEARNING_RATE) * 0.1, 1e-7),
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    else:
        lr_schedule = float(FINE_TUNE_LEARNING_RATE)

    # ── compile model ─────────────────────────────────────────────────────
    with strategy.scope():
        import inspect

        optimizer_kwargs = {
            "learning_rate": lr_schedule,
            "weight_decay": float(WEIGHT_DECAY),
            "clipnorm": 1.0,
            "use_ema": USE_OPTIMIZER_EMA,
            "ema_momentum": EMA_MOMENTUM,
        }
        if "gradient_accumulation_steps" in inspect.signature(
            keras.optimizers.AdamW
        ).parameters:
            optimizer_kwargs["gradient_accumulation_steps"] = max(
                1, int(ACCUMULATION_STEPS)
            )

        if USE_OPTIMIZER_EMA:
            print(f"AdamW EMA enabled (momentum={EMA_MOMENTUM}).")
        model.compile(
            optimizer=keras.optimizers.AdamW(**optimizer_kwargs),
            loss=selected_loss,
            metrics=["accuracy"],
        )

    # ── baseline evaluation ───────────────────────────────────────────────
    initial_best_val_acc = None
    if os.path.exists(FINAL_MODEL_PATH):
        try:
            initial_best_val_acc = _evaluate_val_accuracy(
                FINAL_MODEL_PATH, val_gen, validation_steps, strategy
            )
            print(f"Baseline val_accuracy of saved model: {initial_best_val_acc:.6f}")
        except Exception as exc:
            print(f"Could not evaluate baseline model: {exc}")

    # ── callbacks ─────────────────────────────────────────────────────────
    checkpoint = BestModelSaver(
        FINAL_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        initial_best=initial_best_val_acc,
        verbose=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        min_delta=0.001,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )
    overfit_stopper = OverfittingStopper(min_gap=0.05, patience=2, verbose=1)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"fine_tune_{run_stamp}"
    ft_history_latest = os.path.join(logs_dir, "fine_tune_history.csv")
    ft_history_archive = os.path.join(logs_dir, f"fine_tune_history_{run_stamp}.csv")
    ft_interval_latest = os.path.join(logs_dir, "fine_tune_interval_history.csv")
    ft_interval_archive = os.path.join(
        logs_dir, f"fine_tune_interval_history_{run_stamp}.csv"
    )
    latest_runs_path = os.path.join(logs_dir, "latest_runs.json")

    latest_runs = {}
    if os.path.exists(latest_runs_path):
        try:
            with open(latest_runs_path, "r", encoding="utf-8") as in_file:
                latest_runs = json.load(in_file)
        except Exception:
            latest_runs = {}
    base_train_run_id = (latest_runs.get("train") or {}).get("run_id")

    csv_loggers = [CSVLogger(ft_history_latest, append=False)]
    if SAVE_LOG_ARCHIVE:
        csv_loggers.append(CSVLogger(ft_history_archive, append=False))

    interval_loggers = [
        IntervalMetricsLogger(
            ft_interval_latest, points_per_epoch=12,
            stage="fine_tuning", run_id=run_id,
        ),
    ]
    if SAVE_LOG_ARCHIVE:
        interval_loggers.append(
            IntervalMetricsLogger(
                ft_interval_archive, points_per_epoch=12,
                stage="fine_tuning", run_id=run_id,
            )
        )

    print(f"Fine-tune history log: {ft_history_latest}")
    print(f"Fine-tune interval log: {ft_interval_latest}")
    if SAVE_LOG_ARCHIVE:
        print(f"Archive history log: {ft_history_archive}")
        print(f"Archive interval log: {ft_interval_archive}")

    tensorboard = None
    if tensorboard_available():
        tensorboard = TensorBoard(
            log_dir=os.path.join(logs_dir, "fine_tune_tensorboard"),
            histogram_freq=1, update_freq="epoch", write_graph=False,
        )
    else:
        print("TensorBoard not installed; skipping TensorBoard callback.")

    print(f"\nBatch size: {batch_size}  |  Epochs: {total_epochs}")
    print(f"Steps/epoch: {steps_per_epoch}  |  Validation steps: {validation_steps}")
    print("\n--- Fine-tuning ---\n")

    progress = ProgressEmitter(
        stage="fine_tuning",
        total_epochs=total_epochs,
        completed_epochs_before=0,
        run_start_time=time.time(),
    )

    # ── training loop ─────────────────────────────────────────────────────
    model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=total_epochs,
        callbacks=[
            cb
            for cb in [
                checkpoint, early_stopping, overfit_stopper,
                *csv_loggers, *interval_loggers,
                tensorboard, progress,
            ]
            if cb is not None
        ],
        class_weight=fit_class_weight,
        verbose=1,
    )

    # ── finalise ──────────────────────────────────────────────────────────
    gc.collect()
    best_model_source = _resolve_model_path(FINAL_MODEL_PATH)
    model = load_model(best_model_source)
    model.save(FINAL_MODEL_PATH)

    fine_tune_manifest = {
        "run_id": run_id,
        "run_stamp": run_stamp,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_train_run_id": base_train_run_id,
        "fine_tune_history_latest": ft_history_latest,
        "fine_tune_history_archive": ft_history_archive if SAVE_LOG_ARCHIVE else None,
        "fine_tune_interval_latest": ft_interval_latest,
        "fine_tune_interval_archive": ft_interval_archive if SAVE_LOG_ARCHIVE else None,
        "use_mixup": USE_MIXUP,
        "steps_per_epoch": int(steps_per_epoch),
        "validation_steps": int(validation_steps),
        "fine_tune_epochs": int(total_epochs),
        "fine_tune_data_fraction": float(FINE_TUNE_DATA_FRACTION),
        "fine_tune_unfreeze_layers": int(unfreeze_target),
    }
    latest_runs["fine_tune"] = fine_tune_manifest

    if SAVE_RUN_MANIFESTS or SAVE_LOG_ARCHIVE:
        with open(
            os.path.join(logs_dir, f"fine_tune_run_manifest_{run_stamp}.json"),
            "w", encoding="utf-8",
        ) as out_file:
            json.dump(fine_tune_manifest, out_file, indent=2)
    with open(latest_runs_path, "w", encoding="utf-8") as out_file:
        json.dump(latest_runs, out_file, indent=2)

        print("Fine-tuning complete")
    print(f"Best model saved to: {FINAL_MODEL_PATH}")
    

if __name__ == "__main__":
    main()
