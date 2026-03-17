import keras
import tensorflow as tf
from hardware import configure_tensorflow, get_training_strategy
from keras.models import load_model
from keras.applications.efficientnet_v2 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from training_progress import ProgressEmitter
from config import (
    CHECKPOINT_PATH,
    FINAL_MODEL_PATH,
    TRAIN_DIR,
    VAL_DIR,
    IMG_SIZE,
    LABEL_SMOOTHING,
    UNFREEZE_LAYERS,
)
import os
import gc
import time
import importlib.util


def _resolve_model_path(*candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"No model file found in candidates: {candidates}")


def _tensorboard_available():
    return importlib.util.find_spec('tensorboard') is not None

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    configure_tensorflow()

    if tf.config.list_physical_devices('GPU'):
        keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled: mixed_float16")
    else:
        keras.mixed_precision.set_global_policy('float32')

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    logs_dir = os.path.join(os.path.dirname(FINAL_MODEL_PATH), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    gc.collect()
    keras.backend.clear_session()

    print("=" * 60)
    print("Memory-safe fine-tuning")
    print("=" * 60)

    checkpoint_source = _resolve_model_path(CHECKPOINT_PATH)
    print(f"\nLoading base checkpoint: {checkpoint_source}")
    strategy = get_training_strategy()
    with strategy.scope():
        model = load_model(checkpoint_source)

    print(f"Unfreezing top {UNFREEZE_LAYERS} layers...")
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    trainable_count = sum(1 for layer_obj in model.layers if layer_obj.trainable)
    print(f"   Trainable layers: {trainable_count}")

    batch_size = 4
    autotune = tf.data.AUTOTUNE

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=False,
    )

    def _prep(ds):
        return ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=autotune).prefetch(autotune)

    train_gen = _prep(train_ds)
    val_gen = _prep(val_ds)

    steps_per_epoch = max(1, tf.data.experimental.cardinality(train_ds).numpy())
    cosine_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=5e-5,
        first_decay_steps=max(120, steps_per_epoch * 2),
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1,
    )

    with strategy.scope():
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=cosine_schedule,
                weight_decay=5e-5,
                clipnorm=1.0,
            ),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')],
        )

    checkpoint = ModelCheckpoint(
        FINAL_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
    )
    print("Using cosine LR schedule; ReduceLROnPlateau callback disabled.")
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        mode='max',
        restore_best_weights=True,
        verbose=1,
    )
    csv_logger = CSVLogger(os.path.join(logs_dir, 'fine_tune_history.csv'), append=True)
    tensorboard = None
    if _tensorboard_available():
        tensorboard = TensorBoard(
            log_dir=os.path.join(logs_dir, 'fine_tune_tensorboard'),
            histogram_freq=1,
            update_freq='epoch',
            write_graph=False,
        )
    else:
        print("TensorBoard package not found; continuing without TensorBoard callback.")

    print("\nTraining configuration:")
    print(f"   Batch size: {batch_size}")
    print("   Optimizer: AdamW + CosineDecayRestarts")
    print("\n--- fine-tuning ---\n")

    total_epochs = 20
    progress = ProgressEmitter(
        stage='fine_tuning',
        total_epochs=total_epochs,
        completed_epochs_before=0,
        run_start_time=time.time(),
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=total_epochs,
        callbacks=[cb for cb in [checkpoint, early_stopping, csv_logger, tensorboard, progress] if cb is not None],
        verbose=2,
    )

    gc.collect()
    best_model_source = _resolve_model_path(FINAL_MODEL_PATH)
    model = load_model(best_model_source)
    model.save(FINAL_MODEL_PATH)

    print("\n" + "=" * 60)
    print("Training complete")
    print("Best model saved to:", FINAL_MODEL_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()


