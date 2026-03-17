import os
import json
import random
import time
import importlib.util
import keras
import tensorflow as tf
from hardware import configure_tensorflow, get_training_strategy
from keras.applications import EfficientNetV2B0
from keras.applications.efficientnet_v2 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from training_progress import ProgressEmitter
from config import (IMG_SIZE, BATCH_SIZE, STEPS_PER_EPOCH, VALIDATION_STEPS,
                    EPOCHS_PHASE1, EPOCHS_PHASE2, TRAIN_DIR, VAL_DIR,
                    CHECKPOINT_PATH, INTRA_OP_THREADS, INTER_OP_THREADS,
                    DENSE_UNITS, DROPOUT_RATE, NUM_CLASSES, LABEL_SMOOTHING,
                    LEARNING_RATE_PHASE1, LEARNING_RATE_PHASE2,
                    EARLY_STOPPING_PATIENCE, UNFREEZE_LAYERS,
                    CLASS_INDICES_PATH)


def _tensorboard_available():
    return importlib.util.find_spec('tensorboard') is not None

def main():
    # Stable seeds make the training procedure reproducible across reruns.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['PYTHONHASHSEED'] = '42'
    random.seed(42)
    keras.utils.set_random_seed(42)
    configure_tensorflow()

    if tf.config.list_physical_devices('GPU'):
        keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled: mixed_float16")
    else:
        keras.mixed_precision.set_global_policy('float32')

    tf.config.threading.set_intra_op_parallelism_threads(INTRA_OP_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    logs_dir = os.path.join(os.path.dirname(CHECKPOINT_PATH), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator  # type: ignore[attr-defined]

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=22,
        horizontal_flip=True,
        width_shift_range=0.12,
        height_shift_range=0.12,
        zoom_range=0.2,
        brightness_range=(0.8, 1.2),
        shear_range=0.1,
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
    )

    strategy = get_training_strategy()
    with strategy.scope():
        base_model = EfficientNetV2B0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet',
        )
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(DENSE_UNITS, activation='relu')(x)
        x = Dropout(DROPOUT_RATE)(x)
        outputs = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
        model = Model(inputs=base_model.input, outputs=outputs)

    steps_per_epoch = min(STEPS_PER_EPOCH, max(1, train_gen.samples // BATCH_SIZE))
    validation_steps = min(VALIDATION_STEPS, max(1, val_gen.samples // BATCH_SIZE))

    warmup_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE_PHASE1,
        first_decay_steps=max(200, steps_per_epoch * 2),
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.08,
    )

    with strategy.scope():
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=warmup_schedule,
                weight_decay=1e-4,
                clipnorm=1.0,
            ),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')],
        )

    checkpoint = ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
    )
    print("Using cosine LR schedule; ReduceLROnPlateau callback disabled.")
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=EARLY_STOPPING_PATIENCE,
        mode='max',
        restore_best_weights=True,
        verbose=1,
    )
    csv_logger = CSVLogger(os.path.join(logs_dir, 'train_history.csv'), append=True)
    tensorboard = None
    if _tensorboard_available():
        tensorboard = TensorBoard(
            log_dir=os.path.join(logs_dir, 'tensorboard'),
            histogram_freq=1,
            update_freq='epoch',
            write_graph=False,
        )
    else:
        print("TensorBoard package not found; continuing without TensorBoard callback.")

    with open(CLASS_INDICES_PATH, 'w', encoding='utf-8') as class_file:
        json.dump(train_gen.class_indices, class_file, indent=2)

    total_epochs = EPOCHS_PHASE1 + EPOCHS_PHASE2
    run_start_time = time.time()
    progress_phase1 = ProgressEmitter(
        stage='phase1_warmup',
        total_epochs=total_epochs,
        completed_epochs_before=0,
        run_start_time=run_start_time,
    )

    print(f"\n--- Phase 1: warm-up on {train_gen.samples} training images ---")
    phase1_callbacks: list[keras.callbacks.Callback] = [
        checkpoint,
        early_stopping,
        csv_logger,
        progress_phase1,
    ]
    if tensorboard is not None:
        phase1_callbacks.append(tensorboard)

    phase1_history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=EPOCHS_PHASE1,
        callbacks=phase1_callbacks,
        verbose=2,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False

    fine_tune_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE_PHASE2,
        first_decay_steps=max(160, steps_per_epoch * 2),
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.12,
    )

    with strategy.scope():
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=fine_tune_schedule,
                weight_decay=5e-5,
                clipnorm=1.0,
            ),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')],
        )

    completed_phase1_epochs = len(phase1_history.history.get('loss', []))
    total_epochs_phase2_view = max(1, completed_phase1_epochs + EPOCHS_PHASE2)

    progress_phase2 = ProgressEmitter(
        stage='phase2_finetune',
        total_epochs=total_epochs_phase2_view,
        completed_epochs_before=completed_phase1_epochs,
        run_start_time=run_start_time,
    )

    print("\n--- Phase 2: fine-tuning unfrozen top layers ---")
    phase2_callbacks: list[keras.callbacks.Callback] = [
        checkpoint,
        early_stopping,
        csv_logger,
        progress_phase2,
    ]
    if tensorboard is not None:
        phase2_callbacks.append(tensorboard)

    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=EPOCHS_PHASE2,
        callbacks=phase2_callbacks,
        verbose=2,
    )


if __name__ == "__main__":
    main()