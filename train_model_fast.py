"""
Fast Training Script for Plant Leaf Disease Detection
Optimized to complete training in under 2 hours with high accuracy

Key optimizations:
- MobileNetV2: Lightweight efficient architecture
- Reduced image size: 160x160 (faster processing)
- Larger batch size: 64 (better GPU utilization)
- Mixed precision: Faster computation
- Fewer epochs: 30 total (15 frozen + 15 fine-tune)
- Early stopping: Stops when converged
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time

# Enable mixed precision for faster training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"✓ Mixed precision enabled: {policy.name}")

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# OPTIMIZED Configuration
IMG_SIZE = 160
BATCH_SIZE = 64
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 15
NUM_CLASSES = 46
LEARNING_RATE = 0.001

# Paths
BASE_DIR = '/workspaces/Leaf_Disease_Detection/dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 80)
print("FAST TRAINING - Plant Leaf Disease Detection")
print("=" * 80)
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: Phase1={EPOCHS_PHASE1}, Phase2={EPOCHS_PHASE2}")
print(f"Target: < 2 hours")
print("=" * 80)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✓ GPU Available: {len(gpus)} device(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("\n⚠ No GPU - training will be slower")

start_time = time.time()

# Data generators
print("\n[1/6] Setting up data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"✓ Train: {train_generator.samples} | Val: {val_generator.samples} | Test: {test_generator.samples}")

with open('models/class_indices_fast.json', 'w') as f:
    json.dump(train_generator.class_indices, f, indent=2)

# Build model
print("\n[2/6] Building model with MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print(f"✓ Model: {model.count_params():,} parameters")

# Callbacks
print("\n[3/6] Setting up callbacks...")

checkpoint = ModelCheckpoint(
    'models/best_model_fast.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stop, reduce_lr]

# Phase 1: Frozen base
print("\n[4/6] Phase 1: Training with frozen base...")
print("=" * 80)

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks,
    verbose=1
)

phase1_time = time.time() - start_time
print(f"\n✓ Phase 1: {phase1_time/60:.1f} minutes")

# Phase 2: Fine-tuning
print("\n[5/6] Phase 2: Fine-tuning...")
print("=" * 80)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
    initial_epoch=len(history_phase1.history['loss']),
    callbacks=callbacks,
    verbose=1
)

total_time = time.time() - start_time

print(f"\n{'='*80}")
print(f"TOTAL TIME: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"{'='*80}")

# Evaluate
print("\n[6/6] Evaluating on test set...")

test_loss, test_acc, test_top3 = model.evaluate(test_generator, verbose=1)

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Top-3 Accuracy: {test_top3*100:.2f}%")
print(f"Training Time: {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")
print("=" * 80)

model.save('models/final_model_fast.h5')
print("\n✓ Saved: models/final_model_fast.h5")

# Plot
combined_history = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
}

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(combined_history['accuracy'], label='Train', linewidth=2)
plt.plot(combined_history['val_accuracy'], label='Validation', linewidth=2)
plt.axvline(x=EPOCHS_PHASE1, color='red', linestyle='--', label='Fine-tuning')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Fast Training - Accuracy', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(combined_history['loss'], label='Train', linewidth=2)
plt.plot(combined_history['val_loss'], label='Validation', linewidth=2)
plt.axvline(x=EPOCHS_PHASE1, color='red', linestyle='--', label='Fine-tuning')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Fast Training - Loss', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/training_fast.png', dpi=300)
print("✓ Saved: plots/training_fast.png")

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE!")
print("=" * 80)
print(f"Status: {'✓ SUCCESS' if total_time < 7200 else '⚠ OVER TARGET'}")
print("=" * 80)
