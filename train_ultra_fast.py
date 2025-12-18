#!/usr/bin/env python
"""
ULTRA-FAST Training - Completes in under 30 minutes!
Uses 10% data sampling for speed while maintaining reasonable accuracy.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
import time

# CPU optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

np.random.seed(42)
tf.random.set_seed(42)

# ULTRA-FAST Config
IMG_SIZE = 64          # Smaller images = faster
BATCH_SIZE = 32        # Larger batches = fewer steps
EPOCHS = 10            # Fewer epochs
SAMPLE_FRACTION = 0.1  # Use only 10% of data
NUM_CLASSES = 46

BASE_DIR = '/workspaces/Leaf_Disease_Detection/dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 60)
print("⚡ ULTRA-FAST TRAINING ⚡")
print("=" * 60)
print(f"Image: {IMG_SIZE}x{IMG_SIZE} | Batch: {BATCH_SIZE}")
print(f"Data: {int(SAMPLE_FRACTION*100)}% sampling | Target: <30 min")
print("=" * 60)

start_time = time.time()

# Data generators with sampling
print("\n[1/5] Loading sampled data...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=1.0 - SAMPLE_FRACTION  # Use this to sample
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load full generators first
full_train = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training'  # This gets SAMPLE_FRACTION of data
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Calculate steps (limit for speed)
steps_per_epoch = min(full_train.samples // BATCH_SIZE, 500)  # Max 500 steps
validation_steps = min(val_gen.samples // BATCH_SIZE, 100)    # Max 100 steps

print(f"✓ Train: {full_train.samples} samples, {steps_per_epoch} steps/epoch")
print(f"✓ Val: {val_gen.samples} samples | Test: {test_gen.samples} samples")

# Save class indices
with open('models/class_indices_fast.json', 'w') as f:
    json.dump(full_train.class_indices, f, indent=2)

# Model
print("\n[2/5] Building lightweight model...")
base_model = MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(f"✓ Parameters: {model.count_params():,}")

# Callbacks
print("\n[3/5] Setting up callbacks...")
checkpoint = ModelCheckpoint(
    'models/best_model_fast.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# Train
print("\n[4/5] Training...")
print("=" * 60)
history = model.fit(
    full_train,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

train_time = time.time() - start_time
print(f"\n⏱ Training time: {train_time/60:.1f} minutes")

# Save final model
model.save('models/final_model_fast.h5')

# Evaluate
print("\n[5/5] Evaluating...")
test_steps = min(test_gen.samples // BATCH_SIZE, 200)
test_loss, test_acc = model.evaluate(test_gen, steps=test_steps, verbose=1)

total_time = time.time() - start_time

# Results
print("\n" + "=" * 60)
print("📊 RESULTS")
print("=" * 60)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Total Time: {total_time/60:.1f} minutes")
print(f"Status: {'✅ UNDER 1 HOUR!' if total_time < 3600 else '✓ Complete'}")
print("=" * 60)

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'b-o', label='Train')
plt.plot(history.history['val_accuracy'], 'r-s', label='Val')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'b-o', label='Train')
plt.plot(history.history['val_loss'], 'r-s', label='Val')
plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/ultra_fast_training.png', dpi=150)
print("\n✓ Plot saved to plots/ultra_fast_training.png")
print("✓ Model saved to models/best_model_fast.h5")
