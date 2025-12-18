#!/usr/bin/env python
"""
Quick Training Script - Optimized for CPU, ~30-45 minutes
Uses MobileNetV2 with proper image size for better accuracy
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
import matplotlib.pyplot as plt
import json
import time

# Disable GPU, optimize for CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(42)
tf.random.set_seed(42)

# === CONFIGURATION ===
IMG_SIZE = 128          # MobileNetV2 works well with 128+
BATCH_SIZE = 32         # Larger batches = faster
EPOCHS = 10             # With early stopping
NUM_CLASSES = 46
LEARNING_RATE = 0.001

BASE_DIR = '/workspaces/Leaf_Disease_Detection/dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 60)
print("QUICK TRAINING - MobileNetV2 (CPU Optimized)")
print("=" * 60)
print(f"Image: {IMG_SIZE}x{IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
print("=" * 60)

start_time = time.time()

# === DATA LOADING ===
print("\n[1/4] Loading data...")

# Minimal augmentation for speed
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"✓ Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")
print(f"✓ Classes: {len(train_gen.class_indices)}")

# Save class indices
with open('models/class_indices_quick.json', 'w') as f:
    json.dump(train_gen.class_indices, f, indent=2)

# === MODEL ===
print("\n[2/4] Building MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    alpha=0.5  # Smaller model, faster training
)
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"✓ Trainable parameters: {trainable:,}")

# === CALLBACKS ===
print("\n[3/4] Training...")

callbacks = [
    ModelCheckpoint(
        'models/best_model_quick.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# === TRAINING ===
print("=" * 60)

# Use steps_per_epoch to speed up epochs (sample subset)
steps_per_epoch = min(500, train_gen.samples // BATCH_SIZE)
validation_steps = min(100, val_gen.samples // BATCH_SIZE)

print(f"Steps per epoch: {steps_per_epoch} (of {train_gen.samples // BATCH_SIZE})")

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

train_time = time.time() - start_time
print(f"\n✓ Training completed in {train_time/60:.1f} minutes")

# === EVALUATION ===
print("\n[4/4] Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_gen, verbose=1)

print("\n" + "=" * 60)
print(f"TEST ACCURACY: {test_acc*100:.2f}%")
print(f"TRAINING TIME: {train_time/60:.1f} minutes")
print("=" * 60)

# Save final model
model.save('models/final_model_quick.h5')
print("✓ Model saved to models/final_model_quick.h5")

# === PLOT ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'b-o', label='Train')
plt.plot(history.history['val_accuracy'], 'r-s', label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'b-o', label='Train')
plt.plot(history.history['val_loss'], 'r-s', label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/quick_training.png', dpi=150)
print("✓ Plot saved to plots/quick_training.png")

print("\n" + "=" * 60)
print("DONE! To make predictions:")
print("  python predict_cpu.py path/to/image.jpg")
print("=" * 60)
