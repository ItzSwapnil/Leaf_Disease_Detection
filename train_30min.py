#!/usr/bin/env python
"""
30-Minute Training Script - Uses ALL data for maximum accuracy
Target: 85-92% accuracy in ~30 minutes
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

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(42)
tf.random.set_seed(42)

# === CONFIGURATION ===
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS_PHASE1 = 5      # Frozen base
EPOCHS_PHASE2 = 10     # Fine-tuning
NUM_CLASSES = 46
LR_PHASE1 = 0.001
LR_PHASE2 = 0.0001     # Lower LR for fine-tuning

BASE_DIR = '/workspaces/Leaf_Disease_Detection/dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 60)
print("30-MINUTE TRAINING - Full Dataset + Fine-tuning")
print("=" * 60)
print(f"Image: {IMG_SIZE}x{IMG_SIZE} | Batch: {BATCH_SIZE}")
print(f"Phase 1: {EPOCHS_PHASE1} epochs (frozen) | Phase 2: {EPOCHS_PHASE2} epochs (fine-tune)")
print("=" * 60)

start_time = time.time()

# === DATA LOADING ===
print("\n[1/5] Loading ALL data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
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

# Use MORE data per epoch (2000 steps = ~64000 images per epoch)
steps_per_epoch = 2000
validation_steps = 200
print(f"✓ Using {steps_per_epoch} steps/epoch ({steps_per_epoch * BATCH_SIZE:,} images/epoch)")

# Save class indices
with open('models/class_indices_30min.json', 'w') as f:
    json.dump(train_gen.class_indices, f, indent=2)

# === MODEL ===
print("\n[2/5] Building MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    alpha=0.5
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(
    optimizer=Adam(learning_rate=LR_PHASE1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✓ Total params: {model.count_params():,}")

# === PHASE 1: FROZEN TRAINING ===
print("\n[3/5] Phase 1: Training with frozen base...")
print("=" * 60)

callbacks_p1 = [
    ModelCheckpoint('models/best_model_30min.h5', monitor='val_accuracy', 
                    save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

history1 = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks_p1,
    verbose=1
)

phase1_time = time.time() - start_time
print(f"\n✓ Phase 1 completed in {phase1_time/60:.1f} minutes")

# === PHASE 2: FINE-TUNING ===
print("\n[4/5] Phase 2: Fine-tuning top layers...")
print("=" * 60)

# Unfreeze top 30 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LR_PHASE2),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"✓ Trainable params after unfreezing: {trainable:,}")

callbacks_p2 = [
    ModelCheckpoint('models/best_model_30min.h5', monitor='val_accuracy', 
                    save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
]

history2 = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks_p2,
    verbose=1
)

total_time = time.time() - start_time
print(f"\n✓ Total training time: {total_time/60:.1f} minutes")

# === EVALUATION ===
print("\n[5/5] Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_gen, verbose=1)

print("\n" + "=" * 60)
print(f"🎯 TEST ACCURACY: {test_acc*100:.2f}%")
print(f"⏱️  TOTAL TIME: {total_time/60:.1f} minutes")
print("=" * 60)

# Save final model
model.save('models/final_model_30min.h5')
print("✓ Model saved to models/final_model_30min.h5")

# === PLOT ===
# Combine histories
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, 'b-o', label='Train')
plt.plot(val_acc, 'r-s', label='Validation')
plt.axvline(x=EPOCHS_PHASE1-1, color='g', linestyle='--', label='Fine-tune start')
plt.title(f'Accuracy (Final: {test_acc*100:.1f}%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(loss, 'b-o', label='Train')
plt.plot(val_loss, 'r-s', label='Validation')
plt.axvline(x=EPOCHS_PHASE1-1, color='g', linestyle='--', label='Fine-tune start')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/training_30min.png', dpi=150)
print("✓ Plot saved to plots/training_30min.png")

print("\n" + "=" * 60)
print("DONE! To make predictions:")
print("  python predict_cpu.py path/to/leaf_image.jpg")
print("=" * 60)
