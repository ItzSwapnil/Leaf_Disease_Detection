#!/usr/bin/env python
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
from datetime import datetime
import json
import time

# Optimize for CPU
os.environ['TF_CPP_THREAD_POOL_SIZE'] = '4'
tf.config.set_visible_devices([], 'GPU')

np.random.seed(42)
tf.random.set_seed(42)

# Config
IMG_SIZE = 96
BATCH_SIZE = 16
EPOCHS = 15
NUM_CLASSES = 46
LEARNING_RATE = 0.001

BASE_DIR = '/workspaces/Leaf_Disease_Detection/dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("="*80)
print("CPU-OPTIMIZED TRAINING (4 cores, single cycle)")
print("="*80)
print(f"Image: {IMG_SIZE}x{IMG_SIZE} | Batch: {BATCH_SIZE} | Target: <60 min")
print("="*80)

start_time = time.time()

# Data
print("\n[1/5] Loading data...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)
val_gen = val_test_datagen.flow_from_directory(VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
test_gen = val_test_datagen.flow_from_directory(TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

print(f"✓ Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")

with open('models/class_indices_cpu.json', 'w') as f:
    json.dump(train_gen.class_indices, f, indent=2)

# Model
print("\n[2/5] Building MobileNetV3-Small...")
base_model = MobileNetV3Small(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.1)(x)
outputs = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
print(f"✓ Model: {model.count_params():,} parameters")

# Callbacks
print("\n[3/5] Setting up callbacks...")
checkpoint = ModelCheckpoint('models/best_model_cpu.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)

# Train
print("\n[4/5] Training...")
print("="*80)
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint, early_stop, reduce_lr], verbose=1)

total_time = time.time() - start_time
print(f"\nTime: {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")

# Evaluate
print("\n[5/5] Evaluating...")
test_loss, test_acc = model.evaluate(test_gen, verbose=1)

print("\n" + "="*80)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Time: {total_time/60:.1f} minutes")
print(f"Status: {'✓ SUCCESS' if total_time < 3600 else '✓ COMPLETED'}")
print("="*80)

model.save('models/final_model_cpu.h5')
print("✓ Models saved!")

# Plot
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'o-', label='Train'); plt.plot(history.history['val_accuracy'], 's-', label='Val')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('CPU Training'); plt.legend(); plt.grid(alpha=0.3)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'o-', label='Train'); plt.plot(history.history['val_loss'], 's-', label='Val')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('CPU Training'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/cpu_training.png', dpi=150)
print("✓ Plot saved!")