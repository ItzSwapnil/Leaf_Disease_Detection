"""
Alternative Training Script using MobileNetV2
Lighter and faster than EfficientNet, good for resource-constrained environments
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = 224  # MobileNetV2 standard size
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 46
LEARNING_RATE = 0.001

# Dataset paths
BASE_DIR = '/workspaces/Leaf_Disease_Detection/dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# Create output directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("="*80)
print("Plant Leaf Disease Detection with MobileNetV2")
print("="*80)
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Number of Classes: {NUM_CLASSES}")
print(f"Training Epochs: {EPOCHS}")
print("="*80)

# Data Augmentation and Preprocessing
print("\n[1/5] Setting up data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
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

print(f"✓ Training samples: {train_generator.samples}")
print(f"✓ Validation samples: {val_generator.samples}")
print(f"✓ Test samples: {test_generator.samples}")

# Save class indices
with open('models/class_indices_mobilenet.json', 'w') as f:
    json.dump(train_generator.class_indices, f, indent=2)

# Build the Model
print("\n[2/5] Building the CNN model with MobileNetV2...")

# Load pre-trained MobileNetV2 without top classification layer
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(units=256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(units=NUM_CLASSES, activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print("✓ Model built successfully!")
print(f"✓ Total parameters: {model.count_params():,}")

# Display model summary
model.summary()
print("\nCNN model built and compiled successfully.")

# Setup Callbacks
print("\n[3/5] Setting up training callbacks...")

checkpoint = ModelCheckpoint(
    'models/best_model_mobilenet.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-7,
    verbose=1
)

tensorboard = TensorBoard(
    log_dir=f'logs/mobilenet_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
    histogram_freq=1
)

callbacks = [checkpoint, early_stop, reduce_lr, tensorboard]

print("✓ Callbacks configured")

# Train the Model - Phase 1 (Frozen base)
print("\n[4/5] Training Phase 1: Training with frozen base model...")
print("="*80)

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Phase 1 training completed!")

# Fine-tuning - Phase 2
print("\n[5/5] Training Phase 2: Fine-tuning the model...")
print("="*80)

# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Freeze the first 100 layers
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print(f"✓ Unfroze {len([l for l in base_model.layers if l.trainable])} layers for fine-tuning")

# Update checkpoint for phase 2
checkpoint_phase2 = ModelCheckpoint(
    'models/best_model_mobilenet_finetuned.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

callbacks_phase2 = [checkpoint_phase2, early_stop, reduce_lr, tensorboard]

history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    initial_epoch=len(history_phase1.history['loss']),
    callbacks=callbacks_phase2,
    verbose=1
)

print("\n✓ Phase 2 training completed!")

# Combine histories
history = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
}

# Evaluate on Test Set
print("\nEvaluating model on test set...")
print("="*80)

test_loss, test_accuracy, test_top3_accuracy = model.evaluate(test_generator, verbose=1)

print(f"\n{'='*80}")
print(f"FINAL RESULTS (MobileNetV2)")
print(f"{'='*80}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Top-3 Accuracy: {test_top3_accuracy*100:.2f}%")
print(f"{'='*80}")

# Save the final model
model.save('models/final_model_mobilenet.h5')
print("\n✓ Model saved to 'models/final_model_mobilenet.h5'")

# Plot training history
print("\nGenerating training plots...")

plt.figure(figsize=(15, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.axvline(x=len(history_phase1.history['loss']), color='r', linestyle='--', label='Fine-tuning Start')
plt.title('Model Accuracy (MobileNetV2)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.axvline(x=len(history_phase1.history['loss']), color='r', linestyle='--', label='Fine-tuning Start')
plt.title('Model Loss (MobileNetV2)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('plots/training_history_mobilenet.png', dpi=300, bbox_inches='tight')
print("✓ Training plots saved to 'plots/training_history_mobilenet.png'")

print("\n" + "="*80)
print("Training completed successfully with MobileNetV2! 🎉")
print("="*80)
print(f"Best model saved at: models/best_model_mobilenet_finetuned.h5")
print(f"Final model saved at: models/final_model_mobilenet.h5")
print(f"Class indices saved at: models/class_indices_mobilenet.json")
print("="*80)
