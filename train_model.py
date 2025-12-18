"""
Highly Accurate Plant Leaf Disease Detection System
Uses Transfer Learning with EfficientNet and advanced training techniques
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = 300  # EfficientNet works well with 300x300
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

print("=" * 80)
print("Plant Leaf Disease Detection System")
print("=" * 80)
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Number of Classes: {NUM_CLASSES}")
print(f"Training Epochs: {EPOCHS}")
print("=" * 80)

# Data Augmentation and Preprocessing
print("\n[1/6] Setting up data generators...")

# Training data generator with heavy augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

# Validation and test data generator (only rescaling)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
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
print(f"✓ Classes found: {len(train_generator.class_indices)}")

# Save class indices for later use
import json
with open('models/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f, indent=2)

# Build the Model
print("\n[2/6] Building the CNN model with Transfer Learning...")

def build_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    """
    Build a CNN model using EfficientNetB3 as base
    EfficientNet provides better accuracy than MobileNet with reasonable efficiency
    """
    
    # Load pre-trained EfficientNetB3 without top layers
    base_model = EfficientNetB3(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)
    
    return model, base_model

model, base_model = build_model()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print("✓ Model built successfully!")
print(f"✓ Total parameters: {model.count_params():,}")
print(f"✓ Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# Setup Callbacks
print("\n[3/6] Setting up training callbacks...")

checkpoint = ModelCheckpoint(
    'models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
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

tensorboard = TensorBoard(
    log_dir=f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
    histogram_freq=1
)

callbacks = [checkpoint, early_stop, reduce_lr, tensorboard]

print("✓ Callbacks configured: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard")

# Train the Model - Phase 1 (Frozen base)
print("\n[4/6] Training Phase 1: Training with frozen base model...")
print("=" * 80)

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Phase 1 training completed!")

# Fine-tuning - Phase 2 (Unfreeze some layers)
print("\n[5/6] Training Phase 2: Fine-tuning the model...")
print("=" * 80)

# Unfreeze the top layers of the base model for fine-tuning
base_model.trainable = True

# Freeze the first 80% of layers, fine-tune the rest
fine_tune_at = int(len(base_model.layers) * 0.8)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with a lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print(f"✓ Unfroze {len(base_model.layers) - fine_tune_at} layers for fine-tuning")
print(f"✓ New trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# Update checkpoint path for phase 2
checkpoint_phase2 = ModelCheckpoint(
    'models/best_model_finetuned.h5',
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
print("\n[6/6] Evaluating model on test set...")
print("=" * 80)

test_loss, test_accuracy, test_top3_accuracy = model.evaluate(test_generator, verbose=1)

print(f"\n{'='*80}")
print(f"FINAL RESULTS")
print(f"{'='*80}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Top-3 Accuracy: {test_top3_accuracy*100:.2f}%")
print(f"{'='*80}")

# Save the final model
model.save('models/final_model.h5')
print("\n✓ Model saved to 'models/final_model.h5'")

# Plot training history
print("\n[7/7] Generating training plots...")

plt.figure(figsize=(15, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.axvline(x=len(history_phase1.history['loss']), color='r', linestyle='--', label='Fine-tuning Start')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.axvline(x=len(history_phase1.history['loss']), color='r', linestyle='--', label='Fine-tuning Start')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
print("✓ Training plots saved to 'plots/training_history.png'")

print("\n" + "="*80)
print("Training completed successfully! 🎉")
print("="*80)
print(f"Best model saved at: models/best_model_finetuned.h5")
print(f"Final model saved at: models/final_model.h5")
print(f"Class indices saved at: models/class_indices.json")
print(f"Training plots saved at: plots/training_history.png")
print(f"TensorBoard logs saved at: logs/")
print("="*80)
