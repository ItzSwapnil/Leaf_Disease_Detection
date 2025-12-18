"""
Fast Training Script for Leaf Disease Detection
Optimized to complete all training cycles in under 2 hours while maintaining high accuracy

Key Optimizations:
- MobileNetV3Small architecture (lighter and faster)
- Mixed precision training for GPU acceleration
- Optimized image size (224x224)
- Reduced but effective epochs
- Streamlined augmentation
- Single-phase training with gradual unfreezing
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import set_global_policy
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time

# Enable mixed precision training for speed (requires GPU with compute capability >= 7.0)
try:
    set_global_policy('mixed_float16')
    print("✓ Mixed precision training enabled (GPU acceleration)")
except (ValueError, RuntimeError) as e:
    print(f"⚠ Mixed precision not available: {e}, using default precision")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Optimized Configuration for Fast Training
IMG_SIZE = 224  # Smaller image size for faster processing
BATCH_SIZE = 64  # Larger batch size for better GPU utilization
EPOCHS = 30  # Reduced epochs with early stopping
NUM_CLASSES = 46
LEARNING_RATE = 0.002  # Slightly higher initial learning rate
FINE_TUNE_RATIO = 0.7  # Freeze first 70% of base layers during fine-tuning

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
print("🚀 FAST LEAF DISEASE DETECTION TRAINING")
print("=" * 80)
print(f"Model: MobileNetV3Small (Optimized for Speed)")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Number of Classes: {NUM_CLASSES}")
print(f"Maximum Epochs: {EPOCHS}")
print(f"Target Training Time: < 2 hours")
print("=" * 80)

# Start timing
start_time = time.time()

# Optimized Data Augmentation (less aggressive for faster processing)
print("\n[1/5] Setting up optimized data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Reduced from 40
    width_shift_range=0.15,  # Reduced from 0.2
    height_shift_range=0.15,
    zoom_range=0.15,  # Reduced from 0.2
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators with optimized settings
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

# Save class indices
with open('models/class_indices_fast.json', 'w') as f:
    json.dump(train_generator.class_indices, f, indent=2)

# Build Fast Model
print("\n[2/5] Building fast CNN model...")

def build_fast_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    """
    Build an optimized CNN model using MobileNetV3Small
    MobileNetV3Small is significantly faster than MobileNetV2 and EfficientNet
    while maintaining good accuracy
    """
    
    # Load pre-trained MobileNetV3Small
    base_model = MobileNetV3Small(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
        minimalistic=False  # Use full architecture with SE blocks and hard swish
    )
    
    # Initially freeze base model
    base_model.trainable = False
    
    # Add lightweight classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)  # Smaller dense layer
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax', dtype='float32')(x)  # Ensure float32 output
    
    model = Model(inputs=base_model.input, outputs=output)
    
    return model, base_model

model, base_model = build_fast_model()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print("✓ Fast model built successfully!")
print(f"✓ Total parameters: {model.count_params():,}")
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"✓ Trainable parameters: {trainable_params:,}")
print(f"✓ Model size: ~{model.count_params() / 1e6:.1f}M parameters")

# Setup Callbacks with Aggressive Early Stopping
print("\n[3/5] Setting up training callbacks...")

checkpoint = ModelCheckpoint(
    'models/best_model_fast.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# More aggressive early stopping for faster training
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,  # Reduced from 8-10
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,  # Reduced from 3-4
    min_lr=1e-7,
    verbose=1
)

tensorboard = TensorBoard(
    log_dir=f'logs/fast_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
    histogram_freq=0  # Disable histogram for speed
)

callbacks = [checkpoint, early_stop, reduce_lr, tensorboard]

print("✓ Callbacks configured with aggressive early stopping")

# Train the Model - Phase 1 (Frozen base)
print("\n[4/5] Training Phase 1: Fast training with frozen base...")
print("=" * 80)

phase1_start = time.time()

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,  # Reduced from 20-25
    callbacks=callbacks,
    verbose=1
)

phase1_time = (time.time() - phase1_start) / 60
print(f"\n✓ Phase 1 completed in {phase1_time:.1f} minutes!")

# Fine-tuning - Phase 2 (Gradual unfreezing)
print("\n[5/5] Training Phase 2: Quick fine-tuning...")
print("=" * 80)

phase2_start = time.time()

# Unfreeze last 30% of base model layers
base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * FINE_TUNE_RATIO)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print(f"✓ Unfroze {len(base_model.layers) - fine_tune_at} layers for fine-tuning")
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"✓ New trainable parameters: {trainable_params:,}")

# Update checkpoint for phase 2
checkpoint_phase2 = ModelCheckpoint(
    'models/best_model_fast_finetuned.h5',
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

phase2_time = (time.time() - phase2_start) / 60
print(f"\n✓ Phase 2 completed in {phase2_time:.1f} minutes!")

# Calculate total training time
total_time = (time.time() - start_time) / 60
print(f"\n{'='*80}")
print(f"⏱️  TOTAL TRAINING TIME: {total_time:.1f} minutes ({total_time/60:.2f} hours)")
print(f"{'='*80}")

# Combine histories
history = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
}

# Evaluate on Test Set
print("\n[Final] Evaluating model on test set...")
print("=" * 80)

test_loss, test_accuracy, test_top3_accuracy = model.evaluate(test_generator, verbose=1)

print(f"\n{'='*80}")
print(f"FINAL RESULTS (Fast Training)")
print(f"{'='*80}")
print(f"Training Time: {total_time:.1f} minutes ({total_time/60:.2f} hours)")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Top-3 Accuracy: {test_top3_accuracy*100:.2f}%")
print(f"{'='*80}")

# Check if target was met
if total_time <= 120:
    print(f"✅ SUCCESS! Training completed in under 2 hours!")
else:
    print(f"⚠️  Training took {total_time/60:.2f} hours (target was 2 hours)")

# Save the final model
model.save('models/final_model_fast.h5')
print("\n✓ Model saved to 'models/final_model_fast.h5'")

# Plot training history
print("\nGenerating training plots...")

plt.figure(figsize=(15, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.axvline(x=len(history_phase1.history['loss']), color='r', linestyle='--', 
            label='Fine-tuning Start', linewidth=1.5)
plt.title(f'Model Accuracy (Fast Training)\nFinal: {test_accuracy*100:.2f}%', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss', linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
plt.axvline(x=len(history_phase1.history['loss']), color='r', linestyle='--', 
            label='Fine-tuning Start', linewidth=1.5)
plt.title(f'Model Loss (Fast Training)\nTime: {total_time:.1f} min', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/training_history_fast.png', dpi=300, bbox_inches='tight')
print("✓ Training plots saved to 'plots/training_history_fast.png'")

# Save training summary
summary = {
    'model': 'MobileNetV3Small',
    'training_time_minutes': round(total_time, 2),
    'training_time_hours': round(total_time/60, 2),
    'phase1_time_minutes': round(phase1_time, 2),
    'phase2_time_minutes': round(phase2_time, 2),
    'total_epochs': len(history['accuracy']),
    'test_accuracy': float(test_accuracy),
    'test_top3_accuracy': float(test_top3_accuracy),
    'test_loss': float(test_loss),
    'image_size': IMG_SIZE,
    'batch_size': BATCH_SIZE,
    'target_met': total_time <= 120
}

with open('models/training_summary_fast.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("🎉 FAST TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"✓ Best model: models/best_model_fast_finetuned.h5")
print(f"✓ Final model: models/final_model_fast.h5")
print(f"✓ Class indices: models/class_indices_fast.json")
print(f"✓ Training summary: models/training_summary_fast.json")
print(f"✓ Training plots: plots/training_history_fast.png")
print(f"✓ TensorBoard logs: logs/")
print("="*80)
print(f"\n💡 Training completed in {total_time:.1f} minutes with {test_accuracy*100:.2f}% accuracy!")
print("="*80)
