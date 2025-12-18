import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os
import gc

# === EXTREME MEMORY OPTIMIZATION ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

gc.collect()
tf.keras.backend.clear_session()

print("=" * 60)
print("🎯 MEMORY-SAFE TRAINING TO 99%")
print("=" * 60)

# 1. Load the BEST model we have
print("\n📦 Loading best model (94.46%)...")
model = load_model('models/1_10th_precision_model.h5')

# 2. ONLY unfreeze top 30 layers (less memory)
print("🔓 Unfreezing only top 30 layers...")
for layer in model.layers:
    layer.trainable = False
for layer in model.layers[-30:]:
    layer.trainable = True

trainable_count = sum([1 for l in model.layers if l.trainable])
print(f"   Trainable layers: {trainable_count}")

# 3. TINY batch size
BATCH_SIZE = 4

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    '/workspaces/Leaf_Disease_Detection/dataset/train', 
    target_size=(160, 160), 
    batch_size=BATCH_SIZE, 
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    '/workspaces/Leaf_Disease_Detection/dataset/val', 
    target_size=(160, 160), 
    batch_size=BATCH_SIZE, 
    class_mode='categorical'
)

# 4. Moderate learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# 5. Callbacks
checkpoint = ModelCheckpoint(
    'models/99pct_final_reached.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=2,
    min_lr=1e-7, 
    verbose=1
)

# 6. SHORTER epochs, more of them
STEPS_PER_EPOCH = 150  # Shorter
VALIDATION_STEPS = 50

print(f"\n⚡ Training Config:")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Steps/epoch: {STEPS_PER_EPOCH}")
print(f"   Learning rate: 5e-5")

print("\n--- MEMORY-SAFE FINE-TUNING ---\n")

for epoch_batch in range(4):  # 4 batches of 5 epochs each
    print(f"\n🔄 Batch {epoch_batch + 1}/4")
    gc.collect()  # Clear memory between batches
    
    model.fit(
        train_gen, 
        steps_per_epoch=STEPS_PER_EPOCH, 
        validation_data=val_gen,
        validation_steps=VALIDATION_STEPS,
        epochs=5, 
        callbacks=[checkpoint, reduce_lr],
        verbose=1
    )
    
    # Reload best model for next batch
    gc.collect()
    model = load_model('models/99pct_final_reached.h5')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=max(5e-5 * (0.5 ** epoch_batch), 1e-7)),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

print("\n" + "=" * 60)
print("✅ Training complete!")
print("   Best model saved to: models/99pct_final_reached.h5")
print("=" * 60)


