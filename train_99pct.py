import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# === CPU OPTIMIZATION FOR 4 CORES ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# === SPEED & ACCURACY CONFIGURATION ===
IMG_SIZE = 160         # Optimized balance between CPU speed and accuracy
BATCH_SIZE = 32        
EPOCHS_PHASE1 = 8      # Fast warm-up for the new classifier head
EPOCHS_PHASE2 = 15     # Deep fine-tuning for the 99% goal
NUM_CLASSES = 46

# SUB-SAMPLING: This is the key to finishing in ~1 hour
# 500 steps * 32 images = 16,000 images per epoch (out of 62k total)
# Because of data augmentation, the model sees different images every epoch.
STEPS_PER_EPOCH = 500  
VALIDATION_STEPS = 100 

BASE_DIR = '/workspaces/Leaf_Disease_Detection/dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

# === DATA AUGMENTATION ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical'
)

# === MODEL BUILDING (EfficientNetV2-B0) ===
base_model = EfficientNetV2B0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), 
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='swish')(x) # Swish is faster/better for EfficientNet
x = Dropout(0.4)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Phase 1 Compile: Warming up the classifier
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), # Vital for 99%
    metrics=['accuracy']
)

# === CALLBACKS ===
os.makedirs('models', exist_ok=True)
checkpoint = ModelCheckpoint('models/99pct_leaf_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# === TRAINING EXECUTION ===

print(f"\n--- Starting Phase 1: {EPOCHS_PHASE1} Epochs (Frozen Base) ---")
model.fit(
    train_gen, 
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_gen, 
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS_PHASE1, 
    callbacks=[checkpoint, reduce_lr]
)

print(f"\n--- Starting Phase 2: {EPOCHS_PHASE2} Epochs (Fine-tuning Top 50 Layers) ---")
# On CPU, unfreezing just the top 50 layers is faster than unfreezing everything
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5), # Precision learning rate
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.fit(
    train_gen, 
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_gen, 
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS_PHASE2, 
    callbacks=[checkpoint, reduce_lr, early_stop]
)

print("\nSuccess! Best model saved to models/99pct_leaf_model.h5")
