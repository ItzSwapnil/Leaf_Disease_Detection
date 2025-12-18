import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Optimization for 4-Core CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# === CONFIGURATION FOR 99% TARGET ===
IMG_SIZE = 224         # Higher resolution for better detail
BATCH_SIZE = 32        # Stable batch size for CPU
EPOCHS_PHASE1 = 10     # Warm up the top layers
EPOCHS_PHASE2 = 30     # Deep fine-tuning (Total ~1 hour on 4 cores)
NUM_CLASSES = 46

BASE_DIR = '/workspaces/Leaf_Disease_Detection/dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

# === DATA AUGMENTATION (High Intensity) ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
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

# === MODEL BUILDING (EfficientNetV2) ===
base_model = EfficientNetV2B0(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='swish')(x) # Swish activation often helps reach 99%
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Phase 1: Training the Head
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), # Prevents overfitting
    metrics=['accuracy']
)

# === CALLBACKS ===
checkpoint = ModelCheckpoint('models/99pct_leaf_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)

print("\n--- Phase 1: Warming up (Frozen Base) ---")
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE1, callbacks=[checkpoint, reduce_lr], workers=4)

# === PHASE 2: DEEP FINE-TUNING ===
print("\n--- Phase 2: Deep Fine-tuning (Unfreezing) ---")
base_model.trainable = True # Unfreeze all layers for maximum accuracy

model.compile(
    optimizer=Adam(learning_rate=1e-5), # Extremely low LR for precision
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE2, callbacks=[checkpoint, reduce_lr, early_stop], workers=4)

print("\nTraining Complete. Best model saved to models/99pct_leaf_model.h5")
