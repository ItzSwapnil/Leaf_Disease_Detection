import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# === CPU OPTIMIZATION ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# === 1/10TH DATA CONFIGURATION ===
IMG_SIZE = 160         
BATCH_SIZE = 32        
# 240,000 images total / 10 = 24,000 images per training cycle.
# 24,000 / 32 (batch size) = 750 steps per epoch.
STEPS_PER_EPOCH = 750  
VALIDATION_STEPS = 100 
EPOCHS_PHASE1 = 10     
EPOCHS_PHASE2 = 15     

# === DATA LOADING (With EfficientNet Preprocessing) ===
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # CRITICAL: Fixes the 8% accuracy issue
    rotation_range=15,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    '/workspaces/Leaf_Disease_Detection/dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    '/workspaces/Leaf_Disease_Detection/dataset/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# === MODEL BUILDING ===
base_model = EfficientNetV2B0(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu')(x) 
x = Dropout(0.4)(x)
outputs = Dense(46, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Phase 1: Warming up
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), 
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), # Helps reach 99%
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint('models/1_10th_precision_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)

print(f"\n--- Phase 1: Training on 1/10th of {train_gen.samples} images ---")
model.fit(
    train_gen, 
    steps_per_epoch=STEPS_PER_EPOCH, 
    validation_data=val_gen, 
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS_PHASE1, 
    callbacks=[checkpoint, reduce_lr]
)

# Phase 2: Fine-Tuning
base_model.trainable = True
for layer in base_model.layers[:-50]: # Unfreeze top layers for accuracy boost
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

print("\n--- Phase 2: Deep Fine-tuning (Precision Mode) ---")
model.fit(
    train_gen, 
    steps_per_epoch=STEPS_PER_EPOCH, 
    validation_data=val_gen, 
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS_PHASE2, 
    callbacks=[checkpoint, reduce_lr]
)