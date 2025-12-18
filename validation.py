import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Load the best trained model
model = load_model('models/99pct_final_reached.h5')

# Setup validation data
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_gen = val_datagen.flow_from_directory(
    '/workspaces/Leaf_Disease_Detection/dataset/val',
    target_size=(160, 160),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print("\n--- Verifying Saved Model Accuracy ---")
loss, acc = model.evaluate(val_gen)
print(f"\n🎯 SAVED MODEL ACCURACY: {acc*100:.2f}%")
