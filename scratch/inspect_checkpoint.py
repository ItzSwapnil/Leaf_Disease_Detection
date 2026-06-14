import os
import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = "models/leaf_disease_checkpoint.keras"
print(f"Loading {model_path}...")
try:
    model = load_model(model_path, compile=False)
    print("Successfully loaded model!")
    model.summary()
except Exception as e:
    print(f"Failed to load model: {e}")
