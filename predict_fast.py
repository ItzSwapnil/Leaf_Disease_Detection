"""
Fast Prediction Script
Works with fast-trained model (train_model_fast.py)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# Configuration for fast model
IMG_SIZE = 160
MODEL_PATH = 'models/best_model_fast.h5'
CLASS_INDICES_PATH = 'models/class_indices_fast.json'

def load_class_indices():
    """Load class indices mapping"""
    if not os.path.exists(CLASS_INDICES_PATH):
        print(f"Error: {CLASS_INDICES_PATH} not found!")
        print("Train the model first: python train_model_fast.py")
        sys.exit(1)
    
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

def predict_image(img_path, model, index_to_class):
    """Predict disease from image"""
    # Load and preprocess
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    
    # Top 3
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3 = [(index_to_class[idx], predictions[0][idx]) for idx in top_3_idx]
    
    return index_to_class[predicted_idx], confidence, top_3

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_fast.py <image_path>")
        print("Example: python predict_fast.py leaf_image.jpg")
        sys.exit(1)
    
    img_path = sys.argv[1]
    
    if not os.path.exists(img_path):
        print(f"Error: Image '{img_path}' not found!")
        sys.exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model '{MODEL_PATH}' not found!")
        print("Train the model first: python train_model_fast.py")
        sys.exit(1)
    
    print("=" * 80)
    print("Plant Leaf Disease Detection - Fast Model")
    print("=" * 80)
    
    # Load
    print(f"\nLoading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("✓ Model loaded")
    
    print(f"Loading class indices...")
    index_to_class = load_class_indices()
    print(f"✓ {len(index_to_class)} classes loaded")
    
    # Predict
    print(f"\nAnalyzing: {img_path}...")
    predicted_class, confidence, top_3 = predict_image(img_path, model, index_to_class)
    
    # Results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"\nPredicted Disease: {predicted_class}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nTop 3 Predictions:")
    for i, (class_name, conf) in enumerate(top_3, 1):
        print(f"  {i}. {class_name}: {conf*100:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
