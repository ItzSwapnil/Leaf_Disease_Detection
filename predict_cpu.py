import os
os.environ['TF_CPP_THREAD_POOL_SIZE'] = '4'
import sys
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

IMG_SIZE = 96
MODEL_PATH = 'models/best_model_cpu.h5'
CLASS_INDICES_PATH = 'models/class_indices_cpu.json'

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_cpu.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: Image '{img_path}' not found!")
        sys.exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found! Train first: python train_cpu_optimized.py")
        sys.exit(1)
    
    print("Loading model and class indices...")
    model = load_model(MODEL_PATH)
    
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
    
    print(f"Analyzing: {img_path}...")
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    
    print("\n" + "="*60)
    print(f"Predicted: {index_to_class[predicted_idx]}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
