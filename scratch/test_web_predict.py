import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force PyTorch backend for Keras
os.environ["KERAS_BACKEND"] = "torch"

from src.web.app import load_model_and_classes, predict_disease

# Load model and classes
print("Loading model and classes via app.py...")
load_model_and_classes()

# Run prediction
img_path = "dataset/test/Apple___Cedar_apple_rust/064b14b5-af1b-4bcf-afe7-a061e5669dbb___FREC_C.Rust 9839.JPG"
print(f"\nRunning prediction on: {img_path}")

# Case 1: background removal ON
print("\n--- Test 1: Background removal ON ---")
result_bg_on = predict_disease(img_path, pipeline_options={"use_background_removal": True})
print("Result:")
for k, v in result_bg_on.items():
    if k not in ["image", "cropped_image"]:
        print(f"  {k}: {v}")

# Case 2: background removal OFF
print("\n--- Test 2: Background removal OFF ---")
result_bg_off = predict_disease(img_path, pipeline_options={"use_background_removal": False})
print("Result:")
for k, v in result_bg_off.items():
    if k not in ["image", "cropped_image"]:
        print(f"  {k}: {v}")
