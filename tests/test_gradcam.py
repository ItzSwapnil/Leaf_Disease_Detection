import os
import sys
import numpy as np
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from scripts.gradcam_check import _make_gradcam_heatmap
from src.training.train_model import FamilyDeviationClassifier
import keras

print("Finding model...")
import glob
models = glob.glob("models/leaf_disease_checkpoint.keras")
if not models:
    print("No model found!")
    sys.exit(1)

model_path = models[-1]
print(f"Loading {model_path}...")
model = keras.models.load_model(model_path, custom_objects={"FamilyDeviationClassifier": FamilyDeviationClassifier})

# The callback uses self.model. functional_model is accessed automatically inside _make_gradcam_heatmap.
print("Generating dummy image...")
img = np.random.rand(1, 224, 224, 3).astype(np.float32)

print("Calling _make_gradcam_heatmap...")
try:
    crop_heatmap, disease_heatmap = _make_gradcam_heatmap(
        model=model,
        img_array=img,
        target_layer_name=None,
        pred_index=0,
        backbone_name="DINOv3",
        vit_block_idx=6,
    )
    print("Success! Dimensions:", crop_heatmap.shape, disease_heatmap.shape)
except Exception as e:
    print("FAILED with Exception:")
    traceback.print_exc()
