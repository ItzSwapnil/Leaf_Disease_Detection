import glob
import os
import traceback

import keras
import numpy as np
import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from scripts.gradcam_check import _make_gradcam_heatmap
from src.training.train_model import FamilyDeviationClassifier


def test_gradcam_heatmap_generation():
    print("Finding model...")
    models = glob.glob("models/leaf_disease_checkpoint.keras")
    if not models:
        pytest.skip("No model found!")

    model_path = models[-1]
    print(f"Loading {model_path}...")
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "FamilyDeviationClassifier": FamilyDeviationClassifier
        },
    )

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
        print(
            "Success! Dimensions:", crop_heatmap.shape, disease_heatmap.shape
        )
        assert crop_heatmap.shape == (224, 224)
        assert disease_heatmap.shape == (224, 224)
    except Exception as e:
        print("FAILED with Exception:")
        traceback.print_exc()
        raise e
