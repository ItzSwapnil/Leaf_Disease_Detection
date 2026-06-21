"""Export a PyTorch ``.pt`` model checkpoint to a Keras 3 ``.keras`` file.

Satisfies the requirement to produce and use both formats by wrapping the
pure PyTorch architecture into a Keras model using TorchModuleWrapper.
"""

import argparse
import os

# Crucial for multi-backend support: set Keras backend to torch
os.environ["KERAS_BACKEND"] = "torch"

import json

import keras
import torch

from src.training.train_model import LeafDiseaseModel
from src.training.training_utils import parse_class_structure
from src.utils.config import CLASS_INDICES_PATH, FINAL_MODEL_FILE_PATH
from src.utils.hardware import get_device


def export_to_keras(pt_path: str, keras_path: str) -> None:
    print(f"Loading PyTorch model from: {pt_path}")
    device = get_device()

    # Need class names to initialize model
    with open(CLASS_INDICES_PATH, "r") as f:
        idx_to_class = {int(v): k for k, v in json.load(f).items()}

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)
    crop_names = sorted(list(set(name.split("___")[0] for name in class_names)))
    num_crops = len(crop_names)
    healthy_partners = parse_class_structure(class_names)

    path_hint = str(pt_path).lower()
    if any(token in path_hint for token in ["dino", "vit", "refined"]):
        backbone_name = "DINOv3"
    else:
        backbone_name = "EfficientNetV2B0"

    model = LeafDiseaseModel(backbone_name, num_classes, num_crops, healthy_partners)
    state = torch.load(pt_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to("cpu")  # Ensure CPU for export stability
    model.eval()

    print("Wrapping in Keras 3 architecture...")
    inputs = keras.Input(shape=(3, 224, 224), name="image")
    wrapper = keras.layers.TorchModuleWrapper(model)
    outputs = wrapper(inputs)

    # In Keras 3, TorchModuleWrapper preserves the dict output of LeafDiseaseModel
    keras_model = keras.Model(inputs, outputs)

    print(f"Saving Keras model to: {keras_path}")
    keras_model.save(keras_path)
    print("Export complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export .pt to .keras")
    parser.add_argument("--pt-path", default=FINAL_MODEL_FILE_PATH)
    parser.add_argument("--keras-path", default=FINAL_MODEL_FILE_PATH.replace(".pt", ".keras"))
    args = parser.parse_args()

    export_to_keras(args.pt_path, args.keras_path)
