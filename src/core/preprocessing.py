from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def preprocess_batch_for_model(images: torch.Tensor) -> torch.Tensor:
    images = images.to(torch.float32)
    # If images are [0, 255], divide by 255
    if images.max() > 1.0:
        images = images / 255.0
    return v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(images)

def preprocess_batch_for_model_tf(
    images: torch.Tensor, backbone_name: str | None = None
) -> torch.Tensor:
    return preprocess_batch_for_model(images)

def preprocess_array_for_model(
    image_array: np.ndarray, backbone_name: str | None = None
) -> np.ndarray:
    arr = np.asarray(image_array, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    # Channel last -> channel first -> normalize -> channel last?
    # Numpy images are usually (H, W, C). Torch is (C, H, W).
    # Since we are returning a numpy array, let's normalize along C.
    return (arr - mean) / std

def get_preprocessing_fn(backbone_name: str | None = None):
    # For PyTorch torchvision, standard ImageNet preprocessing applies everywhere.
    return lambda arr: preprocess_array_for_model(arr, backbone_name)
