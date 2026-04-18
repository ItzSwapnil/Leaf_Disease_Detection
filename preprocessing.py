from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

DINO_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
DINO_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def preprocess_batch_for_model(images: tf.Tensor) -> tf.Tensor:
    images = tf.cast(images, tf.float32)
    return preprocess_input(images)


def preprocess_batch_for_model_tf(
    images: tf.Tensor, backbone_name: str | None = None
) -> tf.Tensor:
    images = tf.cast(images, tf.float32)
    if backbone_name == "DINOv3":
        images = images / 255.0
        return (images - DINO_MEAN) / DINO_STD
    return preprocess_input(images)


def preprocess_array_for_model(image_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_array, dtype=np.float32)
    return preprocess_input(arr)
