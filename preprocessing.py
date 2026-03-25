from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

def preprocess_batch_for_model(images: tf.Tensor) -> tf.Tensor:
    images = tf.cast(images, tf.float32)
    return preprocess_input(images)

def preprocess_array_for_model(image_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_array, dtype=np.float32)
    return preprocess_input(arr)
