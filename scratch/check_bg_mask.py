import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Simulate the TF background extraction from saliency_alignment.py
def get_bg_mask_tf(x_orig):
    mean_val = tf.reduce_mean(x_orig, axis=-1, keepdims=True)
    variance = tf.reduce_mean(tf.square(x_orig - mean_val), axis=-1, keepdims=True)
    std_val = tf.sqrt(variance + 1e-8)

    bg_mask = tf.cast((std_val <= 8.0) | (mean_val <= 20.0), dtype=tf.float32)
    return bg_mask.numpy()

# Get 16 random images
val_dir = "dataset/val"
classes = os.listdir(val_dir)
samples = []
for c in classes:
    c_dir = os.path.join(val_dir, c)
    if os.path.isdir(c_dir):
        imgs = os.listdir(c_dir)
        if imgs:
            samples.append(os.path.join(c_dir, random.choice(imgs)))

random.shuffle(samples)
samples = samples[:16]

plt.figure(figsize=(16, 16))
for i, img_path in enumerate(samples):
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = keras.utils.img_to_array(img)

    bg_mask = get_bg_mask_tf(img_array[np.newaxis, ...])[0]

    plt.subplot(4, 4, i + 1)
    # Highlight background in red
    display_img = img_array.copy() / 255.0
    display_img[..., 0] = np.where(bg_mask[..., 0] == 1, 1.0, display_img[..., 0])
    display_img[..., 1] = np.where(bg_mask[..., 0] == 1, 0.0, display_img[..., 1])
    display_img[..., 2] = np.where(bg_mask[..., 0] == 1, 0.0, display_img[..., 2])

    plt.imshow(display_img)
    plt.title(os.path.basename(os.path.dirname(img_path))[:15])
    plt.axis("off")

plt.tight_layout()
plt.savefig("scratch/bg_mask_test.png")
print("Saved to scratch/bg_mask_test.png")
