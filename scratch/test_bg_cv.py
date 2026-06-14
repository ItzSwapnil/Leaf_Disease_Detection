import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


def get_bg_mask_cv(img_array):
    """Robust lab-grade background extractor using OpenCV."""
    # Convert to uint8
    img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)

    # 1. Grab corners to find background colors
    h, w = img_uint8.shape[:2]
    corners = [
        img_uint8[0:10, 0:10],
        img_uint8[0:10, w-10:w],
        img_uint8[h-10:h, 0:10],
        img_uint8[h-10:h, w-10:w]
    ]

    # Check if image is "backgroundless" (if corners have high variance, they are leaf)
    corner_vars = [np.var(c) for c in corners]
    avg_var = np.mean(corner_vars)

    if avg_var > 500: # Highly textured corners -> probably backgroundless
        return np.zeros((h, w), dtype=np.float32)

    # It's a flat background. Let's use simple color clustering / distance from corners
    # Extract median color of the corners
    corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
    median_bg = np.median(corner_pixels, axis=0)

    # Calculate color distance
    diff = np.abs(img_uint8 - median_bg)
    dist = np.sum(diff, axis=-1)

    # Threshold: if close to background color, it's background
    bg_mask = (dist < 45).astype(np.float32)

    # Clean up with morphology
    kernel = np.ones((5, 5), np.uint8)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)

    return bg_mask

val_dir = "dataset/val"
classes = os.listdir(val_dir)
samples = []
target_classes = ["Rice___Leaf_Blast", "Corn_(maize)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Apple___Apple_scab"]
for c in target_classes:
    c_dir = os.path.join(val_dir, c)
    if os.path.isdir(c_dir):
        imgs = os.listdir(c_dir)
        if imgs:
            samples.extend([os.path.join(c_dir, img) for img in imgs[:4]])

random.shuffle(samples)
samples = samples[:8]

plt.figure(figsize=(16, 8))
for i, img_path in enumerate(samples):
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = keras.utils.img_to_array(img)

    bg_mask = get_bg_mask_cv(img_array)

    plt.subplot(2, 4, i + 1)
    display_img = img_array.copy() / 255.0
    display_img[..., 0] = np.where(bg_mask == 1, 1.0, display_img[..., 0])
    display_img[..., 1] = np.where(bg_mask == 1, 0.0, display_img[..., 1])
    display_img[..., 2] = np.where(bg_mask == 1, 0.0, display_img[..., 2])

    plt.imshow(display_img)
    plt.axis("off")
    plt.title(os.path.basename(os.path.dirname(img_path))[:15])

plt.tight_layout()
plt.savefig("scratch/bg_cv.png")
print("Saved to scratch/bg_cv.png")
