import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

@tf.function
def get_bg_mask_tf(x_orig):
    # x_orig shape (B, H, W, C)
    B = tf.shape(x_orig)[0]
    H = tf.shape(x_orig)[1]
    W = tf.shape(x_orig)[2]
    
    # Grab 16x16 corners
    c1 = x_orig[:, 0:16, 0:16, :]
    c2 = x_orig[:, 0:16, W-16:W, :]
    c3 = x_orig[:, H-16:H, 0:16, :]
    c4 = x_orig[:, H-16:H, W-16:W, :]
    
    # Stack corners: (B, 4, 16, 16, C)
    corners = tf.stack([c1, c2, c3, c4], axis=1)
    
    # Calculate variance of corners to detect "backgroundless"
    # If corners are highly textured (part of leaf), variance will be high
    corner_mean = tf.reduce_mean(corners, axis=[2, 3], keepdims=True)
    corner_var = tf.reduce_mean(tf.square(corners - corner_mean), axis=[2, 3])
    avg_var = tf.reduce_mean(corner_var, axis=[1, 2]) # (B,)
    
    # Find background color (mean of corners)
    bg_color = tf.reduce_mean(corners, axis=[1, 2, 3]) # (B, C)
    bg_color = tf.reshape(bg_color, [B, 1, 1, 3])
    
    # Distance to background color
    dist = tf.reduce_sum(tf.abs(x_orig - bg_color), axis=-1) # L1 distance, (B, H, W)
    
    # Threshold
    # If avg_var > 300, it's highly textured (backgroundless), so bg_mask = 0
    # Otherwise, if dist < 45, it's background
    bg_mask = tf.where(
        dist < 45.0,
        tf.ones_like(dist),
        tf.zeros_like(dist)
    )
    
    # Apply "backgroundless" condition
    is_bgless = tf.reshape(avg_var > 300.0, [B, 1, 1])
    bg_mask = tf.where(is_bgless, tf.zeros_like(bg_mask), bg_mask)
    
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
    
    bg_mask = get_bg_mask_tf(img_array[np.newaxis, ...])[0].numpy()
    
    plt.subplot(2, 4, i + 1)
    display_img = img_array.copy() / 255.0
    display_img[..., 0] = np.where(bg_mask == 1, 1.0, display_img[..., 0])
    display_img[..., 1] = np.where(bg_mask == 1, 0.0, display_img[..., 1])
    display_img[..., 2] = np.where(bg_mask == 1, 0.0, display_img[..., 2])
    
    plt.imshow(display_img)
    plt.axis("off")
    plt.title(os.path.basename(os.path.dirname(img_path))[:15])

plt.tight_layout()
plt.savefig("scratch/bg_tf_corners.png")
print("Saved to scratch/bg_tf_corners.png")
