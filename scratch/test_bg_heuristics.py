import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def get_bg_masks(x_orig):
    # Convert to grayscale for spatial variance
    gray = tf.image.rgb_to_grayscale(x_orig)
    
    # 1. Original (pixel-wise color variance)
    mean_color = tf.reduce_mean(x_orig, axis=-1, keepdims=True)
    color_var = tf.reduce_mean(tf.square(x_orig - mean_color), axis=-1, keepdims=True)
    color_std = tf.sqrt(color_var + 1e-8)
    bg_orig = tf.cast((color_std <= 8.0) | (mean_color <= 20.0), dtype=tf.float32)
    
    # 2. Spatial variance (3x3 window)
    mean_pool = tf.nn.avg_pool2d(gray, ksize=3, strides=1, padding="SAME")
    sq_mean_pool = tf.nn.avg_pool2d(tf.square(gray), ksize=3, strides=1, padding="SAME")
    spatial_var = sq_mean_pool - tf.square(mean_pool)
    spatial_std = tf.sqrt(tf.maximum(spatial_var, 0.0) + 1e-8)
    
    # Background if spatial std is low, OR it's black
    bg_spatial_3x3 = tf.cast((spatial_std <= 5.0) | (mean_color <= 20.0), dtype=tf.float32)
    
    # 3. Spatial variance (5x5 window)
    mean_pool_5 = tf.nn.avg_pool2d(gray, ksize=5, strides=1, padding="SAME")
    sq_mean_pool_5 = tf.nn.avg_pool2d(tf.square(gray), ksize=5, strides=1, padding="SAME")
    spatial_var_5 = sq_mean_pool_5 - tf.square(mean_pool_5)
    spatial_std_5 = tf.sqrt(tf.maximum(spatial_var_5, 0.0) + 1e-8)
    
    bg_spatial_5x5 = tf.cast((spatial_std_5 <= 5.0) | (mean_color <= 20.0), dtype=tf.float32)
    
    # 4. Combined: Spatial + Black + White
    # White: mean_color > 220
    # Black: mean_color < 30
    # Ceramic/Gray: spatial_std_5 <= 4.0
    bg_combined = tf.cast((spatial_std_5 <= 3.0) | (mean_color <= 25.0) | (mean_color >= 230.0), dtype=tf.float32)

    return bg_orig.numpy(), bg_spatial_3x3.numpy(), bg_spatial_5x5.numpy(), bg_combined.numpy()

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

plt.figure(figsize=(20, 16))
for i, img_path in enumerate(samples):
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = keras.utils.img_to_array(img)
    
    m1, m2, m3, m4 = get_bg_masks(img_array[np.newaxis, ...])
    m1, m2, m3, m4 = m1[0], m2[0], m3[0], m4[0]
    
    for j, (mask, name) in enumerate(zip([m1, m2, m3, m4], ["Orig", "Spatial 3x3", "Spatial 5x5", "Combined"])):
        plt.subplot(8, 4, i * 4 + j + 1)
        display_img = img_array.copy() / 255.0
        display_img[..., 0] = np.where(mask[..., 0] == 1, 1.0, display_img[..., 0])
        display_img[..., 1] = np.where(mask[..., 0] == 1, 0.0, display_img[..., 1])
        display_img[..., 2] = np.where(mask[..., 0] == 1, 0.0, display_img[..., 2])
        
        plt.imshow(display_img)
        plt.axis("off")
        if i == 0:
            plt.title(name)
        if j == 0:
            plt.text(-10, 112, os.path.basename(os.path.dirname(img_path))[:10], rotation=90, va="center")

plt.tight_layout()
plt.savefig("scratch/bg_spatial.png")
print("Saved to scratch/bg_spatial.png")
