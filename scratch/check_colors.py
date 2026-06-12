import os
import numpy as np
from tensorflow import keras

val_dir = "dataset/val"
scab_dir = os.path.join(val_dir, "Apple___Apple_scab")
imgs = os.listdir(scab_dir)[:2]
for img in imgs:
    img_path = os.path.join(scab_dir, img)
    img_array = keras.utils.img_to_array(keras.utils.load_img(img_path))
    corner1 = img_array[0:10, 0:10].mean(axis=(0,1))
    corner2 = img_array[-10:, -10:].mean(axis=(0,1))
    print(f"Image {img}: Top-left {corner1}, Bottom-right {corner2}")
