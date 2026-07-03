import os
import sys
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import cv2

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.leaf_segmentation import segment_leaf

def segment_largest_only(image):
    # Normalize image to uint8.
    if image.dtype == np.float32 or image.max() <= 1.0:
        img_uint8 = (image * 255.0).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)

    h, w = image.shape[:2]

    # Color Masking (HSV)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    lower_color = np.array([10, 20, 20])
    upper_color = np.array([155, 255, 255])
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # Edge Detection (Canny)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median_val = float(np.median(blurred))
    lower_thresh = int(max(0, (1.0 - 0.33) * median_val))
    upper_thresh = int(min(255, (1.0 + 0.33) * median_val))
    edges = cv2.Canny(blurred, lower_thresh, upper_thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    combined_mask = cv2.bitwise_or(color_mask, dilated_edges)

    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, cleanup_kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, cleanup_kernel)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        # Get largest contour by area
        largest_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [largest_cnt], -1, 255, -1)

    masked_image = np.zeros_like(image)
    masked_image[final_mask == 255] = image[final_mask == 255]
    return masked_image

def main():
    img_path = "/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/Tomato-Mosaic-Virus-300x300.webp"
    img = Image.open(img_path)
    img_transposed = ImageOps.exif_transpose(img).convert("RGB")
    
    img_arr = np.array(img_transposed)
    masked_largest = segment_largest_only(img_arr)
    
    Image.fromarray(masked_largest).save("plots/debug_largest_only.png")
    print("Saved debug_largest_only.png")

if __name__ == "__main__":
    main()
