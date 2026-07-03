import os
import sys
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import cv2

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_grabcut():
    img_path = "/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/Tomato-Mosaic-Virus-300x300.webp"
    img = Image.open(img_path)
    img_transposed = ImageOps.exif_transpose(img).convert("RGB")
    
    # Crop to YOLO bbox
    # YOLO found bbox: (0, 0, 297, 300)
    img_cropped = img_transposed.crop((0, 0, 297, 300))
    img_arr = np.array(img_cropped)
    
    h, w = img_arr.shape[:2]
    
    # Run GrabCut
    mask = np.zeros((h, w), dtype=np.uint8)
    bgdModel = np.zeros((1, 65), dtype=np.float64)
    fgdModel = np.zeros((1, 65), dtype=np.float64)
    
    # Initialize with a rectangle slightly inset from the borders
    rect = (5, 5, w - 10, h - 10)
    
    try:
        cv2.grabCut(img_arr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        # GrabCut outputs: 0=bg, 1=fg, 2=probable bg, 3=probable fg
        binary_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
        
        # Apply morphology to smooth the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        masked_img = np.zeros_like(img_arr)
        masked_img[binary_mask == 255] = img_arr[binary_mask == 255]
        
        Image.fromarray(masked_img).save("plots/debug_grabcut.png")
        print("Saved debug_grabcut.png successfully.")
    except Exception as e:
        print(f"GrabCut failed: {e}")

if __name__ == "__main__":
    test_grabcut()
