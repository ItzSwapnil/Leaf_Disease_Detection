import os
import sys
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import cv2

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.yolo_leaf import YOLOLeafDetector
from src.core.leaf_segmentation import segment_leaf

def main():
    img_path = "/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/Tomato-Mosaic-Virus-300x300.webp"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} does not exist.")
        return

    # 1. Load and transpose
    img = Image.open(img_path)
    img_transposed = ImageOps.exif_transpose(img).convert("RGB")
    img_transposed.save("plots/debug_1_transposed.png")
    print("Saved debug_1_transposed.png")

    # 2. YOLO Detection
    detector = YOLOLeafDetector()
    img_bgr = cv2.cvtColor(np.array(img_transposed), cv2.COLOR_RGB2BGR)
    detection = detector.detect(img_bgr)
    print(f"YOLO detection result: {detection}")

    if detection["found"]:
        x1, y1, x2, y2 = detection["bbox"]
        img_cropped = img_transposed.crop((x1, y1, x2, y2))
        img_cropped.save("plots/debug_2_yolo_cropped.png")
        print("Saved debug_2_yolo_cropped.png")

        # 3. Contour Segmentation inside YOLO
        img_cropped_rgb = np.array(img_cropped)
        seg_res = segment_leaf(img_cropped_rgb)
        print(f"Segmentation success inside YOLO crop: {seg_res['success']}, leaf_count: {seg_res.get('leaf_count')}")
        if seg_res["success"]:
            Image.fromarray(seg_res["masked_image"]).save("plots/debug_3_segmented.png")
            print("Saved debug_3_segmented.png")
        else:
            print("Segmentation failed inside YOLO crop.")
    else:
        # Full image segmentation fallback
        img_rgb = np.array(img_transposed)
        seg_res = segment_leaf(img_rgb)
        print(f"Segmentation success on full image: {seg_res['success']}, leaf_count: {seg_res.get('leaf_count')}")
        if seg_res["success"]:
            Image.fromarray(seg_res["masked_image"]).save("plots/debug_3_segmented_full.png")
            print("Saved debug_3_segmented_full.png")
        else:
            print("Segmentation failed on full image.")

if __name__ == "__main__":
    main()
