import os
import sys
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torch
import cv2

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline.predict import LeafDiseasePredictor
from src.core.leaf_segmentation import segment_leaf

def test_bg_fills():
    img_path = "/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/Tomato-Mosaic-Virus-300x300.webp"
    if not os.path.exists(img_path):
        print("Image not found.")
        return

    predictor = LeafDiseasePredictor()
    
    # 1. Load image and transpose EXIF
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    
    # 2. Run YOLO Leaf Detection
    detector = predictor._get_yolo_leaf_detector()
    x1, y1, x2, y2 = 0, 0, img.width, img.height
    if detector is not None:
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        detection = detector.detect(img_bgr)
        if detection["found"]:
            x1, y1, x2, y2 = detection["bbox"]
            print(f"YOLO bbox: {x1, y1, x2, y2}")
    
    img_cropped = img.crop((x1, y1, x2, y2))
    img_cropped_rgb = np.array(img_cropped)
    
    # 3. Precise Segmentation Mask
    seg_res = segment_leaf(img_cropped_rgb)
    if not seg_res["success"]:
        print("Segmentation failed.")
        return
        
    mask = seg_res["mask"]  # 255 for leaf, 0 for background
    
    # Try different background fill colors
    # We will test: Black (0), Mid Gray (128), Light Gray (200), Off-White (220), White (255)
    fill_colors = {
        "Black (0)": 0,
        "Mid Gray (128)": 128,
        "Light Gray (200)": 200,
        "Off-White (220)": 220,
        "White (255)": 255,
        "Original Unmasked Crop": None
    }
    
    for name, fill_val in fill_colors.items():
        if fill_val is not None:
            # Create a background filled with fill_val
            bg = np.full_like(img_cropped_rgb, fill_val)
            masked_img_arr = np.where(np.expand_dims(mask, axis=-1) == 255, img_cropped_rgb, bg)
            img_to_predict = Image.fromarray(masked_img_arr)
            # Save for visual inspection
            img_to_predict.save(f"plots/debug_fill_{name.split()[0]}.png")
        else:
            img_to_predict = img_cropped
            
        # Run prediction
        tensor = predictor.transform(img_to_predict).unsqueeze(0).to(predictor.device)
        
        all_preds = []
        with torch.no_grad():
            for m in predictor.models:
                out = m(tensor)
                logits = out["disease_output"] if isinstance(out, dict) and "disease_output" in out else out
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                all_preds.append(probs)
        
        predictions = np.mean(all_preds, axis=0)
        top_idx = int(np.argmax(predictions))
        class_name = predictor.idx_to_class[top_idx]
        confidence = float(predictions[top_idx])
        
        print(f"[{name}] Predicted: {class_name} with {confidence*100:.2f}% confidence")

if __name__ == "__main__":
    test_bg_fills()
