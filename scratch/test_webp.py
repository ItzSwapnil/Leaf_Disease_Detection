import os
import sys
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import v2

os.environ["KERAS_BACKEND"] = "torch"

# Add project root to sys.path
ROOT = "/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline.predict import _load_model_robust, LeafDiseasePredictor

def test_webp_image():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/leaf_disease_checkpoint.pt"
    image_path = "/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/Tomato-Mosaic-Virus-300x300.webp"
    
    predictor = LeafDiseasePredictor(model_path=model_path)
    
    # 1. Load image and transpose EXIF
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    
    # 2. YOLO leaf detection
    detector = predictor._get_yolo_leaf_detector()
    if detector is not None:
        import cv2
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        detection = detector.detect(img_bgr)
        if detection["found"]:
            x1, y1, x2, y2 = detection["bbox"]
            print(f"YOLO found bbox: {x1, y1, x2, y2}")
            img = img.crop((x1, y1, x2, y2))
            
            # Precise GrabCut segmentation
            from src.core.leaf_segmentation import segment_leaf_grabcut
            img_arr = np.array(img)
            seg_res = segment_leaf_grabcut(img_arr)
            if seg_res["success"]:
                print("GrabCut segmentation successful.")
                img = Image.fromarray(seg_res["masked_image"])
            else:
                print("GrabCut segmentation failed.")
        else:
            print("YOLO did not find any leaf.")
    
    # Test different transforms
    transform_no_norm = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    transform_with_norm = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_no_norm = transform_no_norm(img).unsqueeze(0).to(device)
    tensor_with_norm = transform_with_norm(img).unsqueeze(0).to(device)

    model = predictor.models[0]
    
    with torch.no_grad():
        out_no_norm = model(tensor_no_norm)
        logits_no_norm = out_no_norm["disease_output"] if isinstance(out_no_norm, dict) else out_no_norm
        probs_no_norm = torch.softmax(logits_no_norm, dim=-1).cpu().numpy()[0]
        
        out_with_norm = model(tensor_with_norm)
        logits_with_norm = out_with_norm["disease_output"] if isinstance(out_with_norm, dict) else out_with_norm
        probs_with_norm = torch.softmax(logits_with_norm, dim=-1).cpu().numpy()[0]

    # Print top 5 classes for both
    indices_no_norm = np.argsort(probs_no_norm)[::-1][:5]
    indices_with_norm = np.argsort(probs_with_norm)[::-1][:5]
    
    print("\n--- Predictions WITHOUT Normalization ---")
    for idx in indices_no_norm:
        class_name = predictor.idx_to_class[idx]
        print(f"  {class_name}: {probs_no_norm[idx]*100:.2f}%")
        
    print("\n--- Predictions WITH Normalization ---")
    for idx in indices_with_norm:
        class_name = predictor.idx_to_class[idx]
        print(f"  {class_name}: {probs_with_norm[idx]*100:.2f}%")

if __name__ == "__main__":
    test_webp_image()
