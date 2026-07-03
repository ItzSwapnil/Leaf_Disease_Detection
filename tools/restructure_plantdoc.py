import os
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

# Standardized crop-disease mapping
PLANTDOC_CLASS_MAP = {
    "Blueberry leaf": "Blueberry___healthy",
    "Tomato leaf yellow virus": "Tomato___Tomato_yellow_leaf_curl_virus",
    "Peach leaf": "Peach___healthy",
    "Raspberry leaf": "Raspberry___healthy",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Corn leaf blight": "Corn___Northern_Leaf_Blight",
    "Potato leaf early blight": "Potato___Early_blight",
    "Bell_pepper leaf": "Pepper_bell___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Bell_pepper leaf spot": "Pepper_bell___Bacterial_spot",
    "Soyabean leaf": "Soybean___healthy",
    "Potato leaf late blight": "Potato___Late_blight",
    "Apple leaf": "Apple___healthy",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Cherry leaf": "Cherry___healthy",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "grape leaf": "Grape___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Apple Scab Leaf": "Apple___Apple_scab",
    "grape leaf black rot": "Grape___Black_rot",
    "Corn rust leaf": "Corn___Common_rust",
    "Corn Gray leaf spot": "Corn___Gray_leaf_spot",
    "Potato leaf": "Potato___healthy",
    "Tomato two spotted spider mites leaf": "Tomato___Two-spotted_spider_mite",
}

def process_split(csv_path: Path, src_img_dir: Path, dest_dir: Path):
    print(f"Processing {csv_path.name}...")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"class": "class_name"})
    
    # Track statistics
    skipped_not_found = 0
    skipped_invalid_box = 0
    saved_crops = 0
    
    # We group by filename to open each image only once
    grouped = df.groupby('filename')
    
    for filename, group in tqdm(grouped):
        img_path = src_img_dir / filename
        if not img_path.exists():
            # Check lowercase/uppercase extensions
            alt_path = src_img_dir / (filename.rsplit('.', 1)[0] + '.' + filename.rsplit('.', 1)[1].lower())
            if alt_path.exists():
                img_path = alt_path
            else:
                skipped_not_found += len(group)
                continue
                
        img = cv2.imread(str(img_path))
        if img is None:
            skipped_not_found += len(group)
            continue
            
        h, w = img.shape[:2]
        
        for idx, row in enumerate(group.itertuples()):
            class_name = row.class_name
            if class_name not in PLANTDOC_CLASS_MAP:
                continue
                
            mapped_class = PLANTDOC_CLASS_MAP[class_name]
            
            # Coordinates
            xmin = int(max(0, row.xmin))
            ymin = int(max(0, row.ymin))
            xmax = int(min(w, row.xmax))
            ymax = int(min(h, row.ymax))
            
            # Validation
            if xmax <= xmin or ymax <= ymin:
                skipped_invalid_box += 1
                continue
                
            # Crop image
            cropped = img[ymin:ymax, xmin:xmax]
            if cropped.size == 0:
                skipped_invalid_box += 1
                continue
                
            # Create destination folder
            out_class_dir = dest_dir / mapped_class
            out_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Construct output filename
            base_name = Path(filename).stem
            ext = Path(filename).suffix
            if not ext:
                ext = ".jpg"
            out_path = out_class_dir / f"{base_name}_crop_{idx}{ext}"
            
            # Save crop
            cv2.imwrite(str(out_path), cropped)
            saved_crops += 1
            
    print(f"Split {csv_path.name} finished: Saved={saved_crops}, Skipped (Not Found)={skipped_not_found}, Skipped (Invalid Box)={skipped_invalid_box}")

def main():
    base_src = Path("plantdoc_download/PlantDoc-Object-Detection-Dataset")
    dest_base = Path("dataset")
    
    # We clean/recreate dataset directory
    train_dest = dest_base / "train"
    val_dest = dest_base / "val"
    
    process_split(base_src / "train_labels.csv", base_src / "TRAIN", train_dest)
    process_split(base_src / "test_labels.csv", base_src / "TEST", val_dest)

if __name__ == "__main__":
    main()
