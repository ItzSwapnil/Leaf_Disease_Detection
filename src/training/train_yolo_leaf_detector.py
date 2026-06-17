"""
Train a YOLOv26 leaf detector using auto-generated bounding box annotations.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path

import cv2

from src.core.leaf_segmentation import segment_leaf
from src.utils.config import MODELS_DIR, TRAIN_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_yolo_dataset(
    train_dir: str | Path,
    output_dir: str | Path,
    num_samples: int = 2000,
    train_fraction: float = 0.9,
    seed: int = 42,
) -> bool:
    """Generate YOLO format dataset using contour-based bounding boxes."""
    random.seed(seed)
    train_dir = Path(train_dir)
    output_dir = Path(output_dir)

    print(f"[*] Scanning images in {train_dir}...")
    image_paths = list(train_dir.rglob("*.jpg")) + list(
        train_dir.rglob("*.png")
    )
    if not image_paths:
        print("[ERROR] No images found in training directory.")
        return False

    random.shuffle(image_paths)

    # Output directories
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"[*] Generating leaf annotations for {num_samples} samples...")
    success_count = 0
    idx = 0

    while success_count < num_samples and idx < len(image_paths):
        img_path = image_paths[idx]
        idx += 1

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        res = segment_leaf(img)

        # If segment_leaf successfully finds leaf contours
        if res["success"] and res["contours"]:
            # Find largest contour
            largest_cnt = max(res["contours"], key=cv2.contourArea)
            rx, ry, rw, rh = cv2.boundingRect(largest_cnt)

            if rw > 10 and rh > 10:
                # Convert to normalized YOLO format (0..1)
                cx = (rx + rw / 2.0) / w
                cy = (ry + rh / 2.0) / h
                nw = rw / w
                nh = rh / h

                # Determine split
                split = (
                    "train" if random.random() < train_fraction else "val"
                )

                # Destination paths
                dest_img_name = f"leaf_{success_count}{img_path.suffix}"
                dest_img_path = output_dir / "images" / split / dest_img_name
                dest_lbl_path = (
                    output_dir
                    / "labels"
                    / split
                    / f"leaf_{success_count}.txt"
                )

                # Copy image and write label
                shutil.copy(str(img_path), str(dest_img_path))
                with open(dest_lbl_path, "w", encoding="utf-8") as f:
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                success_count += 1
                if success_count % 200 == 0:
                    print(f"    Processed {success_count}/{num_samples}...")

    print(
        f"[+] Successfully generated YOLO dataset under {output_dir} with {success_count} images."
    )

    # Write data.yaml config file
    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: leaf
"""
    with open(output_dir / "leaf_data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return success_count > 0


def main():
    """Train YOLOv26 leaf detector."""
    yolo_dataset_dir = PROJECT_ROOT / "dataset" / "yolo_dataset"

    # 1. Generate annotations
    if not (yolo_dataset_dir / "leaf_data.yaml").exists():
        print("[*] Generating auto-labeled YOLO dataset...")
        success = generate_yolo_dataset(
            train_dir=TRAIN_DIR,
            output_dir=yolo_dataset_dir,
            num_samples=2000,
        )
        if not success:
            print("[ERROR] Failed to generate YOLO dataset.")
            sys.exit(1)
    else:
        print("[*] YOLO dataset config already exists, skipping generation.")

    # 2. Fine-tune YOLO26n
    print("[*] Initializing YOLO26n pre-trained model...")
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")

    print("[*] Fine-tuning YOLO26n on leaf detection...")
    model.train(
        data=str(yolo_dataset_dir / "leaf_data.yaml"),
        epochs=5,
        imgsz=224,
        batch=32,
        device=0,  # GPU 0
        project=str(PROJECT_ROOT / "models" / "yolo26_train"),
        name="yolo26_leaf",
        verbose=True,
    )

    # 3. Export trained weights to final location
    src_weights = (
        PROJECT_ROOT
        / "models"
        / "yolo26_train"
        / "yolo26_leaf"
        / "weights"
        / "best.pt"
    )
    dest_weights = Path(MODELS_DIR) / "yolo26_leaf_detector.pt"

    if src_weights.exists():
        os.makedirs(MODELS_DIR, exist_ok=True)
        shutil.copy(str(src_weights), str(dest_weights))
        print(f"[+] YOLOv26 leaf detector saved successfully to {dest_weights}")

        # Clean up training run folder to save space
        shutil.rmtree(str(PROJECT_ROOT / "models" / "yolo26_train"))
    else:
        print("[ERROR] Training completed but best.pt was not found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
