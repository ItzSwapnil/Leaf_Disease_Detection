import os

from src.training.training_utils import build_dynamic_yolo_dataset

TRAIN_DIR = "/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/dataset/train"
train_class_names = sorted(entry.name for entry in os.scandir(TRAIN_DIR) if entry.is_dir())

print("Building dataloader...")
train_loader = build_dynamic_yolo_dataset(TRAIN_DIR, train_class_names, 32, shuffle=True, seed=42, use_yolo=False, fraction=0.2)
print("Iterating...")
for i, batch in enumerate(train_loader):
    print("Batch", i)
    break
print("Done")
