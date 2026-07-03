import os

print("Starting...", flush=True)

try:
    from src.training.training_utils import build_dynamic_yolo_dataset
    print("Imported successfully.", flush=True)

    TRAIN_DIR = "/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/dataset/train"
    train_class_names = sorted(entry.name for entry in os.scandir(TRAIN_DIR) if entry.is_dir())
    print(f"Found {len(train_class_names)} classes.", flush=True)

    print("Building dataloader...", flush=True)
    train_loader = build_dynamic_yolo_dataset(TRAIN_DIR, train_class_names, 32, shuffle=True, seed=42, use_yolo=False, fraction=0.2)
    print("Iterating...", flush=True)
    for i, batch in enumerate(train_loader):
        print(f"Batch {i} loaded.", flush=True)
        break
    print("Done.", flush=True)
except Exception:
    import traceback
    traceback.print_exc()
