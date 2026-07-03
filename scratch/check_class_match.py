import os
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.utils.config import TRAIN_DIR, CLASS_INDICES_PATH

train_class_names = sorted(entry.name for entry in os.scandir(TRAIN_DIR) if entry.is_dir())

print(f"Number of classes in TRAIN_DIR: {len(train_class_names)}")

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

print(f"Number of classes in class_indices.json: {len(class_indices)}")

# Check index matching
mismatches = []
for idx, name in enumerate(train_class_names):
    json_idx = class_indices.get(name)
    if json_idx != idx:
        mismatches.append((name, idx, json_idx))

if mismatches:
    print(f"\nFound {len(mismatches)} mismatches:")
    for name, idx, json_idx in mismatches[:10]:
        print(f"  {name}: TRAIN_DIR index={idx}, JSON index={json_idx}")
else:
    print("\nClass indices match perfectly between TRAIN_DIR and class_indices.json!")
