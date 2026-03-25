#!/usr/bin/env python3
import os, json

root = "dataset"
splits = ["train", "val", "test"]
out = {}

def is_image_file(fn):
    return fn.lower().endswith((".jpg",".jpeg",".png",".bmp",".tiff",".gif"))

for split in splits:
    d = os.path.join(root, split)
    if not os.path.isdir(d):
        continue
    counts = {}
    for cls in sorted(os.listdir(d)):
        cls_dir = os.path.join(d, cls)
        if not os.path.isdir(cls_dir):
            continue
        c = 0
        for _r, _dirs, files in os.walk(cls_dir):
            for f in files:
                if is_image_file(f):
                    c += 1
        counts[cls] = c
    total = sum(counts.values())
    out[split] = {"total_images": total, "per_class": counts}

os.makedirs("reports", exist_ok=True)
with open("reports/dataset_counts.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
