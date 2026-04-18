"""Build a case gallery PNG by sampling images from dataset/train.

Usage: python scripts/generate_case_gallery.py
Saves: plots/case_gallery.png
"""

import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_TRAIN = ROOT / "dataset" / "train"
OUT = ROOT / "plots" / "case_gallery.png"


def collect_sample_images(max_tiles=12):
    classes = [p for p in sorted(DATA_TRAIN.iterdir()) if p.is_dir()]
    imgs = []
    for cls in classes:
        files = [
            f for f in cls.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        if files:
            imgs.append(random.choice(files))
        if len(imgs) >= max_tiles:
            break
    return imgs


def fit_image_to_frame(img, target_size):
    img = img.convert("RGB")
    src_w, src_h = img.size
    target_w, target_h = target_size

    if src_w * target_h == src_h * target_w:
        return img.resize((target_w, target_h), Image.LANCZOS)

    src_ratio = src_w / src_h
    target_ratio = target_w / target_h

    if src_ratio > target_ratio:
        # crop left/right
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        # crop top/bottom
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))

    return img.resize((target_w, target_h), Image.LANCZOS)


def make_grid(images, cols=4, thumb_size=(300, 300), out_path=OUT):
    if not images:
        print("No images found in dataset/train; aborting")
        return
    rows = (len(images) + cols - 1) // cols
    grid_w = cols * thumb_size[0]
    grid_h = rows * thumb_size[1]
    grid = Image.new("RGB", (grid_w, grid_h), (40, 40, 40))
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= len(images):
                break
            img = fit_image_to_frame(Image.open(images[i]), thumb_size)
            x = c * thumb_size[0]
            y = r * thumb_size[1]
            grid.paste(img, (x, y))
            i += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, dpi=(150, 150))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    imgs = collect_sample_images(max_tiles=12)
    make_grid(imgs, cols=4)
