#!/usr/bin/env python3
"""Generate publication-ready figures from repository artifacts and dataset images."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"
DATASET_DIR = ROOT / "dataset"
REPORT_PATH = ROOT / "reports" / "evaluation_report.json"
COUNTS_PATH = ROOT / "reports" / "dataset_counts.json"
CLASS_INDEX_PATH = ROOT / "models" / "class_indices.json"
TRAIN_HISTORY_PATH = ROOT / "models" / "logs" / "train_history.csv"
MISCLASS_SUMMARY_PATH = ROOT / "reports" / "misclassifications" / "summary.json"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
RNG = random.Random(42)
FIG_DPI = 800


def canonical_label(label: str) -> str:
    label = label.strip()
    label = label.replace(" ", "_")
    label = re.sub(r"_+", "_", label)
    return label.lower()


def pretty_label(label: str) -> str:
    if "___" in label:
        crop, disease = label.split("___", 1)
        crop = crop.replace(",", "").replace("_", " ")
        disease = disease.replace("_", " ")
        return f"{crop} / {disease}"
    return label.replace("_", " ")


def crop_of(label: str) -> str:
    if "___" in label:
        return label.split("___", 1)[0].replace(",", "").replace("_", " ")
    return label.replace("_", " ")


def load_counts() -> dict:
    return json.loads(COUNTS_PATH.read_text(encoding="utf-8"))


def load_report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def load_class_order() -> list[str]:
    class_indices = json.loads(CLASS_INDEX_PATH.read_text(encoding="utf-8"))
    return [label for label, _ in sorted(class_indices.items(), key=lambda item: item[1])]


def load_history() -> list[dict]:
    with TRAIN_HISTORY_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {
                "epoch": int(row["epoch"]) + 1,
                "accuracy": float(row["accuracy"]),
                "loss": float(row["loss"]),
                "val_accuracy": float(row["val_accuracy"]),
                "val_loss": float(row["val_loss"]),
            }
            for row in reader
        ]
    return rows


def resolve_dataset_class(split_counts: dict, split_dir: Path, target_label: str) -> str | None:
    target = canonical_label(target_label)
    for class_name in split_counts:
        if canonical_label(class_name) == target and (split_dir / class_name).is_dir():
            return class_name
    return next(
        (child.name for child in sorted(split_dir.iterdir()) if child.is_dir() and canonical_label(child.name) == target),
        None,
    )


def class_color_map(labels: list[str]) -> dict[str, tuple]:
    crops = sorted({crop_of(label) for label in labels})
    palette = sns.color_palette("tab20", n_colors=max(3, len(crops)))
    return {crop: palette[i % len(palette)] for i, crop in enumerate(crops)}


def save_plot(path: Path) -> None:
    plt.tight_layout()
    png_path = path.with_suffix(".png")
    plt.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {png_path.relative_to(ROOT)}")


def plot_crop_distribution(counts: dict) -> None:
    crop_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for split in ("train", "val", "test"):
        for label, n_images in counts[split]["per_class"].items():
            crop_counts[crop_of(label)][split] += int(n_images)

    crops = sorted(crop_counts, key=lambda crop: crop_counts[crop]["train"], reverse=True)
    train_vals = [crop_counts[c]["train"] for c in crops]
    val_vals = [crop_counts[c]["val"] for c in crops]
    test_vals = [crop_counts[c]["test"] for c in crops]

    x = np.arange(len(crops))
    plt.figure(figsize=(14, 6))
    plt.bar(x, train_vals, label="Train", color="#2a9d8f")
    plt.bar(x, val_vals, bottom=train_vals, label="Validation", color="#e9c46a")
    bottom = np.array(train_vals) + np.array(val_vals)
    plt.bar(x, test_vals, bottom=bottom, label="Test", color="#f4a261")
    plt.xticks(x, crops, rotation=45, ha="right")
    plt.ylabel("Images")
    plt.title("Per-Crop Dataset Composition Across Splits")
    plt.legend(frameon=False)
    save_plot(PLOTS_DIR / "crop_distribution.png")


def plot_class_imbalance(counts: dict) -> None:
    items = sorted(counts["train"]["per_class"].items(), key=lambda item: item[1], reverse=True)
    labels = [pretty_label(label) for label, _ in items]
    values = np.array([value for _, value in items], dtype=float)
    ranks = np.arange(1, len(values) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2.2, 1.4])
    axes[0].plot(ranks, values, color="#264653", linewidth=2.4)
    axes[0].fill_between(ranks, values, color="#2a9d8f", alpha=0.18)
    axes[0].scatter([1, len(values)], [values[0], values[-1]], color="#e76f51", s=80, zorder=5)
    axes[0].set_ylabel("Training images")
    axes[0].set_xlabel("Class rank (descending)")
    axes[0].set_title("Long-Tailed Class Distribution in the Training Split")
    axes[0].annotate(labels[0], (1, values[0]), xytext=(3, values[0] * 0.9), arrowprops={"arrowstyle": "->"})
    axes[0].annotate(labels[-1], (len(values), values[-1]), xytext=(len(values) - 13, values[-1] * 10), arrowprops={"arrowstyle": "->"})

    top = items[:8]
    bottom = items[-8:]
    top_labels = [pretty_label(label) for label, _ in reversed(top)]
    top_values = [value for _, value in reversed(top)]
    bottom_labels = [pretty_label(label) for label, _ in bottom]
    bottom_values = [value for _, value in bottom]

    axes[1].barh(np.arange(len(top_labels)), top_values, color="#457b9d", label="Largest classes")
    axes[1].barh(np.arange(len(bottom_labels)) + len(top_labels) + 1, bottom_values, color="#e76f51", label="Smallest classes")
    y_ticks = list(np.arange(len(top_labels))) + list(np.arange(len(bottom_labels)) + len(top_labels) + 1)
    y_labels = top_labels + bottom_labels
    axes[1].set_yticks(y_ticks, y_labels)
    axes[1].set_xlabel("Training images")
    axes[1].set_title("Largest and Smallest Training Classes")
    axes[1].legend(frameon=False, loc="lower right")
    save_plot(PLOTS_DIR / "class_imbalance.png")


def plot_top_bottom_classes(counts: dict) -> None:
    items = sorted(counts["train"]["per_class"].items(), key=lambda item: item[1], reverse=True)
    largest = items[:10]
    smallest = list(reversed(items[-10:]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].barh([pretty_label(label) for label, _ in reversed(largest)], [value for _, value in reversed(largest)], color="#457b9d")
    axes[0].set_title("Top 10 Training Classes by Size")
    axes[0].set_xlabel("Images")

    axes[1].barh([pretty_label(label) for label, _ in smallest], [value for _, value in smallest], color="#e76f51")
    axes[1].set_title("Bottom 10 Training Classes by Size")
    axes[1].set_xlabel("Images")
    save_plot(PLOTS_DIR / "top_bottom_classes.png")


def plot_per_class_f1(report: dict) -> None:
    labels = list(report["per_class_metrics"].keys())
    metrics = report["per_class_metrics"]
    ordered = sorted(labels, key=lambda label: metrics[label]["f1"])
    colors = class_color_map(ordered)
    color_values = [colors[crop_of(label)] for label in ordered]

    plt.figure(figsize=(15, 12))
    plt.barh(np.arange(len(ordered)), [metrics[label]["f1"] * 100 for label in ordered], color=color_values)
    plt.yticks(np.arange(len(ordered)), [pretty_label(label) for label in ordered], fontsize=8)
    plt.xlabel("Validation F1 (%)")
    plt.title("Per-Class Validation F1 Ranked from Lowest to Highest")
    plt.xlim(0, 102)
    save_plot(PLOTS_DIR / "per_class_f1_ranked.png")


def plot_precision_recall_support(report: dict) -> None:
    labels = list(report["per_class_metrics"].keys())
    metrics = report["per_class_metrics"]
    crop_colors = class_color_map(labels)
    plt.figure(figsize=(10, 8))
    for label in labels:
        data = metrics[label]
        plt.scatter(
            data["precision"] * 100,
            data["recall"] * 100,
            s=40 + data["support"] * 0.4,
            color=crop_colors[crop_of(label)],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.4,
        )
    lowest = sorted(labels, key=lambda label: metrics[label]["f1"])[:6]
    for label in lowest:
        data = metrics[label]
        plt.annotate(
            pretty_label(label),
            (data["precision"] * 100, data["recall"] * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    plt.xlabel("Precision (%)")
    plt.ylabel("Recall (%)")
    plt.xlim(80, 101)
    plt.ylim(80, 101)
    plt.title("Per-Class Precision vs Recall (Marker Size = Validation Support)")
    save_plot(PLOTS_DIR / "precision_recall_support.png")


def plot_crop_level_f1(report: dict) -> None:
    per_crop: dict[str, list[float]] = defaultdict(list)
    support_by_crop: dict[str, int] = defaultdict(int)
    for label, data in report["per_class_metrics"].items():
        crop = crop_of(label)
        per_crop[crop].append(float(data["f1"]))
        support_by_crop[crop] += int(data["support"])

    ordered = sorted(per_crop, key=lambda crop: np.mean(per_crop[crop]))
    values = [np.mean(per_crop[crop]) * 100 for crop in ordered]
    supports = [support_by_crop[crop] for crop in ordered]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(np.arange(len(ordered)), values, color=sns.color_palette("crest", len(ordered)))
    plt.xticks(np.arange(len(ordered)), ordered, rotation=45, ha="right")
    plt.ylabel("Macro F1 across crop classes (%)")
    plt.ylim(80, 101)
    plt.title("Crop-Level Validation Macro F1")
    for bar, support in zip(bars, supports):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"n={support}", ha="center", va="bottom", fontsize=8, rotation=90)
    save_plot(PLOTS_DIR / "crop_level_f1.png")


def plot_top_confusions(report: dict) -> None:
    pairs = report["top_confused_pairs"][:12]
    labels = [f"{pretty_label(pair['true_class'])}\n-> {pretty_label(pair['pred_class'])}" for pair in pairs]
    values = [int(pair["count"]) for pair in pairs]
    plt.figure(figsize=(14, 8))
    plt.barh(np.arange(len(labels)), values, color="#6d597a")
    plt.yticks(np.arange(len(labels)), labels, fontsize=8)
    plt.xlabel("Confusion count")
    plt.title("Most Frequent Off-Diagonal Validation Errors")
    save_plot(PLOTS_DIR / "top_confusions.png")


def plot_error_share_by_crop(report: dict, class_order: list[str]) -> None:
    cm = np.array(report["confusion_matrix"], dtype=float)
    row_totals = cm.sum(axis=1)
    correct = np.diag(cm)
    errors = row_totals - correct
    crop_errors: dict[str, float] = defaultdict(float)
    crop_support: dict[str, float] = defaultdict(float)

    for idx, label in enumerate(class_order):
        crop = crop_of(label)
        crop_errors[crop] += float(errors[idx])
        crop_support[crop] += float(row_totals[idx])

    ordered = sorted(crop_errors, key=lambda crop: (crop_errors[crop] / max(crop_support[crop], 1.0)), reverse=True)
    error_rates = [100.0 * crop_errors[crop] / max(crop_support[crop], 1.0) for crop in ordered]
    total_errors = [int(crop_errors[crop]) for crop in ordered]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(np.arange(len(ordered)), error_rates, color="#e76f51")
    ax1.set_xticks(np.arange(len(ordered)), ordered, rotation=45, ha="right")
    ax1.set_ylabel("Error rate within crop (%)")
    ax1.set_title("Validation Error Burden by Crop")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(ordered)), total_errors, color="#264653", marker="o", linewidth=2)
    ax2.set_ylabel("Absolute errors")
    save_plot(PLOTS_DIR / "error_share_by_crop.png")


def plot_rice_confusion(report: dict, class_order: list[str]) -> None:
    rice_labels = [label for label in class_order if label.startswith("Rice___")]
    indices = [class_order.index(label) for label in rice_labels]
    cm = np.array(report["confusion_matrix"], dtype=float)[np.ix_(indices, indices)]
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        normalized * 100,
        annot=True,
        fmt=".1f",
        cmap="magma",
        xticklabels=[pretty_label(label).split(" / ", 1)[1] for label in rice_labels],
        yticklabels=[pretty_label(label).split(" / ", 1)[1] for label in rice_labels],
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Rice-Class Validation Confusion Matrix (Row-Normalized %)")
    save_plot(PLOTS_DIR / "rice_confusion_matrix.png")


def plot_training_dynamics(history: list[dict]) -> None:
    epochs = [row["epoch"] for row in history]
    train_acc = [row["accuracy"] * 100 for row in history]
    val_acc = [row["val_accuracy"] * 100 for row in history]
    train_loss = [row["loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    best_epoch = max(history, key=lambda row: row["val_accuracy"])["epoch"]
    phase_boundary = 5.5

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax in axes.flat:
        ax.axvspan(0.5, phase_boundary, color="#a8dadc", alpha=0.15)
        ax.axvspan(phase_boundary, max(epochs) + 0.5, color="#f4a261", alpha=0.12)
        ax.axvline(best_epoch, color="#d62828", linestyle="--", linewidth=1.5)

    axes[0, 0].plot(epochs, train_acc, marker="o", label="Train")
    axes[0, 0].plot(epochs, val_acc, marker="s", label="Validation")
    axes[0, 0].set_title("Accuracy by Epoch")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(epochs, train_loss, marker="o", label="Train")
    axes[0, 1].plot(epochs, val_loss, marker="s", label="Validation")
    axes[0, 1].set_title("Loss by Epoch")
    axes[0, 1].set_ylabel("Loss")

    axes[1, 0].bar(epochs, [v - t for v, t in zip(val_acc, train_acc)], color="#6a4c93")
    axes[1, 0].set_title("Validation Minus Training Accuracy")
    axes[1, 0].set_ylabel("Gap (percentage points)")
    axes[1, 0].set_xlabel("Epoch")

    axes[1, 1].plot(epochs, val_acc, marker="o", color="#2a9d8f")
    axes[1, 1].annotate(f"Best epoch = {best_epoch}", (best_epoch, max(val_acc)), xytext=(best_epoch + 0.5, max(val_acc) - 2), arrowprops={"arrowstyle": "->"})
    axes[1, 1].set_title("Validation Accuracy with Phase Boundary")
    axes[1, 1].set_ylabel("Validation accuracy (%)")
    axes[1, 1].set_xlabel("Epoch")
    save_plot(PLOTS_DIR / "training_dynamics.png")


def pick_image_from_class(split_dir: Path, class_name: str) -> Path | None:
    class_dir = split_dir / class_name
    if not class_dir.is_dir():
        return None
    candidates = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    return candidates[len(candidates) // 2] if candidates else None


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def fit_image_to_frame(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    img = img.convert("RGB")
    src_w, src_h = img.size
    target_w, target_h = target_size

    if src_w * target_h == src_h * target_w:
        return img.resize((target_w, target_h), Image.LANCZOS)

    src_ratio = src_w / src_h
    target_ratio = target_w / target_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))

    return img.resize((target_w, target_h), Image.LANCZOS)


def make_labeled_gallery(items: list[tuple[Path, str, str]], out_path: Path, cols: int = 4, thumb_size: tuple[int, int] = (720, 560)) -> None:
    if not items:
        print(f"Skipped {out_path.relative_to(ROOT)}: no items")
        return

    rows = math.ceil(len(items) / cols)
    gutter = 26
    label_height = 110
    canvas_width = cols * thumb_size[0] + (cols + 1) * gutter
    canvas_height = rows * (thumb_size[1] + label_height) + (rows + 1) * gutter
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    title_font = load_font(34)
    body_font = load_font(26)

    for idx, (image_path, title, subtitle) in enumerate(items):
        r = idx // cols
        c = idx % cols
        x0 = gutter + c * (thumb_size[0] + gutter)
        y0 = gutter + r * (thumb_size[1] + label_height + gutter)
        tile = fit_image_to_frame(Image.open(image_path), thumb_size)
        x_img = x0
        y_img = y0
        canvas.paste(tile, (x_img, y_img))

        draw = ImageDraw.Draw(canvas)
        draw.rectangle((x_img, y_img, x_img + thumb_size[0], y_img + thumb_size[1]), outline=(210, 210, 210), width=3)
        text_y = y0 + thumb_size[1] + 12
        draw.text((x0 + 12, text_y), title, fill=(24, 24, 27), font=title_font)
        draw.text((x0 + 12, text_y + 42), subtitle, fill=(82, 82, 91), font=body_font)

    canvas.save(out_path, dpi=(600, 600))
    print(f"Saved {out_path.relative_to(ROOT)}")


def build_crop_gallery(counts: dict) -> None:
    items: list[tuple[Path, str, str]] = []
    split_dir = DATASET_DIR / "train"
    crop_to_class: dict[str, str] = {}
    for class_name in sorted(counts["train"]["per_class"]):
        crop = crop_of(class_name)
        if crop not in crop_to_class:
            crop_to_class[crop] = class_name

    for crop in sorted(crop_to_class):
        class_name = crop_to_class[crop]
        image_path = pick_image_from_class(split_dir, class_name)
        if image_path:
            pretty = pretty_label(class_name)
            subtitle = pretty.split(" / ", 1)[1] if " / " in pretty else pretty
            items.append((image_path, crop, subtitle))

    make_labeled_gallery(items, PLOTS_DIR / "crop_gallery.png", cols=4, thumb_size=(780, 600))


def build_hard_class_gallery(report: dict, counts: dict) -> None:
    labels = sorted(report["per_class_metrics"], key=lambda label: report["per_class_metrics"][label]["f1"])[:8]
    split_dir = DATASET_DIR / "val"
    items: list[tuple[Path, str, str]] = []
    for label in labels:
        actual_dir = resolve_dataset_class(counts["val"]["per_class"], split_dir, label)
        if not actual_dir:
            continue
        image_path = pick_image_from_class(split_dir, actual_dir)
        if not image_path:
            continue
        metric = report["per_class_metrics"][label]
        items.append(
            (
                image_path,
                pretty_label(label),
                f"F1={metric['f1'] * 100:.2f}% | support={metric['support']}",
            )
        )
    make_labeled_gallery(items, PLOTS_DIR / "hard_class_gallery.png", cols=4, thumb_size=(760, 560))


def build_misclassification_gallery() -> None:
    if not MISCLASS_SUMMARY_PATH.exists():
        return
    summary = json.loads(MISCLASS_SUMMARY_PATH.read_text(encoding="utf-8"))
    items: list[tuple[Path, str, str]] = []
    for row in summary:
        samples = row.get("sample_paths") or []
        if not samples:
            continue
        image_path = ROOT / samples[0]
        if image_path.exists():
            items.append(
                (
                    image_path,
                    f"True: {pretty_label(row['true'])}",
                    f"Pred: {pretty_label(row['pred'])} | count={row['count']}",
                )
            )
    make_labeled_gallery(items, PLOTS_DIR / "misclassification_gallery.png", cols=2, thumb_size=(1200, 900))


def build_case_gallery(counts: dict) -> None:
    split_dir = DATASET_DIR / "train"
    class_names = sorted(counts["train"]["per_class"])
    if not class_names:
        return

    target_count = min(16, len(class_names))
    indices = np.linspace(0, len(class_names) - 1, num=target_count, dtype=int)
    items: list[tuple[Path, str, str]] = []
    seen: set[str] = set()
    for idx in indices:
        class_name = class_names[int(idx)]
        if class_name in seen:
            continue
        seen.add(class_name)
        image_path = pick_image_from_class(split_dir, class_name)
        if not image_path:
            continue
        pretty = pretty_label(class_name)
        if " / " in pretty:
            crop, disease = pretty.split(" / ", 1)
        else:
            crop, disease = pretty, pretty
        items.append((image_path, crop, disease))

    make_labeled_gallery(items, PLOTS_DIR / "case_gallery.png", cols=4, thumb_size=(840, 620))


def draw_box(ax, xy, width, height, text, facecolor):
    box = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor=facecolor, edgecolor="#1f2937", linewidth=1.2)
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=11, fontweight="bold")


def draw_arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#374151"))


def plot_system_workflow() -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.05, 0.72), 0.18, 0.12, "dataset/train\n dataset/val\n dataset/test", "#d8f3dc")
    draw_box(ax, (0.31, 0.72), 0.18, 0.12, "train_model.py\n two-stage training", "#bee1e6")
    draw_box(ax, (0.57, 0.72), 0.18, 0.12, "models/*.keras\n class_indices.json", "#fde2e4")
    draw_box(ax, (0.79, 0.72), 0.16, 0.12, "predict.py\n app.py", "#fff1b6")

    draw_box(ax, (0.31, 0.47), 0.18, 0.12, "evaluate_model.py\n evaluation report", "#bee1e6")
    draw_box(ax, (0.57, 0.47), 0.18, 0.12, "scripts/generate_figures.py\n plots/*.png", "#bee1e6")
    draw_box(ax, (0.79, 0.47), 0.16, 0.12, "web control panel\n background jobs", "#fff1b6")

    draw_box(ax, (0.43, 0.2), 0.22, 0.12, "reports/main.tex\n report + appendices", "#e9d8fd")

    draw_arrow(ax, (0.23, 0.78), (0.31, 0.78))
    draw_arrow(ax, (0.49, 0.78), (0.57, 0.78))
    draw_arrow(ax, (0.75, 0.78), (0.79, 0.78))
    draw_arrow(ax, (0.66, 0.72), (0.66, 0.59))
    draw_arrow(ax, (0.57, 0.53), (0.49, 0.53))
    draw_arrow(ax, (0.75, 0.53), (0.79, 0.53))
    draw_arrow(ax, (0.40, 0.47), (0.50, 0.32))
    draw_arrow(ax, (0.66, 0.47), (0.58, 0.32))

    ax.text(0.5, 0.92, "Repository Workflow and Deliverable Path", ha="center", va="center", fontsize=16, fontweight="bold")
    save_plot(PLOTS_DIR / "system_workflow.png")


def main() -> None:
    counts = load_counts()
    report = load_report()
    class_order = load_class_order()
    history = load_history()

    plot_crop_distribution(counts)
    plot_class_imbalance(counts)
    plot_top_bottom_classes(counts)
    plot_per_class_f1(report)
    plot_precision_recall_support(report)
    plot_crop_level_f1(report)
    plot_top_confusions(report)
    plot_error_share_by_crop(report, class_order)
    plot_rice_confusion(report, class_order)
    plot_training_dynamics(history)
    build_crop_gallery(counts)
    build_hard_class_gallery(report, counts)
    build_misclassification_gallery()
    build_case_gallery(counts)
    plot_system_workflow()


if __name__ == "__main__":
    main()
