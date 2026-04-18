"""
Preprocessing Documentation and Data Quality Visualizations.

Generates:
- Augmentation effect comparisons
- Class imbalance remediation analysis
- Data quality scoring distributions
- Pipeline flowchart (conceptual)
- Train/val/test distribution comparisons
"""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    IMG_SIZE,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)
from scripts.figure_paths import OTHERS_PLOTS_DIR, prepare_plot_directories

PLOTS_DIR = OTHERS_PLOTS_DIR
prepare_plot_directories()


def generate_class_imbalance_analysis(output_dir: str = None):
    """
    Analyze and visualize class imbalance patterns.
    """
    print("\n[Preprocessing] Analyzing class imbalance...")

    if output_dir is None:
        output_dir = PLOTS_DIR

    # Count images per class in each split
    splits = {"Train": TRAIN_DIR, "Val": VAL_DIR, "Test": TEST_DIR}
    class_counts = {}

    for split_name, split_dir in splits.items():
        class_counts[split_name] = {}
        for class_dir in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_dir)
            if os.path.isdir(class_path):
                count = len(
                    [
                        f
                        for f in os.listdir(class_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                    ]
                )
                class_counts[split_name][class_dir] = count

    # Get all unique classes
    all_classes = sorted(
        set(cls for split in class_counts.values() for cls in split.keys())
    )

    # Create data for plotting
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. Per-split distribution box plot
    ax = axes[0, 0]
    split_data = []
    split_labels = []
    for split_name in ["Train", "Val", "Test"]:
        counts = list(class_counts[split_name].values())
        split_data.append(counts)
        split_labels.append(split_name)

    bp = ax.boxplot(split_data, tick_labels=split_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#3498db", "#2ecc71", "#f39c12"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Images per Class", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution Balance by Split", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Imbalance ratio visualization
    ax = axes[0, 1]
    train_counts = np.array(list(class_counts["Train"].values()))
    imbalance_ratio = train_counts.max() / (train_counts.min() + 1e-10)

    sorted_indices = np.argsort(train_counts)[::-1]
    sorted_counts = train_counts[sorted_indices]
    sorted_classes = [all_classes[i] for i in sorted_indices]

    # Top and bottom classes
    top_n = 10
    display_classes = sorted_classes[:top_n] + sorted_classes[-top_n:]
    display_counts = np.concatenate([sorted_counts[:top_n], sorted_counts[-top_n:]])

    class_names = [
        c.replace("___", " - ").replace("_", " ")[:25] for c in display_classes
    ]
    colors_bar = ["#e74c3c"] * top_n + ["#95a5a6"] * top_n

    y_pos = np.arange(len(display_classes))
    ax.barh(y_pos, display_counts, color=colors_bar, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Sample Count", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Class Imbalance: Top & Bottom 10 Classes\n(Imbalance Ratio: {imbalance_ratio:.1f}:1)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")

    # 3. Cumulative distribution
    ax = axes[1, 0]
    for split_name in ["Train", "Val", "Test"]:
        counts = sorted(class_counts[split_name].values(), reverse=True)
        cumsum = np.cumsum(counts)
        cumsum_pct = cumsum / cumsum[-1] * 100
        ax.plot(
            range(1, len(cumsum) + 1),
            cumsum_pct,
            marker="o",
            label=split_name,
            linewidth=2,
            markersize=4,
            alpha=0.7,
        )

    ax.set_xlabel("Number of Classes (ranked by count)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative % of Samples", fontsize=11, fontweight="bold")
    ax.set_title(
        "Cumulative Sample Distribution (Long-Tail Analysis)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)

    # 4. Class count statistics
    ax = axes[1, 1]
    ax.axis("off")

    stats_text = "Class Distribution Statistics (Train Set)\n" + "=" * 50 + "\n"
    for split_name in ["Train", "Val", "Test"]:
        counts = np.array(list(class_counts[split_name].values()))
        stats_text += f"\n{split_name} Set:\n"
        stats_text += f"  Total classes: {len(counts)}\n"
        stats_text += f"  Total samples: {counts.sum()}\n"
        stats_text += f"  Mean per class: {counts.mean():.1f}\n"
        stats_text += f"  Median per class: {np.median(counts):.1f}\n"
        stats_text += f"  Min: {counts.min()}, Max: {counts.max()}\n"
        stats_text += f"  Std Dev: {counts.std():.1f}\n"
        stats_text += (
            f"  Coefficient of Variation: {counts.std() / counts.mean():.3f}\n"
        )

    ax.text(
        0.1,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "preprocessing_class_imbalance_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("  ✓ Saved: preprocessing_class_imbalance_analysis.png")
    print(f"    Imbalance Ratio: {imbalance_ratio:.1f}:1")


def generate_augmentation_effect_comparison(output_dir: str = None):
    """
    Show impact of augmentation strategies on sample diversity.
    """
    print("\n[Preprocessing] Generating augmentation effect visualizations...")

    if output_dir is None:
        output_dir = PLOTS_DIR

    # Load a sample image
    sample_class_path = os.path.join(TRAIN_DIR, os.listdir(TRAIN_DIR)[0])
    sample_images = [
        f
        for f in os.listdir(sample_class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not sample_images:
        print("  ⚠ No sample images found")
        return

    sample_path = os.path.join(sample_class_path, sample_images[0])
    image = Image.open(sample_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image, dtype=np.float32)
    image_array = tf.expand_dims(image_array, axis=0)

    # Define augmentation    operations
    augmentations = {
        "Original": lambda x: x,
        "Random Flip": lambda x: tf.image.random_flip_left_right(x),
        "Random Rotation": lambda x: tf.image.rot90(
            x, k=tf.random.uniform([], 0, 4, dtype=tf.int32)
        ),
        "Brightness": lambda x: tf.image.adjust_brightness(
            x, tf.random.uniform([], -0.2, 0.2)
        ),
        "Contrast": lambda x: tf.image.adjust_contrast(
            x, tf.random.uniform([], 0.8, 1.2)
        ),
        "JPEG Quality": lambda x: _apply_jpeg_compression(x),
        # 'Gaussian Blur': lambda x: tf.image.gaussian_blur(x, size=5, sigma=tf.random.uniform([], 0.5, 2.0)),
        "Random Crop": lambda x: _random_crop(x, IMG_SIZE),
    }

    # Generate augmented samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (aug_name, aug_fn) in enumerate(augmentations.items()):
        tf.random.set_seed(42)  # For reproducibility

        augmented = aug_fn(image_array)
        augmented = tf.squeeze(augmented, axis=0)
        augmented = tf.cast(tf.clip_by_value(augmented, 0, 255), tf.uint8)

        ax = axes[idx]
        ax.imshow(augmented.numpy())
        ax.set_title(aug_name, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        "Augmentation Strategies: Sample Transformations",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "preprocessing_augmentation_effects.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("  ✓ Saved: preprocessing_augmentation_effects.png")


def _apply_jpeg_compression(x, quality=75):
    """Apply JPEG compression to single image."""
    x_uint8 = tf.cast(tf.clip_by_value(x[0], 0, 255), tf.uint8)
    x_jpeg = tf.io.encode_jpeg(x_uint8, quality=quality)
    x_decoded = tf.io.decode_jpeg(x_jpeg, channels=3)
    return tf.expand_dims(tf.cast(x_decoded, tf.float32), axis=0)


def _random_crop(x, size, min_scale=0.8):
    """Random crop with min scale."""
    x_uint8 = tf.cast(tf.clip_by_value(x[0], 0, 255), tf.uint8)
    crop_size = int(size * tf.random.uniform([], min_scale, 1.0))
    x_cropped = tf.image.random_crop(
        tf.expand_dims(x_uint8, 0), [1, crop_size, crop_size, 3]
    )
    x_resized = tf.image.resize(x_cropped, [size, size])
    return tf.cast(x_resized, tf.float32)


def generate_split_distribution_comparison(output_dir: str = None):
    """
    Compare class distributions across train/val/test splits.
    """
    print("\n[Preprocessing] Generating split distribution comparison...")

    if output_dir is None:
        output_dir = PLOTS_DIR

    # Count classes in each split
    splits = {"Train": TRAIN_DIR, "Val": VAL_DIR, "Test": TEST_DIR}
    all_classes_list = sorted(os.listdir(TRAIN_DIR))

    fig, ax = plt.subplots(figsize=(18, 10))

    x = np.arange(len(all_classes_list))
    width = 0.25

    for idx, (split_name, split_dir) in enumerate(splits.items()):
        counts = []
        for class_dir in all_classes_list:
            class_path = os.path.join(split_dir, class_dir)
            if os.path.isdir(class_path):
                count = len(
                    [
                        f
                        for f in os.listdir(class_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )
                counts.append(count)
            else:
                counts.append(0)

        color_map = {"Train": "#3498db", "Val": "#2ecc71", "Test": "#f39c12"}
        ax.bar(
            x + idx * width,
            counts,
            width,
            label=split_name,
            color=color_map[split_name],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

    class_display_names = [
        c.replace("___", "\n").replace("_", " ")[:20] for c in all_classes_list
    ]
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_display_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    ax.set_title(
        "Dataset Distribution: Train / Val / Test Comparison (Stratified)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "preprocessing_split_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("  ✓ Saved: preprocessing_split_distribution.png")


def generate_image_quality_analysis(output_dir: str = None):
    """
    Analyze image quality metrics (brightness, contrast, etc.)
    """
    print("\n[Preprocessing] Analyzing image quality metrics...")

    if output_dir is None:
        output_dir = PLOTS_DIR

    quality_metrics = {
        "brightness": [],
        "contrast": [],
        "size": [],
    }

    # Sample images from train set
    class_dirs = os.listdir(TRAIN_DIR)[:5]  # Sample from first 5 classes
    image_count = 0
    max_images = 1000

    for class_dir in class_dirs:
        class_path = os.path.join(TRAIN_DIR, class_dir)
        if not os.path.isdir(class_path):
            continue

        for image_file in os.listdir(class_path):
            if image_count >= max_images:
                break

            try:
                image_path = os.path.join(class_path, image_file)
                if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img = Image.open(image_path).convert("RGB")
                img_array = np.array(img, dtype=np.float32)

                # Compute quality metrics
                brightness = np.mean(img_array) / 255
                contrast = np.std(img_array) / 128
                size = img.size

                quality_metrics["brightness"].append(brightness)
                quality_metrics["contrast"].append(contrast)
                quality_metrics["size"].append(max(size))

                image_count += 1
            except Exception:
                continue

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Brightness distribution
    ax = axes[0]
    ax.hist(
        quality_metrics["brightness"],
        bins=50,
        color="#3498db",
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xlabel("Brightness (normalized)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Image Brightness Distribution\n(n={len(quality_metrics['brightness'])})",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(
        np.mean(quality_metrics["brightness"]),
        color="red",
        linestyle="--",
        linewidth=2,
        label="Mean",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Contrast distribution
    ax = axes[1]
    ax.hist(
        quality_metrics["contrast"],
        bins=50,
        color="#2ecc71",
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xlabel("Contrast (normalized)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Image Contrast Distribution\n(n={len(quality_metrics['contrast'])})",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(
        np.mean(quality_metrics["contrast"]),
        color="red",
        linestyle="--",
        linewidth=2,
        label="Mean",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Image size distribution
    ax = axes[2]
    ax.hist(
        quality_metrics["size"], bins=30, color="#f39c12", alpha=0.7, edgecolor="black"
    )
    ax.set_xlabel("Image Size (pixels, max dimension)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Image Size Distribution\n(n={len(quality_metrics['size'])})",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(
        np.mean(quality_metrics["size"]),
        color="red",
        linestyle="--",
        linewidth=2,
        label="Mean",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "preprocessing_image_quality_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("  ✓ Saved: preprocessing_image_quality_analysis.png")
    print(f"    Mean brightness: {np.mean(quality_metrics['brightness']):.3f}")
    print(f"    Mean contrast: {np.mean(quality_metrics['contrast']):.3f}")
    print(f"    Mean image size: {np.mean(quality_metrics['size']):.0f}px")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate preprocessing visualizations"
    )
    parser.add_argument(
        "--output-dir", type=str, default=PLOTS_DIR, help="Output directory"
    )
    args = parser.parse_args()

    print("\n[Preprocessing Suite] Starting visualization generation...")

    try:
        generate_class_imbalance_analysis(args.output_dir)
    except Exception as e:
        print(f"  ⚠ Skipped class imbalance: {e}")

    try:
        generate_augmentation_effect_comparison(args.output_dir)
    except Exception as e:
        print(f"  ⚠ Skipped augmentation effects: {e}")

    try:
        generate_split_distribution_comparison(args.output_dir)
    except Exception as e:
        print(f"  ⚠ Skipped split distribution: {e}")

    try:
        generate_image_quality_analysis(args.output_dir)
    except Exception as e:
        print(f"  ⚠ Skipped image quality: {e}")

    print("\n✓ Preprocessing suite complete!")


if __name__ == "__main__":
    main()
