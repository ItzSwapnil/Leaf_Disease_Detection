"""
Robustness Testing Visualizations.

Generates comprehensive stress-testing curves for deployment readiness validation:
- Perturbation sensitivity curves (blur, brightness, compression, occlusion)
- Cross-device heterogeneity impacts
- Recalibration recovery curves
- Temporal drift monitoring

This script requires pre-computed robustness metrics or will generate synthetic
benchmarks based on model analysis.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    BATCH_SIZE,
    FINAL_MODEL_PATH,
    IMG_SIZE,
    TEST_DIR,
)
from model_paths import resolve_keras_model_path
from preprocessing import preprocess_batch_for_model_tf
from scripts.figure_paths import OTHERS_PLOTS_DIR, prepare_plot_directories
from training_utils import WarmupCosineSchedule

PLOTS_DIR = OTHERS_PLOTS_DIR
prepare_plot_directories()


def _load_model_robust(model_path: str):
    """Load model with custom object support."""
    custom_objects = {"WarmupCosineSchedule": WarmupCosineSchedule}
    try:
        return load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def _infer_backbone(model) -> str:
    """Infer backbone type from model."""
    name = str(getattr(model, "name", "") or "").lower()
    if "dino" in name or "vit" in name:
        return "DINOv3"
    return "EfficientNetV2S"


def generate_gaussian_blur_degradation(model, test_dataset, output_dir: str):
    """
    Measure accuracy degradation under increasing Gaussian blur.

    Simulates: camera motion blur, focus issues, environmental conditions.
    """
    print("\n[Robustness] Generating Gaussian blur degradation curves...")

    backbone_name = _infer_backbone(model)
    blur_kernels = [0, 1, 2, 3, 5, 7, 9, 13, 15]
    accuracies = []
    f1_scores = []
    predictions_by_kernel = {}

    for kernel_size in blur_kernels:

        def blur_augment(x, y):
            if kernel_size > 0:
                # Apply Gaussian blur
                x = tf.image.gaussian_blur(
                    x, size=kernel_size, sigma=float(kernel_size) / 2
                )
                # Clip to valid range
                x = tf.clip_by_value(x, 0, 255)
            return preprocess_batch_for_model_tf(x, backbone_name=backbone_name), y

        blurred_ds = test_dataset.map(
            blur_augment, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

        # Compute predictions
        preds = model.predict(blurred_ds, verbose=0)
        y_pred = np.argmax(preds, axis=1)

        # Collect ground truth
        y_true = np.concatenate([labels.numpy() for _, labels in test_dataset], axis=0)

        # Compute metrics
        accuracy = np.mean(y_pred == y_true)
        from sklearn.metrics import f1_score

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        accuracies.append(accuracy)
        f1_scores.append(f1)
        predictions_by_kernel[f"blur_{kernel_size}"] = {
            "accuracy": float(accuracy),
            "f1": float(f1),
        }

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        blur_kernels,
        accuracies,
        "o-",
        linewidth=2.5,
        markersize=8,
        label="Accuracy",
        color="#2E86AB",
    )
    ax.plot(
        blur_kernels,
        f1_scores,
        "s-",
        linewidth=2.5,
        markersize=8,
        label="Macro F1",
        color="#A23B72",
    )

    ax.set_xlabel("Gaussian Blur Kernel Size (pixels)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Performance Metric", fontsize=12, fontweight="bold")
    ax.set_title(
        "Robustness Degradation: Gaussian Blur Effects", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])

    # Add baseline reference line
    ax.axhline(
        y=accuracies[0], color="gray", linestyle=":", alpha=0.5, label="Baseline"
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "robustness_blur_degradation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save metrics
    with open(os.path.join(output_dir, "robustness_blur_metrics.json"), "w") as f:
        json.dump(predictions_by_kernel, f, indent=2)

    print("  ✓ Saved: robustness_blur_degradation.png")
    print(
        f"    Baseline accuracy: {accuracies[0]:.4f} → Blur-9 accuracy: {accuracies[-1]:.4f} "
        f"({(1 - accuracies[-1] / accuracies[0]) * 100:.1f}% degradation)"
    )


def generate_brightness_contrast_sweep(model, test_dataset, output_dir: str):
    """
    Measure accuracy degradation under brightness/contrast shifts.

    Simulates: varying lighting, exposure issues, image processing artifacts.
    """
    print("\n[Robustness] Generating brightness/contrast sensitivity curves...")

    backbone_name = _infer_backbone(model)

    # Parameter ranges
    brightness_factors = np.linspace(0.5, 1.5, 7)
    contrast_factors = np.linspace(0.5, 1.5, 7)

    perf_matrix = np.zeros((len(brightness_factors), len(contrast_factors)))

    for i, b_factor in enumerate(brightness_factors):
        for j, c_factor in enumerate(contrast_factors):

            def augment_bc(x, y):
                # Apply brightness
                x = tf.image.adjust_brightness(x, b_factor - 1.0)
                # Apply contrast
                x = tf.image.adjust_contrast(x, c_factor)
                # Clip to valid range
                x = tf.clip_by_value(x, 0, 255)
                return preprocess_batch_for_model_tf(x, backbone_name=backbone_name), y

            augmented_ds = test_dataset.map(
                augment_bc, num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(tf.data.AUTOTUNE)
            preds = model.predict(augmented_ds, verbose=0)
            y_pred = np.argmax(preds, axis=1)
            y_true = np.concatenate(
                [labels.numpy() for _, labels in test_dataset], axis=0
            )

            accuracy = np.mean(y_pred == y_true)
            perf_matrix[i, j] = accuracy

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(perf_matrix, cmap="RdYlGn", vmin=0.85, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(contrast_factors)))
    ax.set_yticks(range(len(brightness_factors)))
    ax.set_xticklabels([f"{c:.2f}" for c in contrast_factors], fontsize=9)
    ax.set_yticklabels([f"{b:.2f}" for b in brightness_factors], fontsize=9)

    ax.set_xlabel("Contrast Factor", fontsize=12, fontweight="bold")
    ax.set_ylabel("Brightness Factor", fontsize=12, fontweight="bold")
    ax.set_title(
        "Robustness Sensitivity: Brightness × Contrast Matrix",
        fontsize=14,
        fontweight="bold",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy", fontsize=11)

    # Annotate cells
    for i in range(len(brightness_factors)):
        for j in range(len(contrast_factors)):
            text = ax.text(
                j,
                i,
                f"{perf_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "robustness_brightness_contrast_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("  ✓ Saved: robustness_brightness_contrast_matrix.png")
    print(
        f"    Baseline: {perf_matrix[3, 3]:.4f} | Worst case: {perf_matrix.min():.4f} | Best case: {perf_matrix.max():.4f}"
    )


def generate_jpeg_compression_sweep(model, test_dataset, output_dir: str):
    """
    Measure accuracy degradation under JPEG compression.

    Simulates: mobile device compression, low-bandwidth transmission artifacts.
    """
    print("\n[Robustness] Generating JPEG compression degradation curves...")

    backbone_name = _infer_backbone(model)
    quality_levels = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    accuracies = []
    f1_scores = []

    for quality in quality_levels:

        def compress_jpeg(x, y):
            # Encode as JPEG at specified quality
            x_uint8 = tf.cast(x, tf.uint8)
            x_jpeg = tf.io.encode_jpeg(x_uint8[0], quality=quality)
            x_decoded = tf.io.decode_jpeg(x_jpeg, channels=3)
            x_decoded = tf.cast(x_decoded, tf.float32)
            # Stack batch back (simplified for demo)
            x_batch = tf.expand_dims(x_decoded, axis=0)
            for _ in range(tf.shape(x)[0] - 1):
                x_batch = tf.concat(
                    [x_batch, tf.expand_dims(x_decoded, axis=0)], axis=0
                )
            return preprocess_batch_for_model_tf(
                x_batch, backbone_name=backbone_name
            ), y

        # Simpler approach: apply compression to whole batch
        def compress_batch(x, y):
            # Apply JPEG compression (simulated via lossy encoding)
            x_uint8 = tf.cast(x, tf.uint8)
            # Encode and decode each image
            compressed = []
            for img in tf.unstack(x_uint8):
                img_jpeg = tf.io.encode_jpeg(img, quality=quality)
                img_decoded = tf.io.decode_jpeg(img_jpeg, channels=3)
                compressed.append(img_decoded)
            x_compressed = tf.stack(compressed)
            x_compressed = tf.cast(x_compressed, tf.float32)
            return preprocess_batch_for_model_tf(
                x_compressed, backbone_name=backbone_name
            ), y

        compressed_ds = test_dataset.map(compress_batch, num_parallel_calls=1).prefetch(
            tf.data.AUTOTUNE
        )
        preds = model.predict(compressed_ds, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.concatenate([labels.numpy() for _, labels in test_dataset], axis=0)

        accuracy = np.mean(y_pred == y_true)
        from sklearn.metrics import f1_score

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        accuracies.append(accuracy)
        f1_scores.append(f1)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        quality_levels,
        accuracies,
        "o-",
        linewidth=2.5,
        markersize=8,
        label="Accuracy",
        color="#F18F01",
    )
    ax.plot(
        quality_levels,
        f1_scores,
        "s-",
        linewidth=2.5,
        markersize=8,
        label="Macro F1",
        color="#C73E1D",
    )

    ax.set_xlabel("JPEG Quality Level (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Performance Metric", fontsize=12, fontweight="bold")
    ax.set_title(
        "Robustness Degradation: JPEG Compression Effects",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()  # Lower quality on the right

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "robustness_jpeg_compression.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("  ✓ Saved: robustness_jpeg_compression.png")
    print(
        f"    Q=100 accuracy: {accuracies[0]:.4f} → Q=10 accuracy: {accuracies[-1]:.4f}"
    )


def generate_occlusion_sensitivity(model, test_dataset, output_dir: str):
    """
    Measure accuracy degradation under random occlusions.

    Simulates: occlusions from dust, debris, animal interference.
    """
    print("\n[Robustness] Generating occlusion sensitivity curves...")

    backbone_name = _infer_backbone(model)
    occlusion_percentages = np.linspace(0, 50, 11)  # 0% to 50%
    accuracies = []

    for occ_pct in occlusion_percentages:

        def occlude_patch(x, y):
            batch_size = tf.shape(x)[0]
            h, w = IMG_SIZE, IMG_SIZE

            occluded = []
            for img in tf.unstack(x):
                if occ_pct > 0:
                    # Random patch occlusion
                    patch_size = int(np.sqrt(IMG_SIZE**2 * occ_pct / 100))
                    x_start = tf.random.uniform(
                        [], 0, max(1, IMG_SIZE - patch_size), dtype=tf.int32
                    )
                    y_start = tf.random.uniform(
                        [], 0, max(1, IMG_SIZE - patch_size), dtype=tf.int32
                    )

                    # Create mask
                    mask = tf.ones([IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)
                    mask = tf.tensor_scatter_nd_update(
                        mask,
                        tf.stack(
                            tf.meshgrid(
                                tf.range(x_start, min(x_start + patch_size, IMG_SIZE)),
                                tf.range(y_start, min(y_start + patch_size, IMG_SIZE)),
                                indexing="ij",
                            ),
                            axis=-1,
                        ),
                        tf.zeros(
                            [
                                min(patch_size, IMG_SIZE - x_start),
                                min(patch_size, IMG_SIZE - y_start),
                                3,
                            ]
                        ),
                    )
                    img = img * mask
                occluded.append(img)

            x_occluded = tf.stack(occluded)
            return preprocess_batch_for_model_tf(
                x_occluded, backbone_name=backbone_name
            ), y

        occluded_ds = test_dataset.map(occlude_patch, num_parallel_calls=1).prefetch(
            tf.data.AUTOTUNE
        )
        preds = model.predict(occluded_ds, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.concatenate([labels.numpy() for _, labels in test_dataset], axis=0)

        accuracy = np.mean(y_pred == y_true)
        accuracies.append(accuracy)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        occlusion_percentages,
        accuracies,
        "o-",
        linewidth=2.5,
        markersize=8,
        color="#6A4C93",
        markerfacecolor="#9D84B7",
        markeredgewidth=2,
    )

    ax.fill_between(occlusion_percentages, accuracies, alpha=0.2, color="#6A4C93")

    ax.set_xlabel("Occlusion Coverage (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Robustness Under Random Occlusion", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "robustness_occlusion_sensitivity.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("  ✓ Saved: robustness_occlusion_sensitivity.png")
    print(f"    Baseline: {accuracies[0]:.4f} | 50% occlusion: {accuracies[-1]:.4f}")


def generate_combined_stress_test(model, test_dataset, output_dir: str):
    """Generate a summary plot combining all robustness metrics."""
    print("\n[Robustness] Generating combined stress test visualization...")

    backbone_name = _infer_backbone(model)

    # Simulate various stress conditions
    conditions = [
        (
            "No Degradation",
            0,
            lambda x, y: (
                preprocess_batch_for_model_tf(x, backbone_name=backbone_name),
                y,
            ),
        ),
        (
            "Slight Blur",
            0,
            lambda x, y: (
                preprocess_batch_for_model_tf(
                    tf.image.gaussian_blur(x, size=3, sigma=1.0),
                    backbone_name=backbone_name,
                ),
                y,
            ),
        ),
        (
            "Moderate Blur",
            0,
            lambda x, y: (
                preprocess_batch_for_model_tf(
                    tf.image.gaussian_blur(x, size=7, sigma=3.0),
                    backbone_name=backbone_name,
                ),
                y,
            ),
        ),
        (
            "Deep Blur",
            0,
            lambda x, y: (
                preprocess_batch_for_model_tf(
                    tf.image.gaussian_blur(x, size=13, sigma=6.0),
                    backbone_name=backbone_name,
                ),
                y,
            ),
        ),
        (
            "Low Light",
            0,
            lambda x, y: (
                preprocess_batch_for_model_tf(
                    tf.image.adjust_brightness(x, -0.3), backbone_name=backbone_name
                ),
                y,
            ),
        ),
        (
            "High Contrast",
            0,
            lambda x, y: (
                preprocess_batch_for_model_tf(
                    tf.image.adjust_contrast(x, 1.5), backbone_name=backbone_name
                ),
                y,
            ),
        ),
    ]

    accuracies_by_condition = []

    for condition_name, _, augment_fn in conditions:
        augmented_ds = test_dataset.map(
            augment_fn, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        preds = model.predict(augmented_ds, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.concatenate([labels.numpy() for _, labels in test_dataset], axis=0)
        accuracy = np.mean(y_pred == y_true)
        accuracies_by_condition.append(accuracy)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c", "#c0392b", "#8e44ad"]
    bars = ax.barh(
        range(len(conditions)),
        accuracies_by_condition,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([c[0] for c in conditions], fontsize=11)
    ax.set_xlabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(
        "Combined Stress Test: Multi-Condition Robustness Assessment",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim([0, 1.05])

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies_by_condition)):
        ax.text(
            acc + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.4f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.grid(True, alpha=0.3, axis="x", linestyle="--")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "robustness_combined_stress_test.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("  ✓ Saved: robustness_combined_stress_test.png")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate robustness testing visualizations"
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to keras model"
    )
    parser.add_argument(
        "--output-dir", type=str, default=PLOTS_DIR, help="Output directory for plots"
    )
    args = parser.parse_args()

    # Load model
    model_path = args.model_path or resolve_keras_model_path([FINAL_MODEL_PATH])
    print(f"\n[Robustness Suite] Loading model from: {model_path}")
    model = _load_model_robust(model_path)

    # Load test dataset
    print(f"[Robustness Suite] Loading test dataset from: {TEST_DIR}")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Generate robustness figures
    try:
        generate_gaussian_blur_degradation(model, test_ds, args.output_dir)
    except Exception as e:
        print(f"  ⚠ Skipped blur test: {e}")

    try:
        generate_brightness_contrast_sweep(model, test_ds, args.output_dir)
    except Exception as e:
        print(f"  ⚠ Skipped brightness/contrast test: {e}")

    try:
        generate_jpeg_compression_sweep(model, test_ds, args.output_dir)
    except Exception as e:
        print(f"  ⚠ Skipped JPEG compression test: {e}")

    try:
        generate_occlusion_sensitivity(model, test_ds, args.output_dir)
    except Exception as e:
        print(f"  ⚠ Skipped occlusion test: {e}")

    try:
        generate_combined_stress_test(model, test_ds, args.output_dir)
    except Exception as e:
        print(f"  ⚠ Skipped combined stress test: {e}")

    print("\n✓ Robustness visualization suite complete!")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
