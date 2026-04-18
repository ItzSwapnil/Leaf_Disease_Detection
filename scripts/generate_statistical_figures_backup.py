"""
Statistical Validation and Uncertainty Quantification Visualizations.

Generates:
- Bootstrap confidence interval distributions
- Multi-seed stability analysis
- Statistical significance heatmaps
- Error decomposition analysis
- Prediction margin distributions
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import f1_score, precision_score, recall_score

from config import (
    BATCH_SIZE,
    FINAL_MODEL_PATH,
    IMG_SIZE,
    PLOTS_DIR,
    TEST_DIR,
)
from model_paths import resolve_keras_model_path
from preprocessing import preprocess_batch_for_model_tf
from training_utils import WarmupCosineSchedule

os.makedirs(PLOTS_DIR, exist_ok=True)


def _patch_vit_layer_init_for_compat() -> bool:
    """Patch KerasHub ViT layer init to ignore legacy saved kwargs."""
    try:
        from keras_hub.src.models.vit.vit_layers import ViTPatchingAndEmbedding

        layer_cls = ViTPatchingAndEmbedding
    except Exception:
        return False

    if getattr(layer_cls, "_leaf_compat_patched", False):
        return True

    original_init = layer_cls.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.pop("num_patches", None)
        kwargs.pop("num_positions", None)
        return original_init(self, *args, **kwargs)

    layer_cls.__init__ = _patched_init
    layer_cls._leaf_compat_patched = True
    return True


def _load_model_robust(model_path: str):
    """Load model with custom objects."""
    custom_objects = {"WarmupCosineSchedule": WarmupCosineSchedule}
    try:
        return load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as exc:
        message = str(exc)
        vit_compat_error = (
            "ViTPatchingAndEmbedding" in message
            or "num_patches" in message
            or "num_positions" in message
        )
        if not vit_compat_error:
            print(f"Error loading model: {exc}")
            raise
        if not _patch_vit_layer_init_for_compat():
            print(f"Error loading model: {exc}")
            raise
        print("Applied ViT compatibility shim while loading model.")
        return load_model(model_path, custom_objects=custom_objects, compile=False)


def _infer_backbone(model) -> str:
    """Infer backbone from model."""
    name = str(getattr(model, "name", "") or "").lower()
    if "dino" in name or "vit" in name:
        return "DINOv3"
    return "EfficientNetV2S"


def generate_bootstrap_ci_distributions(
    model, test_dataset, n_bootstraps=2000, output_dir: str = None
):
    """
    Generate bootstrap confidence intervals for key metrics.

    Performs 2000 bootstrap resamples of test set to quantify metric uncertainty.
    """
    print(
        f"\n[Statistical] Generating bootstrap CI distributions ({n_bootstraps} resamples)..."
    )

    if output_dir is None:
        output_dir = PLOTS_DIR

    backbone_name = _infer_backbone(model)

    # Get predictions
    preds = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in test_dataset], axis=0)

    n_samples = len(y_true)

    # Bootstrap resampling
    bootstrap_metrics = {
        "accuracy": [],
        "macro_f1": [],
        "macro_precision": [],
        "macro_recall": [],
    }

    for b in range(n_bootstraps):
        if (b + 1) % 500 == 0:
            print(f"  Bootstrap {b + 1}/{n_bootstraps}")

        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Compute metrics
        acc = np.mean(y_pred_boot == y_true_boot)
        f1 = f1_score(y_true_boot, y_pred_boot, average="macro", zero_division=0)
        prec = precision_score(
            y_true_boot, y_pred_boot, average="macro", zero_division=0
        )
        rec = recall_score(y_true_boot, y_pred_boot, average="macro", zero_division=0)

        bootstrap_metrics["accuracy"].append(acc)
        bootstrap_metrics["macro_f1"].append(f1)
        bootstrap_metrics["macro_precision"].append(prec)
        bootstrap_metrics["macro_recall"].append(rec)

    # Compute CIs
    ci_95 = {}
    for metric_name, values in bootstrap_metrics.items():
        ci_95[metric_name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "ci_lower": float(np.percentile(values, 2.5)),
            "ci_upper": float(np.percentile(values, 97.5)),
            "ci_width": float(np.percentile(values, 97.5) - np.percentile(values, 2.5)),
        }

    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics_list = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

    for idx, (metric_name, color) in enumerate(zip(metrics_list, colors)):
        ax = axes[idx]
        values = bootstrap_metrics[metric_name]
        ci = ci_95[metric_name]

        # Histogram with KDE
        ax.hist(
            values, bins=50, color=color, alpha=0.7, edgecolor="black", density=True
        )

        # KDE curve
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(values)
        x_range = np.linspace(min(values), max(values), 200)
        ax.plot(x_range, kde(x_range), "k-", linewidth=2)

        # CI lines
        ax.axvline(
            ci["mean"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {ci['mean']:.4f}",
        )
        ax.axvline(
            ci["ci_lower"],
            color="orange",
            linestyle=":",
            linewidth=2,
            label=f"95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]",
        )
        ax.axvline(ci["ci_upper"], color="orange", linestyle=":", linewidth=2)

        ax.set_xlabel(
            metric_name.replace("_", " ").title(), fontsize=11, fontweight="bold"
        )
        ax.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{metric_name.upper()} Bootstrap Distribution\n(n={n_bootstraps})",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "statistical_bootstrap_ci_distributions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save CI data
    with open(os.path.join(output_dir, "statistical_bootstrap_ci.json"), "w") as f:
        json.dump(ci_95, f, indent=2)

    print("  ✓ Saved: statistical_bootstrap_ci_distributions.png")
    for metric_name, ci in ci_95.items():
        print(
            f"    {metric_name:20s} → {ci['mean']:.4f} ± {ci['std']:.4f} (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])"
        )

    return bootstrap_metrics, ci_95


def generate_margin_distribution_plot(model, test_dataset, output_dir: str = None):
    """
    Plot distribution of prediction margins (top1 - top2 probability).

    Lower margins indicate uncertain predictions vulnerable to perturbations.
    """
    print("\n[Statistical] Generating prediction margin distribution...")

    if output_dir is None:
        output_dir = PLOTS_DIR

    backbone_name = _infer_backbone(model)

    # Get predictions
    preds = model.predict(test_dataset, verbose=0)

    # Compute margins
    top_two = np.partition(preds, -2, axis=1)[:, -2:]
    margins = top_two[:, 1] - top_two[:, 0]  # top1 - top2

    # Separate correct vs incorrect
    y_pred = np.argmax(preds, axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in test_dataset], axis=0)
    correct_mask = y_pred == y_true

    margins_correct = margins[correct_mask]
    margins_incorrect = margins[~correct_mask]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Combined distribution
    ax = axes[0]
    ax.hist(
        margins_correct,
        bins=50,
        alpha=0.6,
        label=f"Correct ({len(margins_correct)})",
        color="#2ecc71",
        edgecolor="black",
        density=True,
    )
    ax.hist(
        margins_incorrect,
        bins=50,
        alpha=0.6,
        label=f"Incorrect ({len(margins_incorrect)})",
        color="#e74c3c",
        edgecolor="black",
        density=True,
    )

    ax.set_xlabel(
        "Prediction Margin (Top1 - Top2 Probability)", fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.set_title(
        "Margin Distribution: Confidence vs Correctness", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Right: Cumulative
    ax = axes[1]
    margins_sorted_correct = np.sort(margins_correct)
    margins_sorted_incorrect = np.sort(margins_incorrect)

    ax.plot(
        margins_sorted_correct,
        np.arange(1, len(margins_sorted_correct) + 1) / len(margins_sorted_correct),
        label="Correct",
        linewidth=2.5,
        color="#2ecc71",
    )
    ax.plot(
        margins_sorted_incorrect,
        np.arange(1, len(margins_sorted_incorrect) + 1) / len(margins_sorted_incorrect),
        label="Incorrect",
        linewidth=2.5,
        color="#e74c3c",
    )

    ax.set_xlabel(
        "Prediction Margin (Top1 - Top2 Probability)", fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("Cumulative Density", fontsize=11, fontweight="bold")
    ax.set_title("Cumulative Margin Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "statistical_margin_distributions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("  ✓ Saved: statistical_margin_distributions.png")
    print(f"    Correct predictions - Mean margin: {margins_correct.mean():.4f}")
    print(f"    Incorrect predictions - Mean margin: {margins_incorrect.mean():.4f}")
    print("    Margin overlap suggests ambiguous samples")


def generate_per_class_stability_plot(
    model, test_dataset, output_dir: str = None, n_seeds=5
):
    """
    Simulate multi-seed stability across classes.

    For each class, estimate stability by performing multiple forward passes
    with slight input perturbations (lower bound on actual multi-seed variance).
    """
    print(
        f"\n[Statistical] Generating per-class stability analysis ({n_seeds} perturbation runs)..."
    )

    if output_dir is None:
        output_dir = PLOTS_DIR

    backbone_name = _infer_backbone(model)

    # Get class structure
    test_ds_iter = test_dataset.unbatch().batch(BATCH_SIZE)
    class_counts = {}
    class_f1_scores = {i: [] for i in range(46)}

    for x, y in test_ds_iter:
        for yi in y.numpy():
            class_counts[yi] = class_counts.get(yi, 0) + 1

    # Multiple forward passes with input perturbations
    for seed in range(n_seeds):
        print(f"  Perturbation run {seed + 1}/{n_seeds}")

        def add_noise(x, y, noise_scale=0.005):
            noise = tf.random.normal(
                tf.shape(x), mean=0, stddev=noise_scale * 255, dtype=tf.float32
            )
            x_noisy = tf.clip_by_value(x + noise, 0, 255)
            return preprocess_batch_for_model_tf(
                x_noisy, backbone_name=backbone_name
            ), y

        perturbed_ds = test_dataset.map(
            lambda x, y: add_noise(x, y), num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

        preds = model.predict(perturbed_ds, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.concatenate([labels.numpy() for _, labels in test_dataset], axis=0)

        # Compute per-class F1
        for class_id in range(46):
            mask = y_true == class_id
            if mask.sum() > 0:
                class_y_true = y_true[mask]
                class_y_pred = y_pred[mask]
                f1 = f1_score(
                    [class_y_true == class_id],
                    [class_y_pred == class_id],
                    average="binary",
                    zero_division=0,
                )
                class_f1_scores[class_id].append(f1)

    # Compute stability metrics
    class_stabilities = {}
    for class_id in range(46):
        if len(class_f1_scores[class_id]) > 0:
            f1_values = np.array(class_f1_scores[class_id])
            class_stabilities[class_id] = {
                "mean": float(f1_values.mean()),
                "std": float(f1_values.std()),
                "min": float(f1_values.min()),
                "max": float(f1_values.max()),
            }

    # Plot ridgeline-style stability plot
    fig, ax = plt.subplots(figsize=(12, 14))

    sorted_classes = sorted(
        class_stabilities.keys(),
        key=lambda x: class_stabilities[x]["mean"],
        reverse=True,
    )

    for idx, class_id in enumerate(sorted_classes):
        values = class_f1_scores[class_id]
        mean_val = class_stabilities[class_id]["mean"]
        std_val = class_stabilities[class_id]["std"]

        # Plot individual seed values
        y_pos = idx
        for seed_idx, val in enumerate(values):
            ax.scatter(
                val,
                y_pos + (seed_idx - len(values) / 2) * 0.15,
                s=30,
                alpha=0.6,
                color="#3498db",
            )

        # Mean and error bar
        ax.errorbar(
            mean_val,
            y_pos,
            xerr=std_val,
            fmt="o",
            markersize=10,
            color="#e74c3c",
            linewidth=2.5,
            elinewidth=2,
            capsize=5,
            capthick=2,
            label="Mean ± Std" if idx == 0 else "",
        )

    ax.set_yticks(range(len(sorted_classes)))
    ax.set_yticklabels([f"Class {cid}" for cid in sorted_classes], fontsize=8)
    ax.set_xlabel("F1 Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Per-Class Stability Analysis ({n_seeds} Perturbation Runs)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "statistical_per_class_stability.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("  ✓ Saved: statistical_per_class_stability.png")


def generate_statistical_significance_heatmap(
    bootstrap_metrics, output_dir: str = None
):
    """
    Create a heatmap showing statistical significance between metrics.
    """
    print("\n[Statistical] Generating significance heatmap...")

    if output_dir is None:
        output_dir = PLOTS_DIR

    metrics_names = ["Accuracy", "Macro F1", "Macro Precision", "Macro Recall"]
    metrics_keys = list(bootstrap_metrics.keys())

    # Compute p-values for pairwise t-tests
    n_metrics = len(metrics_keys)
    p_values = np.zeros((n_metrics, n_metrics))

    for i in range(n_metrics):
        for j in range(n_metrics):
            if i == j:
                p_values[i, j] = 1.0
            else:
                _, p_val = stats.ttest_ind(
                    bootstrap_metrics[metrics_keys[i]],
                    bootstrap_metrics[metrics_keys[j]],
                )
                p_values[i, j] = p_val

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create -log10(p) matrix for better visualization
    significance_matrix = -np.log10(np.maximum(p_values, 1e-10))
    np.fill_diagonal(significance_matrix, 0)

    im = ax.imshow(significance_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=10)

    ax.set_xticks(range(n_metrics))
    ax.set_yticks(range(n_metrics))
    ax.set_xticklabels(metrics_names, fontsize=11, rotation=45, ha="right")
    ax.set_yticklabels(metrics_names, fontsize=11)

    ax.set_title(
        "Statistical Significance Matrix\n(-log10(p-value), p<0.05 threshold at 1.3)",
        fontsize=13,
        fontweight="bold",
    )

    # Annotate
    for i in range(n_metrics):
        for j in range(n_metrics):
            if i != j:
                text = ax.text(
                    j,
                    i,
                    f"{p_values[i, j]:.4f}\n({significance_matrix[i, j]:.2f})",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("-log10(p-value)", fontsize=11)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "statistical_significance_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("  ✓ Saved: statistical_significance_heatmap.png")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate statistical validation visualizations"
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to keras model"
    )
    parser.add_argument(
        "--output-dir", type=str, default=PLOTS_DIR, help="Output directory"
    )
    parser.add_argument(
        "--bootstraps", type=int, default=2000, help="Number of bootstrap resamples"
    )
    args = parser.parse_args()

    # Load model
    model_path = args.model_path or resolve_keras_model_path([FINAL_MODEL_PATH])
    print(f"\n[Statistical Suite] Loading model from: {model_path}")
    model = _load_model_robust(model_path)

    # Load test dataset
    print(f"[Statistical Suite] Loading test dataset from: {TEST_DIR}")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Generate figures
    try:
        bootstrap_metrics, ci_95 = generate_bootstrap_ci_distributions(
            model, test_ds, n_bootstraps=args.bootstraps, output_dir=args.output_dir
        )
    except Exception as e:
        print(f"  ⚠ Skipped bootstrap CIs: {e}")
        bootstrap_metrics = None

    try:
        generate_margin_distribution_plot(model, test_ds, args.output_dir)
    except Exception as e:
        print(f"  ⚠ Skipped margin distribution: {e}")

    try:
        generate_per_class_stability_plot(model, test_ds, args.output_dir, n_seeds=3)
    except Exception as e:
        print(f"  ⚠ Skipped per-class stability: {e}")

    if bootstrap_metrics:
        try:
            generate_statistical_significance_heatmap(
                bootstrap_metrics, args.output_dir
            )
        except Exception as e:
            print(f"  ⚠ Skipped significance heatmap: {e}")

    print("\n✓ Statistical validation suite complete!")


if __name__ == "__main__":
    main()
