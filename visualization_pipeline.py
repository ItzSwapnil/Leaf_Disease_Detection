import os
import json
import csv
import re

import tensorflow.keras as keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    BATCH_SIZE,
    CLASS_INDICES_PATH,
    EPOCHS_PHASE1,
    FINAL_MODEL_PATH,
    IMG_SIZE,
    PLOTS_DIR,
    TEST_DIR,
    TRAIN_DIR,
)
from learning_curve_utils import (
    best_epoch_from_values,
    build_best_epoch_markers,
    combine_train_and_fine_metrics,
    trim_train_metrics_to_restore_epoch,
)
from model_paths import resolve_keras_model_path
from preprocessing import preprocess_batch_for_model
from training_utils import WarmupCosineSchedule

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Module-level model cache to avoid reloading
_MODEL_PATH = None
os.makedirs(PLOTS_DIR, exist_ok=True)

# Class distribution

def generate_class_distribution():
    
    print("\nGenerating class distribution plot...")

    classes = sorted(os.listdir(TRAIN_DIR))
    image_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    counts = []

    for cls in classes:
        cls_path = os.path.join(TRAIN_DIR, cls)
        if os.path.isdir(cls_path):
            count = sum(
                1
                for f in os.listdir(cls_path)
                if os.path.isfile(os.path.join(cls_path, f))
                and f.lower().endswith(image_exts)
            )
            counts.append(count)

    plt.figure(figsize=(16, 10))
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(classes)))
    display_names = [c.replace("___", " - ").replace("_", " ") for c in classes]

    bars = plt.barh(range(len(classes)), counts, color=colors)
    plt.yticks(range(len(classes)), display_names, fontsize=8)
    plt.xlabel("Number of Images", fontsize=12)
    plt.ylabel("Disease Class", fontsize=12)
    plt.title("Dataset Class Distribution (Training Set)", fontsize=14, fontweight="bold")

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_width() + 50,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=7,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/class_distribution.png")

# Confusion matrix

def generate_confusion_matrix():
    
    print("\nGenerating confusion matrix...")

    global _MODEL_PATH
    if _MODEL_PATH is None:
        _MODEL_PATH = resolve_keras_model_path([FINAL_MODEL_PATH])
    model = load_model(
        _MODEL_PATH,
        custom_objects={"WarmupCosineSchedule": WarmupCosineSchedule}
    )

    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    prep_test = test_ds.map(
        lambda x, y: (preprocess_batch_for_model(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    print("  Computing predictions on test set...")
    predictions = model.predict(prep_test, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)

    class_names = [name.split("___")[-1][:15] for name in test_ds.class_names]
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm_norm, annot=False, fmt=".1f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Normalised Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/confusion_matrix.png")

    # Persist class indices
    class_indices = {name: idx for idx, name in enumerate(test_ds.class_names)}
    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=2)
    print(f"Saved: {CLASS_INDICES_PATH}")

    # Summary metrics
    report = classification_report(
        y_true, y_pred, target_names=test_ds.class_names, output_dict=True,
    )
    print(f"\nClassification Report Summary:")
    print(f"  Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
    print(f"  Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")

    return cm, class_indices

# Learning curves (epoch-level)

def _read_metrics_from_csv(csv_path):
    
    if not os.path.exists(csv_path):
        return {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    metrics = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    with open(csv_path, "r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            acc_key = "accuracy" if "accuracy" in row else "acc"
            val_acc_key = "val_accuracy" if "val_accuracy" in row else "val_acc"

            def _f(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return np.nan

            metrics["accuracy"].append(_f(row.get(acc_key)))
            metrics["val_accuracy"].append(_f(row.get(val_acc_key)))
            metrics["loss"].append(_f(row.get("loss")))
            metrics["val_loss"].append(_f(row.get("val_loss")))

    return metrics

def _read_interval_metrics(csv_path, epoch_offset=0.0):
    
    train_x, train_acc, train_loss = [], [], []
    val_x, val_acc, val_loss = [], [], []

    if not os.path.exists(csv_path):
        return {
            "train_x": train_x, "train_acc": train_acc, "train_loss": train_loss,
            "val_x": val_x, "val_acc": val_acc, "val_loss": val_loss,
        }

    with open(csv_path, "r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            try:
                epoch_num = float(row.get("epoch", 0.0))
                epoch_prog = float(row.get("epoch_progress", 0.0))
            except (TypeError, ValueError):
                continue

            x = epoch_offset + max(0.0, epoch_num - 1.0) + max(0.0, min(1.0, epoch_prog))

            def _f(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return np.nan

            train_x.append(x)
            train_loss.append(_f(row.get("loss")))
            train_acc.append(_f(row.get("accuracy")))

            row_type = str(row.get("row_type", "")).strip().lower()
            if row_type == "epoch_end":
                val_x.append(x)
                val_loss.append(_f(row.get("val_loss")))
                val_acc.append(_f(row.get("val_accuracy")))

    return {
        "train_x": train_x, "train_acc": train_acc, "train_loss": train_loss,
        "val_x": val_x, "val_acc": val_acc, "val_loss": val_loss,
    }

def _resolve_learning_curve_logs(logs_dir):
    
    default_paths = {
        "train_interval_log": os.path.join(logs_dir, "train_interval_history.csv"),
        "fine_tune_interval_log": os.path.join(logs_dir, "fine_tune_interval_history.csv"),
        "train_log": os.path.join(logs_dir, "train_history.csv"),
        "fine_tune_log": os.path.join(logs_dir, "fine_tune_history.csv"),
    }

    latest_runs_path = os.path.join(logs_dir, "latest_runs.json")
    if not os.path.exists(latest_runs_path):
        return default_paths

    try:
        with open(latest_runs_path, "r", encoding="utf-8") as in_file:
            latest_runs = json.load(in_file)
    except Exception:
        return default_paths

    train_info = latest_runs.get("train") or {}
    fine_info = latest_runs.get("fine_tune") or {}
    has_fine_run = bool(fine_info)

    def _pick(preferred, fallback):
        if preferred and os.path.exists(preferred):
            return preferred
        return fallback

    return {
        "train_interval_log": _pick(
            train_info.get("train_interval_archive"),
            default_paths["train_interval_log"],
        ),
        "fine_tune_interval_log": _pick(
            fine_info.get("fine_tune_interval_archive"),
            default_paths["fine_tune_interval_log"] if has_fine_run
            else os.path.join(logs_dir, "_no_fine_tune_interval.csv"),
        ),
        "train_log": _pick(
            train_info.get("train_history_archive"),
            default_paths["train_log"],
        ),
        "fine_tune_log": _pick(
            fine_info.get("fine_tune_history_archive"),
            default_paths["fine_tune_log"] if has_fine_run
            else os.path.join(logs_dir, "_no_fine_tune_history.csv"),
        ),
    }

def _read_restore_epochs_from_log(log_path, script_name=None):
    
    if not os.path.exists(log_path):
        return []

    pattern = re.compile(
        r"Restoring model weights from the end of the best epoch:\s*(\d+)\."
    )
    restore_epochs = []
    in_target_run = script_name is None

    with open(log_path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            normalized = line.strip()
            if normalized.startswith("Started:"):
                in_target_run = (script_name is None) or (script_name in normalized)
                continue
            if not in_target_run:
                continue
            m = pattern.search(normalized)
            if m:
                try:
                    restore_epochs.append(int(m.group(1)))
                except ValueError:
                    continue

    seen = set()
    return [e for e in restore_epochs if not (e in seen or seen.add(e))]

def generate_learning_curves_from_logs():
    
    print("\nGenerating learning curves...")

    logs_dir = os.path.join(os.path.dirname(FINAL_MODEL_PATH), "logs")
    resolved_logs = _resolve_learning_curve_logs(logs_dir)
    train_log = resolved_logs["train_log"]
    fine_tune_log = resolved_logs["fine_tune_log"]
    print(f"Using train log: {train_log}")
    print(f"Using fine-tune log: {fine_tune_log}")

    # Check for restore epochs in a log.md if it exists
    project_root = os.path.dirname(os.path.abspath(__file__))
    restore_epochs = _read_restore_epochs_from_log(
        os.path.join(project_root, "LOG.md"), script_name="train_model.py"
    )
    if restore_epochs:
        print(f"Detected restore epochs: {restore_epochs}")

    train_metrics_raw = _read_metrics_from_csv(train_log)
    fine_metrics = _read_metrics_from_csv(fine_tune_log)
    train_metrics, effective_len, dropped = trim_train_metrics_to_restore_epoch(
        train_metrics_raw, restore_epochs,
    )
    if dropped > 0:
        print(f"Trimmed to restored epoch {effective_len} (dropped {dropped}).")

    combined, phase_boundary = combine_train_and_fine_metrics(
        train_metrics, fine_metrics, EPOCHS_PHASE1,
    )
    acc = combined["accuracy"]
    val_acc_epoch = combined["val_accuracy"]
    loss = combined["loss"]
    val_loss_epoch = combined["val_loss"]

    marker_lines, marker_note, _, _, _ = build_best_epoch_markers(
        train_metrics["val_accuracy"], fine_metrics["val_accuracy"], phase_boundary,
    )

    if not marker_lines and len(val_acc_epoch) > 0:
        val_acc_arr = np.array(val_acc_epoch, dtype=np.float64)
        if np.any(np.isfinite(val_acc_arr)):
            inferred = best_epoch_from_values(val_acc_arr) or 1
            marker_lines.append((
                float(inferred), "darkorange", ":", f"Best Epoch ({inferred})",
            ))

    if len(acc) == 0:
        raise FileNotFoundError(
            "No training metrics found. Run train_model.py first."
        )

    total_epochs = len(acc)
    train_x = np.arange(1, total_epochs + 1, dtype=np.float64)
    val_x = train_x.copy()
    train_acc = np.array(acc, dtype=np.float64)
    train_loss = np.array(loss, dtype=np.float64)
    val_acc = np.array(val_acc_epoch, dtype=np.float64)
    val_loss = np.array(val_loss_epoch, dtype=np.float64)

    if train_x.size > 1:
        order = np.argsort(train_x)
        train_x, train_acc, train_loss = train_x[order], train_acc[order], train_loss[order]
        val_x, val_acc, val_loss = val_x[order], val_acc[order], val_loss[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy subplot
    ax1.plot(train_x, train_acc, "b-", label="Training Accuracy", linewidth=1.6)
    ax1.plot(val_x, val_acc, "r-", label="Validation Accuracy", linewidth=2)
    if phase_boundary > 0:
        ax1.axvline(phase_boundary, color="gray", linestyle="--", alpha=0.6, label="Fine-tune Start")
    for epoch, color, style, label in marker_lines:
        ax1.axvline(epoch, color=color, linestyle=style, alpha=0.85, linewidth=1.4, label=label)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Model Accuracy Over Training", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.0, 1.0])

    # Loss subplot
    ax2.plot(train_x, train_loss, "b-", label="Training Loss", linewidth=1.6)
    ax2.plot(val_x, val_loss, "r-", label="Validation Loss", linewidth=2)
    if phase_boundary > 0:
        ax2.axvline(phase_boundary, color="gray", linestyle="--", alpha=0.6, label="Fine-tune Start")
    for epoch, color, style, label in marker_lines:
        ax2.axvline(epoch, color=color, linestyle=style, alpha=0.85, linewidth=1.4, label=label)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Model Loss Over Training", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    valid_losses = np.concatenate([
        train_loss[~np.isnan(train_loss)], val_loss[~np.isnan(val_loss)],
    ])
    if valid_losses.size > 0:
        low, high = float(np.min(valid_losses)), float(np.max(valid_losses))
        margin = max(0.05, (high - low) * 0.12)
        ax2.set_ylim([max(0.0, low - margin), high + margin])

    if marker_note:
        ax1.text(
            0.02, 0.04, marker_note, transform=ax1.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/learning_curves.png")

# Model architecture diagram

def generate_model_architecture_diagram():
    
    print("\nGenerating model architecture diagram...")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    layers = [
        ("Input\n(224x224x3)", "#87CEEB", 0.8),
        ("EfficientNetV2-S\n(ImageNet Pretrained)", "#90EE90", 1.2),
        ("GlobalAvgPool2D\n(1280)", "#FFFACD", 0.6),
        ("BatchNorm\n(1280)", "#FFFACD", 0.5),
        ("Dense + Swish\n(512)", "#F08080", 0.6),
        ("Dropout (0.4)", "#D3D3D3", 0.4),
        ("Dense + Swish\n(256)", "#F08080", 0.6),
        ("Dropout (0.2)", "#D3D3D3", 0.4),
        ("Dense + Softmax\n(46 classes)", "#DDA0DD", 0.7),
    ]

    y_start = 11
    y_spacing = 1.1

    for i, (name, color, height) in enumerate(layers):
        y = y_start - i * y_spacing
        rect = plt.Rectangle(
            (2, y - height / 2), 6, height,
            facecolor=color, edgecolor="black", linewidth=2, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(5, y, name, ha="center", va="center", fontsize=11, fontweight="bold")

        if i < len(layers) - 1:
            ax.annotate(
                "", xy=(5, y - height / 2 - 0.05), xytext=(5, y - height / 2 - 0.3),
                arrowprops=dict(arrowstyle="->", color="black", lw=2),
            )

    ax.set_title(
        "EfficientNetV2-S + SOTA Classification Head",
        fontsize=14, fontweight="bold", y=0.98,
    )
    ax.text(
        0.5, 0.5,
        "Total Parameters: ~21.4M\nTrainable (Phase 1): ~1.2M\nTrainable (Phase 2): ~21.4M",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_architecture.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/model_architecture.png")

# Sample predictions

def generate_sample_predictions():
    
    print("\nGenerating sample predictions visualisation...")

    global _MODEL_PATH
    if _MODEL_PATH is None:
        _MODEL_PATH = resolve_keras_model_path([FINAL_MODEL_PATH])
    model = load_model(
        _MODEL_PATH,
        custom_objects={"WarmupCosineSchedule": WarmupCosineSchedule}
    )

    sample_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        shuffle=True,
        seed=42,
    )

    idx_to_class = {idx: name for idx, name in enumerate(sample_ds.class_names)}

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, (img_batch, label_batch) in enumerate(sample_ds.take(12)):
        pred = model.predict(preprocess_batch_for_model(img_batch), verbose=0)
        pred_class = int(np.argmax(pred))
        confidence = float(np.max(pred)) * 100
        true_class = int(label_batch.numpy()[0])

        display_img = img_batch.numpy()[0].astype(np.uint8)
        axes[i].imshow(display_img)

        pred_name = idx_to_class[pred_class].replace("___", "\n").replace("_", " ")
        color = "green" if pred_class == true_class else "red"
        axes[i].set_title(f"Pred: {pred_name}\n({confidence:.1f}%)", fontsize=8, color=color)
        axes[i].axis("off")

    plt.suptitle(
        "Sample Predictions (Green = Correct, Red = Incorrect)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sample_predictions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/sample_predictions.png")

# Main

def main():
    print("generating plots...")
    generate_class_distribution()
    generate_learning_curves_from_logs()
    generate_model_architecture_diagram()
    generate_confusion_matrix()
    generate_sample_predictions()
    print("All figures generated successfully.")
    print(f"Output directory: {PLOTS_DIR}/")
    print("  - class_distribution.png")
    print("  - learning_curves.png")
    print("  - model_architecture.png")
    print("  - confusion_matrix.png")
    print("  - sample_predictions.png")


if __name__ == "__main__":
    main()
