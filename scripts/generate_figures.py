import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import tensorflow.keras as keras

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from src.utils.config import (
    BATCH_SIZE,
    CLASS_INDICES_PATH,
    FINAL_MODEL_PATH,
    IMG_SIZE,
    TEST_DIR,
    TRAIN_DIR,
    WARMUP_EPOCHS,
)
from src.training.learning_curve_utils import (
    best_epoch_from_values,
)
from src.utils.model_paths import resolve_keras_model_path
from src.core.preprocessing import preprocess_batch_for_model_tf
from scripts.figure_paths import (
    OTHERS_PLOTS_DIR,
    backbone_plot_dir,
    prepare_plot_directories,
)
from src.training.training_utils import WarmupCosineSchedule

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Module-level model cache to avoid reloading
_MODEL_PATH = None
MODEL_PATH_OVERRIDE = None
FIGURE_OUTPUT_DIR = OTHERS_PLOTS_DIR
prepare_plot_directories()


def _patch_vit_layer_init_for_compat():

    try:
        from keras_hub.src.models.vit import vit_layers

        layer_cls = vit_layers.ViTPatchingAndEmbedding
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

    custom_objects = {"WarmupCosineSchedule": WarmupCosineSchedule}
    try:
        return load_model(
            model_path, custom_objects=custom_objects, compile=False
        )
    except Exception as exc:
        message = str(exc)
        vit_compat_error = (
            "ViTPatchingAndEmbedding" in message
            or "num_patches" in message
            or "num_positions" in message
        )
        if not vit_compat_error:
            raise
        if not _patch_vit_layer_init_for_compat():
            raise
        print("Applied ViT compatibility shim while loading model.")
        return load_model(
            model_path, custom_objects=custom_objects, compile=False
        )


def _infer_backbone_from_model(model) -> str:

    name = str(getattr(model, "name", "") or "").lower()
    if "dino" in name or "vit" in name:
        return "DINOv3"

    for layer in model.layers:
        layer_name = str(getattr(layer, "name", "") or "").lower()
        class_name = layer.__class__.__name__.lower()
        if "dino" in layer_name or "vit" in layer_name:
            return "DINOv3"
        if "vit" in class_name:
            return "DINOv3"

    return "EfficientNetV2B0"


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
    display_names = [
        c.replace("___", " - ").replace("_", " ") for c in classes
    ]

    bars = plt.barh(range(len(classes)), counts, color=colors)
    plt.yticks(range(len(classes)), display_names, fontsize=8)
    plt.xlabel("Number of Images", fontsize=12)
    plt.ylabel("Disease Class", fontsize=12)
    plt.title(
        "Dataset Class Distribution (Training Set)",
        fontsize=14,
        fontweight="bold",
    )

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_width() + 50,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=7,
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OTHERS_PLOTS_DIR, "class_distribution.png"),
        dpi=800,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {OTHERS_PLOTS_DIR}/class_distribution.png")


# Confusion matrix


def generate_confusion_matrix():

    print("\nGenerating confusion matrix...")

    global _MODEL_PATH
    if _MODEL_PATH is None:
        path_candidates = (
            [MODEL_PATH_OVERRIDE]
            if MODEL_PATH_OVERRIDE
            else [FINAL_MODEL_PATH]
        )
        _MODEL_PATH = resolve_keras_model_path(path_candidates)
    model = _load_model_robust(_MODEL_PATH)
    backbone_name = _infer_backbone_from_model(model)
    global FIGURE_OUTPUT_DIR
    FIGURE_OUTPUT_DIR = backbone_plot_dir(backbone_name)

    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    prep_test = test_ds.map(
        lambda x, y: (
            preprocess_batch_for_model_tf(x, backbone_name=backbone_name),
            y,
        ),
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
        cm_norm,
        annot=False,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(
        "Normalised Confusion Matrix (Test Set)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_OUTPUT_DIR, "confusion_matrix.png"),
        dpi=800,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {FIGURE_OUTPUT_DIR}/confusion_matrix.png")

    # Persist class indices
    class_indices = {name: idx for idx, name in enumerate(test_ds.class_names)}
    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=2)
    print(f"Saved: {CLASS_INDICES_PATH}")

    # Summary metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=test_ds.class_names,
        output_dict=True,
    )
    print("\nClassification Report Summary:")
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
            val_acc_key = (
                "val_accuracy" if "val_accuracy" in row else "val_acc"
            )

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
            "train_x": train_x,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "val_x": val_x,
            "val_acc": val_acc,
            "val_loss": val_loss,
        }

    with open(csv_path, "r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            try:
                epoch_num = float(row.get("epoch", 0.0))
                epoch_prog = float(row.get("epoch_progress", 0.0))
            except (TypeError, ValueError):
                continue

            x = (
                epoch_offset
                + max(0.0, epoch_num - 1.0)
                + max(0.0, min(1.0, epoch_prog))
            )

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
        "train_x": train_x,
        "train_acc": train_acc,
        "train_loss": train_loss,
        "val_x": val_x,
        "val_acc": val_acc,
        "val_loss": val_loss,
    }


def _read_interval_epoch_end_metrics(csv_path):

    empty = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    if not csv_path or not os.path.exists(csv_path):
        return empty, []

    def _f(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    rows_by_epoch = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            if str(row.get("row_type", "")).strip().lower() != "epoch_end":
                continue
            try:
                epoch = int(float(row.get("epoch", 0.0)))
            except (TypeError, ValueError):
                continue
            if epoch <= 0:
                continue
            rows_by_epoch[epoch] = row

    if not rows_by_epoch:
        return empty, []

    metrics = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    lr_values = []
    for epoch in sorted(rows_by_epoch):
        row = rows_by_epoch[epoch]
        metrics["accuracy"].append(_f(row.get("accuracy")))
        metrics["val_accuracy"].append(_f(row.get("val_accuracy")))
        metrics["loss"].append(_f(row.get("loss")))
        metrics["val_loss"].append(_f(row.get("val_loss")))
        lr_values.append(_f(row.get("learning_rate")))

    return metrics, lr_values


def _get_latest_runs(logs_dir):

    latest_runs_path = os.path.join(logs_dir, "latest_runs.json")
    if not os.path.exists(latest_runs_path):
        return {}
    try:
        with open(latest_runs_path, "r", encoding="utf-8") as in_file:
            return json.load(in_file)
    except Exception:
        return {}


def _path_if_exists(path_value):

    if not path_value:
        return None
    try:
        if os.path.exists(path_value):
            return path_value
    except Exception:
        return None
    return None


def _resolve_phase_logs(logs_dir):

    latest_runs = _get_latest_runs(logs_dir)
    train_info = latest_runs.get("train") or {}
    fine_info = latest_runs.get("fine_tune") or {}
    refine_info = latest_runs.get("refine") or {}

    def _pick(*candidates):
        for candidate in candidates:
            resolved = _path_if_exists(candidate)
            if resolved:
                return resolved
        return None

    train_epochs_cfg = int(train_info.get("epochs_phase1") or 0) + int(
        train_info.get("epochs_phase2") or 0
    )
    fine_epochs_cfg = int(fine_info.get("fine_tune_epochs") or 0)
    refine_epochs_cfg = int(refine_info.get("epochs") or 0)

    phases = [
        {
            "key": "train",
            "label": "Training",
            "history": _pick(
                train_info.get("train_history_archive"),
                train_info.get("train_history_latest"),
                os.path.join(logs_dir, "train_history.csv"),
            ),
            "interval": _pick(
                train_info.get("train_interval_archive"),
                train_info.get("train_interval_latest"),
                os.path.join(logs_dir, "train_interval_history.csv"),
            ),
            "configured_epochs": train_epochs_cfg,
            "warmup_epochs": min(int(WARMUP_EPOCHS), max(0, train_epochs_cfg)),
            "run_stamp": train_info.get("run_stamp"),
        },
        {
            "key": "fine_tune",
            "label": "Fine-tuning",
            "history": _pick(
                fine_info.get("fine_tune_history_archive"),
                fine_info.get("fine_tune_history_latest"),
                os.path.join(logs_dir, "fine_tune_history.csv"),
            ),
            "interval": _pick(
                fine_info.get("fine_tune_interval_archive"),
                fine_info.get("fine_tune_interval_latest"),
                os.path.join(logs_dir, "fine_tune_interval_history.csv"),
            ),
            "configured_epochs": fine_epochs_cfg,
            "warmup_epochs": min(int(WARMUP_EPOCHS), max(0, fine_epochs_cfg)),
            "run_stamp": fine_info.get("run_stamp"),
        },
        {
            "key": "refine",
            "label": "Refining",
            "history": _pick(
                os.path.join(logs_dir, "refine_history.csv"),
            ),
            "interval": _pick(
                os.path.join(logs_dir, "refine_interval_history.csv"),
            ),
            "configured_epochs": refine_epochs_cfg,
            "warmup_epochs": min(1, max(0, refine_epochs_cfg)),
            "run_stamp": refine_info.get("run_stamp"),
        },
    ]

    return phases, latest_runs


def _epoch_fallback_interval(metrics, epoch_offset=0.0):

    count = len(metrics["accuracy"])
    if count == 0:
        return {
            "train_x": [],
            "train_acc": [],
            "train_loss": [],
            "val_x": [],
            "val_acc": [],
            "val_loss": [],
            "lr_x": [],
            "lr": [],
        }

    x = [epoch_offset + float(i + 1) for i in range(count)]
    return {
        "train_x": x,
        "train_acc": list(metrics["accuracy"]),
        "train_loss": list(metrics["loss"]),
        "val_x": x,
        "val_acc": list(metrics["val_accuracy"]),
        "val_loss": list(metrics["val_loss"]),
        "lr_x": [],
        "lr": [],
    }


def _read_interval_with_lr(csv_path, epoch_offset=0.0):

    parsed = _read_interval_metrics(csv_path, epoch_offset=epoch_offset)
    lr_x, lr_values = [], []
    if not csv_path or not os.path.exists(csv_path):
        parsed["lr_x"] = lr_x
        parsed["lr"] = lr_values
        return parsed

    with open(csv_path, "r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            try:
                epoch_num = float(row.get("epoch", 0.0))
                epoch_prog = float(row.get("epoch_progress", 0.0))
                lr = float(row.get("learning_rate"))
            except (TypeError, ValueError):
                continue
            x = (
                epoch_offset
                + max(0.0, epoch_num - 1.0)
                + max(0.0, min(1.0, epoch_prog))
            )
            lr_x.append(x)
            lr_values.append(lr)

    parsed["lr_x"] = lr_x
    parsed["lr"] = lr_values
    return parsed


def _phase_best_epoch(metrics):

    local = best_epoch_from_values(metrics.get("val_accuracy", []))
    if local is None:
        return None
    return int(local)


def _find_refine_best_safe_epoch(logs_dir, run_stamp):

    if not run_stamp:
        return None
    snapshot_dir = Path(logs_dir) / f"refine_snapshots_{run_stamp}"
    if not snapshot_dir.exists():
        return None
    safe_files = list(
        snapshot_dir.glob("safe_epoch_*_val_accuracy_*.weights.h5")
    )
    if not safe_files:
        return None

    best_epoch = None
    best_metric = float("-inf")
    pattern = re.compile(
        r"safe_epoch_(\d+)_val_accuracy_([0-9]+\.[0-9]+)\.weights\.h5"
    )
    for path in safe_files:
        match = pattern.match(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        metric = float(match.group(2))
        if metric >= best_metric:
            best_metric = metric
            best_epoch = epoch
    return best_epoch


def _dedupe_legend(ax):

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    out_handles, out_labels = [], []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        out_handles.append(handle)
        out_labels.append(label)
    if out_handles:
        ax.legend(out_handles, out_labels)


def _resolve_learning_curve_logs(logs_dir):

    default_paths = {
        "train_interval_log": os.path.join(
            logs_dir, "train_interval_history.csv"
        ),
        "fine_tune_interval_log": os.path.join(
            logs_dir, "fine_tune_interval_history.csv"
        ),
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
            default_paths["fine_tune_interval_log"]
            if has_fine_run
            else os.path.join(logs_dir, "_no_fine_tune_interval.csv"),
        ),
        "train_log": _pick(
            train_info.get("train_history_archive"),
            default_paths["train_log"],
        ),
        "fine_tune_log": _pick(
            fine_info.get("fine_tune_history_archive"),
            default_paths["fine_tune_log"]
            if has_fine_run
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
                in_target_run = (script_name is None) or (
                    script_name in normalized
                )
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

    model_for_logs = _MODEL_PATH or FINAL_MODEL_PATH
    logs_dir = os.path.join(os.path.dirname(model_for_logs), "logs")
    phases, latest_runs = _resolve_phase_logs(logs_dir)

    phase_runs = []
    for phase in phases:
        history_metrics = (
            _read_metrics_from_csv(phase["history"])
            if phase["history"]
            else {
                "accuracy": [],
                "val_accuracy": [],
                "loss": [],
                "val_loss": [],
            }
        )
        history_len = len(history_metrics["accuracy"])
        configured_epochs = int(phase.get("configured_epochs", 0))
        ended_early = configured_epochs > 0 and history_len < configured_epochs
        interval_lr_values = []
        if phase.get("interval") and os.path.exists(phase["interval"]):
            _, interval_lr_values = _read_interval_epoch_end_metrics(
                phase["interval"]
            )

        restore_target_local = None
        if phase["key"] == "refine":
            refine_best_safe = _find_refine_best_safe_epoch(
                logs_dir, phase.get("run_stamp")
            )
            if (
                refine_best_safe is not None
                and 1 <= int(refine_best_safe) <= history_len
            ):
                restore_target_local = int(refine_best_safe)
        elif ended_early:
            best_local = _phase_best_epoch(history_metrics)
            if best_local is not None and best_local < history_len:
                restore_target_local = int(best_local)

        needs_epoch_end_fallback = (
            phase.get("interval")
            and os.path.exists(phase["interval"])
            and (
                ended_early
                or (
                    restore_target_local is not None
                    and restore_target_local < history_len
                )
            )
        )

        selected_metrics = history_metrics
        lr_epoch_values = []
        used_epoch_end_fallback = False
        if needs_epoch_end_fallback:
            interval_epoch_metrics, interval_lr = (
                _read_interval_epoch_end_metrics(phase["interval"])
            )
            if len(interval_epoch_metrics["accuracy"]) > 0:
                selected_metrics = interval_epoch_metrics
                lr_epoch_values = interval_lr
                used_epoch_end_fallback = True
        if not lr_epoch_values and interval_lr_values:
            lr_epoch_values = interval_lr_values

        selected_len = len(selected_metrics["accuracy"])
        if (
            restore_target_local is not None
            and restore_target_local > selected_len
        ):
            restore_target_local = None

        source_name = (
            "interval(epoch_end)" if used_epoch_end_fallback else "history"
        )
        print(
            f"Using {phase['key']} {source_name}: "
            f"epochs={selected_len}"
            + (
                f", configured={configured_epochs}"
                if configured_epochs > 0
                else ""
            )
        )

        phase_runs.append(
            {
                "key": phase["key"],
                "label": phase["label"],
                "metrics": selected_metrics,
                "lr_epoch_values": lr_epoch_values,
                "warmup_epochs": int(phase.get("warmup_epochs", 0)),
                "configured_epochs": configured_epochs,
                "ended_early": bool(ended_early),
                "restore_target_local": restore_target_local,
            }
        )

    total_epochs = sum(len(run["metrics"]["accuracy"]) for run in phase_runs)
    if total_epochs == 0:
        raise FileNotFoundError(
            "No training metrics found. Run train_model.py first."
        )

    phase_offsets = {}
    running_offset = 0
    for run in phase_runs:
        phase_offsets[run["key"]] = running_offset
        running_offset += len(run["metrics"]["accuracy"])

    combined = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    lr_x, lr_values = [], []
    for run in phase_runs:
        metrics = run["metrics"]
        phase_len = len(metrics["accuracy"])
        for key in combined:
            combined[key].extend(metrics[key])

        if run["lr_epoch_values"]:
            offset = phase_offsets[run["key"]]
            for epoch_idx, lr in enumerate(run["lr_epoch_values"], start=1):
                if epoch_idx > phase_len:
                    break
                if np.isfinite(lr):
                    lr_x.append(float(offset + epoch_idx))
                    lr_values.append(float(lr))

    train_x = np.arange(1, total_epochs + 1, dtype=np.float64)
    val_x = train_x.copy()
    train_acc = np.array(combined["accuracy"], dtype=np.float64)
    train_loss = np.array(combined["loss"], dtype=np.float64)
    val_acc = np.array(combined["val_accuracy"], dtype=np.float64)
    val_loss = np.array(combined["val_loss"], dtype=np.float64)

    marker_lines = []
    phase_start_markers = []
    warmup_markers = []
    restore_markers = []

    for run in phase_runs:
        key = run["key"]
        length = len(run["metrics"]["accuracy"])
        if length <= 0:
            continue
        start_epoch = phase_offsets[key] + 1
        phase_start_markers.append(
            (
                float(start_epoch),
                "#5C6B73",
                "--",
                f"{run['label']} Start",
            )
        )

        warmup_epochs = min(int(run["warmup_epochs"]), length)
        if warmup_epochs > 0:
            warmup_end = phase_offsets[key] + warmup_epochs
            warmup_markers.append(
                (
                    float(warmup_end),
                    "#8E24AA",
                    "-.",
                    f"{run['label']} Warmup End",
                )
            )

        best_local = _phase_best_epoch(run["metrics"])
        if best_local is not None:
            best_global = phase_offsets[key] + best_local
            marker_lines.append(
                (
                    float(best_global),
                    "#2E7D32",
                    ":",
                    f"{run['label']} Best (epoch {best_global})",
                )
            )

        restore_target_local = run["restore_target_local"]
        if restore_target_local is not None and restore_target_local < length:
            restore_event_global = phase_offsets[key] + length
            restore_target_global = phase_offsets[key] + restore_target_local
            restore_markers.append(
                (
                    float(restore_event_global),
                    "#C62828",
                    "--",
                    f"{run['label']} Restore Event ({restore_event_global} -> {restore_target_global})",
                )
            )

    global_best = best_epoch_from_values(combined["val_accuracy"])
    if global_best is not None:
        marker_lines.append(
            (
                float(global_best),
                "#E65100",
                "-",
                f"Global Best (epoch {global_best})",
            )
        )

    all_markers = (
        phase_start_markers + warmup_markers + marker_lines + restore_markers
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy subplot
    ax1.plot(
        train_x,
        train_acc,
        "b-o",
        label="Training Accuracy",
        linewidth=1.6,
        markersize=3,
    )
    ax1.plot(
        val_x,
        val_acc,
        "r-o",
        label="Validation Accuracy",
        linewidth=2,
        markersize=3,
    )
    for epoch, color, style, label in all_markers:
        ax1.axvline(
            epoch,
            color=color,
            linestyle=style,
            alpha=0.85,
            linewidth=1.4,
            label=label,
        )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(
        "Model Accuracy Over Training", fontsize=14, fontweight="bold"
    )
    _dedupe_legend(ax1)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.0, 1.0])

    # Loss subplot
    ax2.plot(
        train_x,
        train_loss,
        "b-o",
        label="Training Loss",
        linewidth=1.6,
        markersize=3,
    )
    ax2.plot(
        val_x,
        val_loss,
        "r-o",
        label="Validation Loss",
        linewidth=2,
        markersize=3,
    )
    for epoch, color, style, label in all_markers:
        ax2.axvline(
            epoch,
            color=color,
            linestyle=style,
            alpha=0.85,
            linewidth=1.4,
            label=label,
        )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Model Loss Over Training", fontsize=14, fontweight="bold")
    _dedupe_legend(ax2)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    valid_losses = np.concatenate(
        [
            train_loss[~np.isnan(train_loss)],
            val_loss[~np.isnan(val_loss)],
        ]
    )
    if valid_losses.size > 0:
        low, high = float(np.min(valid_losses)), float(np.max(valid_losses))
        margin = max(0.05, (high - low) * 0.12)
        ax2.set_ylim([max(0.0, low - margin), high + margin])

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_OUTPUT_DIR, "learning_curves.png"),
        dpi=800,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {FIGURE_OUTPUT_DIR}/learning_curves.png")

    # Overview figure with per-phase accuracy curves and learning-rate evolution.
    fig, ax_left = plt.subplots(figsize=(14, 5))
    phase_curve_colors = {
        "train": "#1D4ED8",
        "fine_tune": "#F59E0B",
        "refine": "#16A34A",
    }
    for run in phase_runs:
        key = run["key"]
        offset = phase_offsets.get(key, 0)
        metrics = run["metrics"]
        phase_len = len(metrics["accuracy"])
        if phase_len <= 0:
            continue
        x_phase = np.arange(1, phase_len + 1, dtype=np.float64) + float(offset)
        acc_phase = np.array(metrics["accuracy"], dtype=np.float64)
        val_acc_phase = np.array(metrics["val_accuracy"], dtype=np.float64)
        color = phase_curve_colors.get(key, "#374151")

        ax_left.plot(
            x_phase,
            acc_phase,
            color=color,
            linewidth=1.4,
            linestyle="--",
            alpha=0.7,
            label=f"{run['label']} Train Acc",
        )
        ax_left.plot(
            x_phase,
            val_acc_phase,
            color=color,
            linewidth=2.0,
            linestyle="-",
            alpha=0.95,
            label=f"{run['label']} Val Acc",
        )
    ax_left.set_xlabel("Global Epoch", fontsize=12)
    ax_left.set_ylabel("Accuracy", fontsize=12)
    ax_left.set_ylim([0.0, 1.0])
    ax_left.grid(True, alpha=0.3)

    for epoch, color, style, label in all_markers:
        ax_left.axvline(
            epoch,
            color=color,
            linestyle=style,
            alpha=0.55,
            linewidth=1.1,
            label=label,
        )

    if lr_x and lr_values:
        ax_right = ax_left.twinx()
        ax_right.plot(
            np.array(lr_x, dtype=np.float64),
            np.array(lr_values, dtype=np.float64),
            color="#1E88E5",
            alpha=0.6,
            linewidth=1.4,
            label="Learning Rate",
        )
        ax_right.set_yscale("log")
        ax_right.set_ylabel("Learning Rate", fontsize=12)

        left_h, left_l = ax_left.get_legend_handles_labels()
        right_h, right_l = ax_right.get_legend_handles_labels()
        merged_h = left_h + right_h
        merged_l = left_l + right_l
        seen = set()
        clean_h, clean_l = [], []
        for handle, label in zip(merged_h, merged_l):
            if label in seen:
                continue
            seen.add(label)
            clean_h.append(handle)
            clean_l.append(label)
        ax_left.legend(clean_h, clean_l, loc="lower right", fontsize=8)
    else:
        _dedupe_legend(ax_left)
        ax_left.legend(loc="lower right", fontsize=8)

    ax_left.set_title(
        "Training Timeline Overview", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_OUTPUT_DIR, "training_timeline_overview.png"),
        dpi=800,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {FIGURE_OUTPUT_DIR}/training_timeline_overview.png")


# Model architecture diagram


def generate_model_architecture_diagram():
    """Generate backbone-specific architecture diagrams for DINOv3 vs EfficientNetV2."""
    print("\nGenerating backbone-specific architecture diagrams...")

    global _MODEL_PATH, FIGURE_OUTPUT_DIR

    if _MODEL_PATH is None:
        path_candidates = (
            [MODEL_PATH_OVERRIDE]
            if MODEL_PATH_OVERRIDE
            else [FINAL_MODEL_PATH]
        )
        _MODEL_PATH = resolve_keras_model_path(path_candidates)

    model_path_lower = str(_MODEL_PATH).lower()

    # Check if this is a DINO model or EfficientNet model
    # DINOv3 uses "refined" model; EfficientNet uses "b0", "b1", "s" in path
    if (
        "refined" in model_path_lower
        or "dino" in model_path_lower
        or "vit" in model_path_lower
    ):
        _generate_dinov3_architecture()
    else:
        _generate_efficientnetv2_architecture()

    print("Saved architecture diagrams to backbone-specific plot folders")


def _draw_vertical_architecture_pipeline(
    ax,
    layers,
    *,
    x_center=5.0,
    box_width=8.2,
    box_height=0.95,
    y_top=13.2,
    gap=0.52,
    text_size=12,
):
    """Draw a centered vertical architecture pipeline with consistent spacing/arrows."""
    from matplotlib.patches import FancyArrowPatch

    x_left = x_center - box_width / 2
    step = box_height + gap

    for i, (name, color) in enumerate(layers):
        y_center = y_top - i * step

        rect = plt.Rectangle(
            (x_left, y_center - box_height / 2),
            box_width,
            box_height,
            facecolor=color,
            edgecolor="black",
            linewidth=2.0,
            alpha=0.9,
        )
        ax.add_patch(rect)

        ax.text(
            x_center,
            y_center,
            name,
            ha="center",
            va="center",
            fontsize=text_size,
            fontweight="bold",
            linespacing=1.35,
        )

        if i < len(layers) - 1:
            next_y_center = y_top - (i + 1) * step
            y_start = y_center - box_height / 2 - 0.04
            y_end = next_y_center + box_height / 2 + 0.04
            arrow = FancyArrowPatch(
                (x_center, y_start),
                (x_center, y_end),
                arrowstyle="-|>",
                color="black",
                linewidth=2.2,
                mutation_scale=18,
                shrinkA=0,
                shrinkB=0,
            )
            ax.add_patch(arrow)


def _generate_dinov3_architecture():
    """Generate architecture schematic for DINOv3 Vision Transformer."""
    fig, ax = plt.subplots(figsize=(12.5, 13.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.8)
    ax.axis("off")

    # Centered vertical layout shared across all model architecture plots.
    layers = [
        ("Input Image\n224×224×3 RGB", "#87CEEB"),
        ("Patch Embedding\n(14×14=196 patches, 768-dim)", "#FFB6C1"),
        ("Transformer Blocks ×12\n(12-heads, 3072 hidden-dim)", "#90EE90"),
        ("Backbone Output\n768-dim features", "#FFD700"),
        ("Layer Norm\n(Trainable scale & bias)", "#FFFACD"),
        ("Dense(512, Swish)\n+ Dropout(0.4)", "#F08080"),
        ("Dense(256, Swish)\n+ Dropout(0.2)", "#F08080"),
        ("Softmax Output\n46 classes", "#98FB98"),
    ]

    vertical_step = 0.95 + 0.52
    y_top = 1.55 + (len(layers) - 1) * vertical_step

    _draw_vertical_architecture_pipeline(
        ax,
        layers,
        x_center=5.0,
        box_width=8.2,
        box_height=0.95,
        y_top=y_top,
        gap=0.52,
        text_size=12,
    )

    ax.set_title(
        "DINOv3 Vision Transformer  |  ViT-Base-Patch16  |  ~87M params",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.text(
        5.0,
        0.46,
        "Training: 5+10→10→30 epochs | Self-supervised ViT with progressive unfreezing",
        fontsize=12.5,
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.75
        ),
        linespacing=1.3,
    )

    plt.tight_layout(pad=0.8)
    output_dir = FIGURE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, "model_architecture.png")
    plt.savefig(output_path, dpi=800, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def _generate_efficientnetv2_architecture():
    """Generate architecture schematic for EfficientNetV2."""
    global _MODEL_PATH

    # Detect if this is B0 or S variant
    model_path_lower = str(_MODEL_PATH).lower()
    is_b0 = "efficientnetv2b0" in model_path_lower or "-b0" in model_path_lower
    is_s = "efficientnetv2s" in model_path_lower or "-s" in model_path_lower

    if is_b0:
        variant_name = "V2-B0"
        input_size = "128×128"
        params = "~7M"
        head_activation = "ReLU"
    elif is_s:
        variant_name = "V2-S"
        input_size = "224×224"
        params = "~21M"
        head_activation = "Swish"
    else:
        variant_name = "V2"
        input_size = "224×224"
        params = "~21M"
        head_activation = "Swish"

    fig, ax = plt.subplots(figsize=(12.5, 11.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Match DINO layout style: centered vertical pipeline with equal spacing.
    stages = [
        ("Input Image\n" + input_size + "×3 RGB", "#87CEEB"),
        ("Stem Conv\n3×3 + BatchNorm", "#FFB6C1"),
        ("MBConv Stages\nProgressive blocks", "#90EE90"),
        ("Backbone Output\nFeature embeddings", "#FFD700"),
        ("GlobalAvgPool\n+ BatchNorm", "#FFE4B5"),
        (f"Dense Head\n512→256 ({head_activation})", "#F08080"),
        ("Output\n46 classes", "#98FB98"),
    ]

    vertical_step = 0.95 + 0.52
    pipeline_center_y = 6.0
    y_top = pipeline_center_y + ((len(stages) - 1) * vertical_step) / 2

    _draw_vertical_architecture_pipeline(
        ax,
        stages,
        x_center=5.0,
        box_width=8.2,
        box_height=0.95,
        y_top=y_top,
        gap=0.52,
        text_size=12,
    )

    ax.set_title(
        f"EfficientNetV2 {variant_name}  |  {input_size} input  |  {params}",
        fontsize=16,
        fontweight="bold",
        pad=8,
    )
    ax.text(
        5.0,
        0.58,
        f"MBConv backbone + {head_activation} classifier head",
        fontsize=12,
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.6
        ),
    )

    plt.tight_layout(pad=0.3)
    output_dir = FIGURE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, "model_architecture.png")
    plt.savefig(output_path, dpi=800, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved: {output_path}")


# Sample predictions


def generate_sample_predictions():

    print("\nGenerating sample predictions visualisation...")

    global _MODEL_PATH
    if _MODEL_PATH is None:
        path_candidates = (
            [MODEL_PATH_OVERRIDE]
            if MODEL_PATH_OVERRIDE
            else [FINAL_MODEL_PATH]
        )
        _MODEL_PATH = resolve_keras_model_path(path_candidates)
    model = _load_model_robust(_MODEL_PATH)
    backbone_name = _infer_backbone_from_model(model)

    sample_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        shuffle=True,
        seed=42,
    )

    idx_to_class = {
        idx: name for idx, name in enumerate(sample_ds.class_names)
    }

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, (img_batch, label_batch) in enumerate(sample_ds.take(12)):
        pred = model.predict(
            preprocess_batch_for_model_tf(
                img_batch, backbone_name=backbone_name
            ),
            verbose=0,
        )
        pred_class = int(np.argmax(pred))
        confidence = float(np.max(pred)) * 100
        true_class = int(label_batch.numpy()[0])

        display_img = img_batch.numpy()[0].astype(np.uint8)
        axes[i].imshow(display_img)

        pred_name = (
            idx_to_class[pred_class].replace("___", "\n").replace("_", " ")
        )
        color = "green" if pred_class == true_class else "red"
        axes[i].set_title(
            f"Pred: {pred_name}\n({confidence:.1f}%)", fontsize=8, color=color
        )
        axes[i].axis("off")

    plt.suptitle(
        "Sample Predictions (Green = Correct, Red = Incorrect)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_OUTPUT_DIR, "sample_predictions.png"),
        dpi=800,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {FIGURE_OUTPUT_DIR}/sample_predictions.png")


# Main


def main():
    global MODEL_PATH_OVERRIDE, _MODEL_PATH, FIGURE_OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Generate core figures.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit model path for model-dependent plots.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory for generated figures.",
    )
    args = parser.parse_args()
    MODEL_PATH_OVERRIDE = args.model_path

    # Resolve selected canonical model path once and bind output directory early,
    # so learning/timeline/architecture are generated per selected backbone.
    path_candidates = (
        [MODEL_PATH_OVERRIDE] if MODEL_PATH_OVERRIDE else [FINAL_MODEL_PATH]
    )
    _MODEL_PATH = resolve_keras_model_path(path_candidates)

    # If explicit output-dir provided, use it; otherwise infer from model path
    if args.output_dir:
        FIGURE_OUTPUT_DIR = Path(args.output_dir)
    else:
        model_path_lower = str(_MODEL_PATH).lower()
        model_basename = os.path.basename(_MODEL_PATH).lower()
        if (
            "efficientnet" in model_basename
            or "efficientnetv2s" in model_path_lower
            or "efficientnetv2" in model_path_lower
        ):
            # Try to determine B0 vs S variant from the path
            if (
                "efficientnetv2b0" in model_path_lower
                or "-b0" in model_path_lower
            ):
                FIGURE_OUTPUT_DIR = backbone_plot_dir("EfficientNetV2-B0")
            elif (
                "efficientnetv2s" in model_path_lower
                or "-s" in model_path_lower
            ):
                FIGURE_OUTPUT_DIR = backbone_plot_dir("EfficientNetV2-S")
            else:
                FIGURE_OUTPUT_DIR = backbone_plot_dir("EfficientNetV2")
        else:
            FIGURE_OUTPUT_DIR = backbone_plot_dir("DINOv3")

    print("generating plots...")
    generate_class_distribution()
    generate_learning_curves_from_logs()
    generate_model_architecture_diagram()
    generate_confusion_matrix()
    generate_sample_predictions()
    print("All figures generated successfully.")
    print(f"Output directories: {OTHERS_PLOTS_DIR}/ and {FIGURE_OUTPUT_DIR}/")
    print("  - class_distribution.png")
    print("  - learning_curves.png")
    print("  - training_timeline_overview.png")
    print("  - model_architecture.png")
    print("  - confusion_matrix.png")
    print("  - sample_predictions.png")


if __name__ == "__main__":
    main()
