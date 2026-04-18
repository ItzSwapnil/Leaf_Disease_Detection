from __future__ import annotations

import argparse
import gc
import json
import math
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tensorflow.keras.models import load_model

from config import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CALIBRATION_BINS,
    CONFIDENCE_REJECT_THRESHOLD,
    ENSEMBLE_MODEL_PATHS,
    ENTROPY_REJECT_THRESHOLD,
    FINAL_MODEL_PATH,
    IMG_SIZE,
    MC_DROPOUT_ENABLED,
    MC_DROPOUT_MAX_SAMPLES,
    MC_DROPOUT_PASSES,
    MCNEMAR_BASELINE_MODEL_PATH,
    OOD_DIR,
    OOD_MAHALANOBIS_REG,
    OOD_MAX_SAMPLES,
    OOD_MSP_THRESHOLD,
    ROBUSTNESS_BLUR_SIGMAS,
    ROBUSTNESS_BRIGHTNESS_FACTORS,
    ROBUSTNESS_EVAL_ENABLED,
    ROBUSTNESS_FOG_LEVELS,
    ROBUSTNESS_MAX_SAMPLES,
    ROBUSTNESS_NOISE_SIGMAS,
    ROBUSTNESS_OCCLUSION_FRACS,
    ROBUSTNESS_SEED,
    SAVE_LOG_ARCHIVE,
    TEMPERATURE_SCALING_LR,
    TEMPERATURE_SCALING_STEPS,
    TEST_DIR,
    VAL_DIR,
)
from evaluation.calibration import (
    apply_temperature,
    bootstrap_ci,
    confidence_rejection_metrics,
    entropy_rejection_metrics,
    expected_calibration_error,
    mcnemar_test,
    optimize_temperature,
    prediction_entropy,
)
from evaluation.reliability_plot import plot_reliability_diagram
from evaluation.robustness import evaluate_robustness_suite
from hardware import configure_tensorflow
from model_paths import resolve_keras_model_path
from preprocessing import preprocess_batch_for_model_tf
from training_utils import WarmupCosineSchedule

EVAL_BATCH_SIZE = 32


def _load_model(path: str):
    custom_objects = {"WarmupCosineSchedule": WarmupCosineSchedule}
    try:
        return load_model(path, custom_objects=custom_objects)
    except TypeError as exc:
        error_text = str(exc)
        if "ViTPatchingAndEmbedding" not in error_text:
            raise

        if not _patch_vit_layer_init_for_compat():
            raise RuntimeError(
                "Failed to load ViT/DINO checkpoint due to keras-hub version mismatch. "
                "Install a compatible keras-hub version or retrain with current stack."
            ) from exc

        print(
            "Detected KerasHub ViT checkpoint compatibility mismatch; "
            "retrying load with compatibility shim."
        )
        return load_model(path, custom_objects=custom_objects)


def _patch_vit_layer_init_for_compat() -> bool:
    """Patch keras-hub ViT layer init to ignore legacy serialized kwargs."""
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


def _infer_backbone_from_model(model) -> str:
    """Best-effort backbone name from loaded model/layer names."""
    model_name = str(getattr(model, "name", "")).lower()
    layer_names = [str(getattr(layer, "name", "")).lower() for layer in model.layers]
    haystack = " ".join([model_name, *layer_names])

    if "dinov3" in haystack or "vit" in haystack:
        return "DINOv3"
    if "efficientnetv2b0" in haystack:
        return "EfficientNetV2B0"
    if "efficientnetv2b1" in haystack:
        return "EfficientNetV2B1"
    if "efficientnetv2b2" in haystack:
        return "EfficientNetV2B2"
    if "efficientnetv2b3" in haystack:
        return "EfficientNetV2B3"
    if "efficientnetv2s" in haystack:
        return "EfficientNetV2S"
    if "efficientnetv2m" in haystack:
        return "EfficientNetV2M"
    if "efficientnetv2l" in haystack:
        return "EfficientNetV2L"
    return "Unknown"


def _dataset_from_directory(path: str, backbone_name: str | None = None):
    ds = keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )
    gen = ds.map(
        lambda x, y: (preprocess_batch_for_model_tf(x, backbone_name=backbone_name), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    return ds, gen


def _load_unlabeled_dataset(path: str, backbone_name: str | None = None):
    ds = keras.utils.image_dataset_from_directory(
        path,
        labels=None,
        label_mode=None,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )
    return ds.map(
        lambda x: preprocess_batch_for_model_tf(x, backbone_name=backbone_name),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


def _collect_label_indices(dataset) -> np.ndarray:
    one_hot = np.concatenate([labels.numpy() for _, labels in dataset], axis=0)
    return np.argmax(one_hot, axis=1).astype(np.int64)


def _take_dataset(dataset, max_samples: int):
    if max_samples <= 0:
        return dataset
    batches = max(1, int(math.ceil(float(max_samples) / float(EVAL_BATCH_SIZE))))
    return dataset.take(batches)


def _slice_to_limit(arr: np.ndarray, max_samples: int) -> np.ndarray:
    if max_samples <= 0:
        return arr
    return arr[:max_samples]


def _collect_dataset_arrays(dataset, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    images_parts = []
    labels_parts = []
    collected = 0
    limit = max(0, int(max_samples))

    for batch_images, batch_labels in dataset:
        images_np = np.asarray(
            batch_images.numpy() if hasattr(batch_images, "numpy") else batch_images,
            dtype=np.float32,
        )
        labels_np = np.asarray(
            batch_labels.numpy() if hasattr(batch_labels, "numpy") else batch_labels,
            dtype=np.float32,
        )

        if limit > 0 and collected >= limit:
            break

        if limit > 0:
            remaining = limit - collected
            if images_np.shape[0] > remaining:
                images_np = images_np[:remaining]
                labels_np = labels_np[:remaining]

        images_parts.append(images_np)
        labels_parts.append(labels_np)
        collected += int(images_np.shape[0])

        if limit > 0 and collected >= limit:
            break

    if not images_parts:
        empty_images = np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        empty_labels = np.empty((0,), dtype=np.int64)
        return empty_images, empty_labels

    images = np.concatenate(images_parts, axis=0)
    labels_one_hot = np.concatenate(labels_parts, axis=0)
    labels = np.argmax(labels_one_hot, axis=1).astype(np.int64)
    return images, labels


def _build_logits_model(model):
    if not model.layers:
        return None
    last_layer = model.layers[-1]
    activation_name = getattr(getattr(last_layer, "activation", None), "__name__", "")
    if activation_name == "softmax":
        return keras.Model(inputs=model.input, outputs=last_layer.input)
    return None


def _build_feature_model(model):
    if len(model.layers) < 2:
        return None
    return keras.Model(inputs=model.input, outputs=model.layers[-2].output)


def _predict_probs_and_logits(model, dataset):
    probs = np.asarray(model.predict(dataset, verbose=1), dtype=np.float64)
    logits_model = _build_logits_model(model)
    if logits_model is None:
        logits = np.log(np.clip(probs, 1e-8, 1.0))
        return probs, logits
    logits = np.asarray(logits_model.predict(dataset, verbose=0), dtype=np.float64)
    if logits.ndim != 2:
        logits = logits.reshape((logits.shape[0], -1))
    return probs, logits


def _macro_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    precision, _, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return float(precision)


def _macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return float(recall)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return float(f1)


def _top_confused_pairs(
    cm: np.ndarray, class_names: list[str], max_items: int = 15
) -> list[dict]:
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)
    flat_idx = np.argsort(cm_offdiag, axis=None)[::-1]
    pairs: list[dict] = []
    for idx in flat_idx:
        count = int(cm_offdiag.flat[idx])
        if count <= 0:
            break
        true_idx, pred_idx = np.unravel_index(idx, cm_offdiag.shape)
        pairs.append(
            {
                "true_class": class_names[int(true_idx)],
                "pred_class": class_names[int(pred_idx)],
                "count": count,
            }
        )
        if len(pairs) >= int(max_items):
            break
    return pairs


def _compute_bootstrap_intervals(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": bootstrap_ci(
            lambda yt, yp: float(accuracy_score(yt, yp)),
            y_true,
            y_pred,
            n_boot=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
        "macro_precision": bootstrap_ci(
            _macro_precision,
            y_true,
            y_pred,
            n_boot=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
        "macro_recall": bootstrap_ci(
            _macro_recall,
            y_true,
            y_pred,
            n_boot=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
        "macro_f1": bootstrap_ci(
            _macro_f1,
            y_true,
            y_pred,
            n_boot=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
    }


def _predict_features(feature_model, dataset, max_samples: int) -> np.ndarray:
    if feature_model is None:
        return np.empty((0, 0), dtype=np.float64)
    subset = _take_dataset(dataset, max_samples)
    features = np.asarray(feature_model.predict(subset, verbose=0), dtype=np.float64)
    if features.ndim != 2:
        features = features.reshape((features.shape[0], -1))
    return _slice_to_limit(features, max_samples)


def _fit_mahalanobis(
    features: np.ndarray, labels: np.ndarray, reg: float
) -> tuple[np.ndarray, np.ndarray]:
    unique_labels = np.unique(labels)
    means = np.stack(
        [features[labels == cls].mean(axis=0) for cls in unique_labels], axis=0
    )
    centered = features - means[np.searchsorted(unique_labels, labels)]
    cov = np.cov(centered, rowvar=False)
    cov = cov + np.eye(cov.shape[0], dtype=np.float64) * float(reg)
    inv_cov = np.linalg.pinv(cov)
    return means, inv_cov


def _mahalanobis_min_distance(
    features: np.ndarray, means: np.ndarray, inv_cov: np.ndarray
) -> np.ndarray:
    distances = np.empty(features.shape[0], dtype=np.float64)
    for idx, row in enumerate(features):
        diff = means - row
        per_class = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        distances[idx] = float(np.min(per_class))
    return distances


def _compute_ood_report(
    model,
    val_gen,
    y_true: np.ndarray,
    val_probs: np.ndarray,
    backbone_name: str | None = None,
) -> dict | None:
    if not os.path.isdir(OOD_DIR):
        return None

    try:
        ood_gen = _load_unlabeled_dataset(OOD_DIR, backbone_name=backbone_name)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"Failed to load OOD data: {exc}"}

    if OOD_MAX_SAMPLES > 0:
        id_probs = _slice_to_limit(val_probs, OOD_MAX_SAMPLES)
        id_labels = _slice_to_limit(y_true, OOD_MAX_SAMPLES)
    else:
        id_probs = val_probs
        id_labels = y_true

    ood_probs = np.asarray(
        model.predict(_take_dataset(ood_gen, OOD_MAX_SAMPLES), verbose=0),
        dtype=np.float64,
    )
    ood_probs = _slice_to_limit(ood_probs, OOD_MAX_SAMPLES)
    if ood_probs.size == 0:
        return {"status": "unavailable", "reason": "No OOD samples available."}

    id_msp = np.max(id_probs, axis=1)
    ood_msp = np.max(ood_probs, axis=1)
    msp_threshold = float(OOD_MSP_THRESHOLD)

    msp_ood_recall = float(np.mean(ood_msp < msp_threshold))
    msp_id_false_positive = float(np.mean(id_msp < msp_threshold))

    y_ood = np.concatenate(
        [
            np.zeros(id_msp.shape[0], dtype=np.int32),
            np.ones(ood_msp.shape[0], dtype=np.int32),
        ]
    )
    msp_scores = np.concatenate([1.0 - id_msp, 1.0 - ood_msp])
    msp_auroc = float(roc_auc_score(y_ood, msp_scores))

    feature_model = _build_feature_model(model)
    if feature_model is None:
        return {
            "status": "partial",
            "msp": {
                "threshold": msp_threshold,
                "ood_recall": msp_ood_recall,
                "id_false_positive_rate": msp_id_false_positive,
                "auroc": msp_auroc,
            },
            "reason": "Feature head unavailable for Mahalanobis distance.",
        }

    id_features = _predict_features(
        feature_model, val_gen, max_samples=id_probs.shape[0]
    )
    ood_features = _predict_features(
        feature_model, ood_gen, max_samples=ood_probs.shape[0]
    )
    means, inv_cov = _fit_mahalanobis(id_features, id_labels, reg=OOD_MAHALANOBIS_REG)

    id_dist = _mahalanobis_min_distance(id_features, means, inv_cov)
    ood_dist = _mahalanobis_min_distance(ood_features, means, inv_cov)
    mahal_threshold = float(np.percentile(id_dist, 95.0))
    mahal_ood_recall = float(np.mean(ood_dist > mahal_threshold))
    mahal_id_false_positive = float(np.mean(id_dist > mahal_threshold))

    mahal_scores = np.concatenate([id_dist, ood_dist])
    mahal_auroc = float(roc_auc_score(y_ood, mahal_scores))

    id_msp_score = 1.0 - id_msp
    ood_msp_score = 1.0 - ood_msp
    msp_mu = float(np.mean(id_msp_score))
    msp_sigma = float(np.std(id_msp_score) + 1e-8)
    md_mu = float(np.mean(id_dist))
    md_sigma = float(np.std(id_dist) + 1e-8)
    id_combo = 0.5 * ((id_msp_score - msp_mu) / msp_sigma) + 0.5 * (
        (id_dist - md_mu) / md_sigma
    )
    ood_combo = 0.5 * ((ood_msp_score - msp_mu) / msp_sigma) + 0.5 * (
        (ood_dist - md_mu) / md_sigma
    )
    combo_auroc = float(roc_auc_score(y_ood, np.concatenate([id_combo, ood_combo])))

    return {
        "status": "ok",
        "id_samples": int(id_probs.shape[0]),
        "ood_samples": int(ood_probs.shape[0]),
        "msp": {
            "threshold": msp_threshold,
            "ood_recall": msp_ood_recall,
            "id_false_positive_rate": msp_id_false_positive,
            "auroc": msp_auroc,
        },
        "mahalanobis": {
            "threshold": mahal_threshold,
            "ood_recall": mahal_ood_recall,
            "id_false_positive_rate": mahal_id_false_positive,
            "auroc": mahal_auroc,
            "regularization": float(OOD_MAHALANOBIS_REG),
        },
        "ensemble_score": {
            "auroc": combo_auroc,
            "description": "0.5 * z(MSP) + 0.5 * z(Mahalanobis)",
        },
    }


def _compute_mc_dropout_report(model, val_gen, y_true: np.ndarray) -> dict | None:
    if not MC_DROPOUT_ENABLED or MC_DROPOUT_PASSES <= 0:
        return None

    subset = _take_dataset(val_gen, MC_DROPOUT_MAX_SAMPLES)
    pass_probs = []
    for _ in range(int(MC_DROPOUT_PASSES)):
        probs_parts = []
        for batch_images, _ in subset:
            probs_parts.append(
                np.asarray(model(batch_images, training=True).numpy(), dtype=np.float64)
            )
        if not probs_parts:
            return None
        pass_probs.append(np.concatenate(probs_parts, axis=0))

    stacked = np.stack(pass_probs, axis=0)
    mean_probs = np.mean(stacked, axis=0)
    var_per_sample = np.mean(np.var(stacked, axis=0), axis=1)
    y_subset = _slice_to_limit(y_true, mean_probs.shape[0])
    y_pred = np.argmax(mean_probs, axis=1)
    correct = y_pred == y_subset
    entropy = prediction_entropy(mean_probs)

    correct_var = var_per_sample[correct]
    wrong_var = var_per_sample[~correct]

    return {
        "passes": int(MC_DROPOUT_PASSES),
        "samples": int(mean_probs.shape[0]),
        "mean_variance_correct": float(np.mean(correct_var))
        if correct_var.size
        else 0.0,
        "mean_variance_incorrect": float(np.mean(wrong_var)) if wrong_var.size else 0.0,
        "mean_entropy_correct_bits": float(np.mean(entropy[correct]))
        if np.any(correct)
        else 0.0,
        "mean_entropy_incorrect_bits": float(np.mean(entropy[~correct]))
        if np.any(~correct)
        else 0.0,
    }


def _evaluate_ensemble(
    model_paths: list[str], val_gen, y_true: np.ndarray
) -> dict | None:
    existing = [path for path in model_paths if path and os.path.exists(path)]
    if len(existing) < 2:
        return None

    probs_list = []
    for path in existing:
        member = _load_model(path)
        probs_list.append(
            np.asarray(member.predict(val_gen, verbose=0), dtype=np.float64)
        )
        del member
        gc.collect()

    mean_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    y_pred = np.argmax(mean_probs, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "model_paths": existing,
        "validation_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def _save_evaluation_report(report: dict):
    def _dict_or_empty(value):
        return value if isinstance(value, dict) else {}

    reports_dir = os.path.join("reports")
    os.makedirs(reports_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_json = os.path.join(reports_dir, "evaluation_report.json")
    archive_json = os.path.join(reports_dir, f"evaluation_report_{stamp}.json")
    latest_md = os.path.join(reports_dir, "evaluation_report.md")
    archive_md = os.path.join(reports_dir, f"evaluation_report_{stamp}.md")

    json_targets = [latest_json]
    if SAVE_LOG_ARCHIVE:
        json_targets.append(archive_json)

    for path in json_targets:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    metrics = _dict_or_empty(report.get("metrics"))
    calibration = _dict_or_empty(report.get("calibration"))
    temperature_scaling = _dict_or_empty(calibration.get("temperature_scaling"))
    uncalibrated = _dict_or_empty(calibration.get("uncalibrated"))
    calibrated = _dict_or_empty(calibration.get("temperature_scaled"))
    rejection = _dict_or_empty(report.get("rejection"))
    rejection_conf = _dict_or_empty(rejection.get("confidence"))
    rejection_entropy = _dict_or_empty(rejection.get("entropy"))
    entropy_threshold_label = (
        f"{ENTROPY_REJECT_THRESHOLD:.2f} ratio"
        if ENTROPY_REJECT_THRESHOLD <= 1.0
        else f"{ENTROPY_REJECT_THRESHOLD:.2f} bits"
    )
    uncertainty = _dict_or_empty(report.get("uncertainty"))
    ood = _dict_or_empty(report.get("ood_detection"))
    robustness = _dict_or_empty(report.get("robustness"))
    statistical_tests = _dict_or_empty(report.get("statistical_tests"))
    mcnemar = _dict_or_empty(statistical_tests.get("mcnemar"))

    md_lines = [
        "# Evaluation Report",
        "",
        f"- Generated: {report.get('generated_at')}",
        f"- Model: {report.get('model_path')}",
        f"- Validation samples: {report.get('validation_samples')}",
        f"- Classes: {report.get('num_classes')}",
        "",
        "## Aggregate Metrics",
        "",
        f"- Validation loss: {metrics.get('validation_loss', 0.0):.4f}",
        f"- Validation accuracy: {metrics.get('validation_accuracy', 0.0) * 100:.2f}%",
        f"- Macro precision: {metrics.get('macro_precision', 0.0):.4f}",
        f"- Macro recall: {metrics.get('macro_recall', 0.0):.4f}",
        f"- Macro F1: {metrics.get('macro_f1', 0.0):.4f}",
        "",
        "## Calibration",
        "",
        f"- Temperature: {temperature_scaling.get('temperature', 1.0):.4f}",
        f"- ECE (uncalibrated): {uncalibrated.get('ece', 0.0):.4f}",
        f"- ECE (temperature-scaled): {calibrated.get('ece', 0.0):.4f}",
        "",
        "## Rejection Metrics",
        "",
        f"- Confidence threshold ({CONFIDENCE_REJECT_THRESHOLD:.2f}) coverage: "
        f"{rejection_conf.get('coverage', 0.0) * 100:.2f}%",
        f"- Entropy threshold ({entropy_threshold_label}) coverage: "
        f"{rejection_entropy.get('coverage', 0.0) * 100:.2f}%",
        "",
        "## Uncertainty / OOD",
        "",
        f"- MC Dropout enabled: {'yes' if uncertainty.get('mc_dropout') else 'no'}",
        f"- OOD report status: {ood.get('status', 'unavailable')}",
        f"- Robustness report status: {robustness.get('status', 'unavailable')}",
        "",
    ]
    if robustness and robustness.get("worst_case"):
        md_lines.extend(
            [
                f"- Worst robustness accuracy drop: {robustness['worst_case'].get('accuracy_drop', 0.0) * 100:.2f}%",
                "",
            ]
        )
    if mcnemar:
        md_lines.extend(
            [
                "## McNemar Test",
                "",
                f"- n01 (baseline correct, proposed wrong): {mcnemar.get('n01_baseline_correct_proposed_wrong', 0)}",
                f"- n10 (proposed correct, baseline wrong): {mcnemar.get('n10_proposed_correct_baseline_wrong', 0)}",
                f"- p-value: {mcnemar.get('p_value', 1.0):.6f}",
                "",
            ]
        )

    md_lines.extend(["## Top Confused Class Pairs", ""])
    top_pairs = report.get("top_confused_pairs") or []
    if top_pairs:
        for pair in top_pairs[:10]:
            md_lines.append(
                f"- {pair['true_class']} -> {pair['pred_class']}: {pair['count']}"
            )
    else:
        md_lines.append("- No notable confusion pairs found.")
    md_lines.append("")
    md_content = "\n".join(md_lines)

    md_targets = [latest_md]
    if SAVE_LOG_ARCHIVE:
        md_targets.append(archive_md)

    for path in md_targets:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_content)

    print(f"Saved evaluation report: {latest_json}")
    print(f"Saved evaluation report: {latest_md}")
    if SAVE_LOG_ARCHIVE:
        print(f"Saved archive report: {archive_json}")
        print(f"Saved archive report: {archive_md}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate leaf disease model and generate metrics/reports."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Explicit model path to evaluate (e.g., models/leaf_disease_refined.keras). "
            "If omitted, uses configured FINAL_MODEL_PATH with standard resolver fallbacks."
        ),
    )
    args = parser.parse_args()

    configure_tensorflow()

    if args.model_path:
        model_path = resolve_keras_model_path([args.model_path])
    else:
        model_path = resolve_keras_model_path([FINAL_MODEL_PATH])

    print(f"Loading model: {model_path}")
    model = _load_model(model_path)
    detected_backbone = _infer_backbone_from_model(model)
    if detected_backbone == "Unknown":
        print(
            "Backbone inference failed; using default preprocessing path "
            "(EfficientNet-style normalization)."
        )
    else:
        print(f"Backbone-locked preprocessing: {detected_backbone}")

    val_ds, val_gen = _dataset_from_directory(VAL_DIR, backbone_name=detected_backbone)
    y_true = _collect_label_indices(val_ds)
    class_names = list(val_ds.class_names)

    test_ds = None
    test_gen = None
    if os.path.isdir(TEST_DIR):
        test_ds, test_gen = _dataset_from_directory(
            TEST_DIR, backbone_name=detected_backbone
        )

    print("\n--- Evaluating model on validation set ---")
    val_metrics = model.evaluate(val_gen, verbose=1, return_dict=True)
    val_loss = float(val_metrics.get("loss", 0.0))
    val_acc = float(val_metrics.get("accuracy", val_metrics.get("acc", 0.0)))

    test_loss = None
    test_acc = None
    if test_gen is not None:
        print("\n--- Evaluating model on test set ---")
        test_metrics = model.evaluate(test_gen, verbose=1, return_dict=True)
        test_loss = float(test_metrics.get("loss", 0.0))
        test_acc = float(test_metrics.get("accuracy", test_metrics.get("acc", 0.0)))

    val_probs, val_logits = _predict_probs_and_logits(model, val_gen)
    y_pred = np.argmax(val_probs, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    top_confused_pairs = _top_confused_pairs(cm, class_names)

    print(f"\nValidation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc * 100:.2f}%")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1-score: {f1:.4f}")

    uncalibrated = expected_calibration_error(
        val_probs, y_true, n_bins=CALIBRATION_BINS
    )
    temperature_info = optimize_temperature(
        val_logits,
        y_true,
        steps=TEMPERATURE_SCALING_STEPS,
        learning_rate=TEMPERATURE_SCALING_LR,
    )
    temperature = float(temperature_info["temperature"])
    scaled_probs = apply_temperature(val_logits, temperature)
    calibrated = expected_calibration_error(
        scaled_probs, y_true, n_bins=CALIBRATION_BINS
    )

    reliability_uncal_path = os.path.join("plots", "reliability_uncalibrated.png")
    reliability_cal_path = os.path.join("plots", "reliability_temperature_scaled.png")
    plot_reliability_diagram(
        uncalibrated, reliability_uncal_path, title="Reliability (Uncalibrated)"
    )
    plot_reliability_diagram(
        calibrated, reliability_cal_path, title="Reliability (Temperature Scaled)"
    )

    bootstrap_intervals = _compute_bootstrap_intervals(y_true, y_pred)

    rejection_conf = confidence_rejection_metrics(
        val_probs, y_true, CONFIDENCE_REJECT_THRESHOLD
    )
    rejection_entropy = entropy_rejection_metrics(
        val_probs, y_true, ENTROPY_REJECT_THRESHOLD
    )

    mc_dropout_report = _compute_mc_dropout_report(model, val_gen, y_true)
    ood_report = _compute_ood_report(
        model,
        val_gen,
        y_true,
        val_probs,
        backbone_name=detected_backbone,
    )

    robustness_report = None
    if ROBUSTNESS_EVAL_ENABLED:
        robust_images, robust_labels = _collect_dataset_arrays(
            val_gen, ROBUSTNESS_MAX_SAMPLES
        )
        if robust_images.size == 0:
            robustness_report = {
                "status": "unavailable",
                "reason": "No samples available for robustness suite.",
            }
        else:
            robustness_report = evaluate_robustness_suite(
                predictor=lambda arr: np.asarray(
                    model.predict(arr, batch_size=EVAL_BATCH_SIZE, verbose=0),
                    dtype=np.float64,
                ),
                images=robust_images,
                labels=robust_labels,
                blur_sigmas=tuple(float(v) for v in ROBUSTNESS_BLUR_SIGMAS),
                brightness_factors=tuple(
                    float(v) for v in ROBUSTNESS_BRIGHTNESS_FACTORS
                ),
                noise_sigmas=tuple(float(v) for v in ROBUSTNESS_NOISE_SIGMAS),
                fog_levels=tuple(float(v) for v in ROBUSTNESS_FOG_LEVELS),
                occlusion_fracs=tuple(float(v) for v in ROBUSTNESS_OCCLUSION_FRACS),
                seed=ROBUSTNESS_SEED,
            )

    mcnemar_report = None
    baseline_path = str(MCNEMAR_BASELINE_MODEL_PATH).strip()
    if baseline_path and os.path.exists(baseline_path):
        try:
            baseline_model = _load_model(baseline_path)
            baseline_probs = np.asarray(
                baseline_model.predict(val_gen, verbose=0), dtype=np.float64
            )
            baseline_pred = np.argmax(baseline_probs, axis=1)
            mcnemar_report = mcnemar_test(y_true, y_pred, baseline_pred)
        except Exception as exc:
            mcnemar_report = {"status": "error", "reason": str(exc)}

    ensemble_report = _evaluate_ensemble(list(ENSEMBLE_MODEL_PATHS), val_gen, y_true)

    per_class_metrics = {}
    for class_name in class_names:
        row = class_report.get(class_name, {})
        per_class_metrics[class_name] = {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1-score", 0.0)),
            "support": int(row.get("support", 0)),
        }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": model_path,
        "validation_samples": int(len(y_true)),
        "test_samples": int(len(test_ds)) if test_ds is not None else None,
        "num_classes": int(len(class_names)),
        "metrics": {
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        },
        "calibration": {
            "bins": int(CALIBRATION_BINS),
            "uncalibrated": uncalibrated,
            "temperature_scaling": temperature_info,
            "temperature_scaled": calibrated,
            "plots": {
                "uncalibrated": reliability_uncal_path,
                "temperature_scaled": reliability_cal_path,
            },
        },
        "bootstrap_confidence_intervals": bootstrap_intervals,
        "rejection": {
            "confidence": rejection_conf,
            "entropy": rejection_entropy,
        },
        "statistical_tests": {
            "mcnemar": mcnemar_report,
        },
        "uncertainty": {
            "mc_dropout": mc_dropout_report,
        },
        "ood_detection": ood_report,
        "robustness": robustness_report,
        "ensemble": ensemble_report,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "top_confused_pairs": top_confused_pairs,
    }

    _save_evaluation_report(report)
    gc.collect()


if __name__ == "__main__":
    main()
