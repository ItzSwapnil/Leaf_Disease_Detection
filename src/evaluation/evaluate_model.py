from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)

# Add project root to sys.path to support running directly as a script
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluation.metrics.calibration import (
    expected_calibration_error,
    confidence_rejection_metrics,
    entropy_rejection_metrics,
    apply_temperature,
)
from src.evaluation.metrics.robustness import evaluate_robustness_suite
from src.pipeline.predict import _load_model_robust
from src.training.training_utils import build_dynamic_yolo_dataset
from src.utils.config import BATCH_SIZE, VAL_DIR


def run_bootstrap_ci(
    logits: np.ndarray,
    labels: np.ndarray,
    temp: float,
    n_iterations: int = 200,
    seed: int = 42,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    n_samples = len(labels)
    accs = []
    f1s = []
    eces = []

    # Scale probs with temperature
    logits_scaled = logits / temp
    exp_logits_scaled = np.exp(
        logits_scaled - np.max(logits_scaled, axis=1, keepdims=True)
    )
    probs_scaled = exp_logits_scaled / np.sum(
        exp_logits_scaled, axis=1, keepdims=True
    )

    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_logits = logits[indices]
        boot_labels = labels[indices]
        boot_probs_scaled = probs_scaled[indices]

        preds = np.argmax(boot_logits, axis=1)
        acc = accuracy_score(boot_labels, preds)
        _, _, f1, _ = precision_recall_fscore_support(
            boot_labels, preds, average="macro", zero_division=0
        )

        # Calculate ECE
        bin_edges = np.linspace(0.0, 1.0, 11)
        confidences = np.max(boot_probs_scaled, axis=1)
        predictions = np.argmax(boot_probs_scaled, axis=1)
        correct = (predictions == boot_labels).astype(np.float64)
        bin_ids = np.clip(
            np.digitize(confidences, bin_edges, right=True) - 1, 0, 9
        )
        ece = 0.0
        for idx in range(10):
            mask = bin_ids == idx
            count = np.sum(mask)
            if count == 0:
                continue
            acc_bin = np.mean(correct[mask])
            conf_bin = np.mean(confidences[mask])
            ece += (count / n_samples) * abs(acc_bin - conf_bin)

        accs.append(float(acc))
        f1s.append(float(f1))
        eces.append(float(ece))

    return {
        "accuracy": [
            float(np.percentile(accs, 2.5)),
            float(np.percentile(accs, 97.5)),
        ],
        "f1": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "ece": [
            float(np.percentile(eces, 2.5)),
            float(np.percentile(eces, 97.5)),
        ],
    }


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Evaluate leaf disease PyTorch model."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to the .pt model to evaluate.",
    )
    args: argparse.Namespace = parser.parse_args()

    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # CPU Thread optimization
    from src.utils.config import INTRA_OP_THREADS, INTER_OP_THREADS

    torch.set_num_threads(INTRA_OP_THREADS)
    try:
        torch.set_num_interop_threads(INTER_OP_THREADS)
    except RuntimeError:
        pass

    # CUDA benchmark & TF32 optimizations for RTX GPUs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Load Model
    model: torch.nn.Module
    backbone_name: str
    model, backbone_name = _load_model_robust(args.model_path)
    model.to(device)
    model.eval()

    # Load Dataset
    val_class_names: list[str] = sorted(
        entry.name for entry in os.scandir(VAL_DIR) if entry.is_dir()
    )
    _skip_yolo: bool = "dinov3" in backbone_name.lower()

    val_loader = build_dynamic_yolo_dataset(
        VAL_DIR,
        val_class_names,
        int(BATCH_SIZE),
        shuffle=False,
        use_yolo=not _skip_yolo,
    )

    y_true: list[int] = []
    all_logits: list[np.ndarray] = []
    all_images: list[np.ndarray] = []

    print("Starting Evaluation...")
    device_type = device.type if device.type in ("cuda", "cpu") else "cuda"
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            for images, _, labels in val_loader:
                images_dev = images.to(device, non_blocking=True)
                outputs = model(images_dev)
                disease_out = (
                    outputs["disease_output"]
                    if isinstance(outputs, dict)
                    else outputs
                )
                all_logits.append(disease_out.cpu().float().numpy())
                y_true.extend(labels.cpu().numpy())
                all_images.append(images.numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.array(y_true)
    images_all = np.concatenate(all_images, axis=0)

    # Compute uncalibrated probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs_uncal = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    y_pred = np.argmax(probs_uncal, axis=1)

    print("\n--- Classification Report ---")
    report_str = classification_report(
        labels, y_pred, target_names=val_class_names, zero_division=0
    )
    print(report_str)

    # Compute core metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(labels, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    # Load temperature parameter
    temperature = getattr(model, "temperature", 1.0)
    print(f"Model Temperature Parameter: {temperature:.4f}")

    # Compute calibration metrics
    print("Computing calibration metrics...")
    uncal_calib = expected_calibration_error(probs_uncal, labels, n_bins=10)

    probs_cal = apply_temperature(logits, temperature)
    cal_calib = expected_calibration_error(probs_cal, labels, n_bins=10)

    # Run Bootstrap CIs
    print("Running bootstrap confidence intervals...")
    bootstrap_cis = run_bootstrap_ci(logits, labels, temperature)

    # Compute Rejection Curves
    print("Computing rejection metrics...")
    coverages_conf = []
    accuracies_conf = []
    coverages_ent = []
    accuracies_ent = []

    for thresh in np.linspace(0.0, 0.99, 50):
        res_conf = confidence_rejection_metrics(probs_cal, labels, thresh)
        coverages_conf.append(float(res_conf["coverage"]))
        accuracies_conf.append(float(res_conf["accepted_accuracy"]))

        res_ent = entropy_rejection_metrics(probs_cal, labels, thresh)
        coverages_ent.append(float(res_ent["coverage"]))
        accuracies_ent.append(float(res_ent["accepted_accuracy"]))

    # Evaluate Robustness Suite on a subset to ensure fast execution
    print("Running robustness perturbations evaluation...")
    subset_size = min(200, len(images_all))
    subset_indices = np.random.default_rng(42).choice(
        len(images_all), size=subset_size, replace=False
    )
    images_subset = images_all[subset_indices]
    labels_subset = labels[subset_indices]

    def predictor_fn(imgs: np.ndarray) -> np.ndarray:
        model.eval()
        preds = []
        # Transpose NHWC [B, H, W, C] back to NCHW [B, C, H, W] for PyTorch
        imgs_nchw = np.transpose(imgs, (0, 3, 1, 2))
        with torch.no_grad():
            with torch.amp.autocast(device_type=device_type, dtype=dtype):
                for i in range(0, len(imgs_nchw), BATCH_SIZE):
                    batch = torch.from_numpy(
                        imgs_nchw[i:i + BATCH_SIZE]
                    ).to(device)
                    outputs = model(batch)
                    disease_out = (
                        outputs["disease_output"]
                        if isinstance(outputs, dict)
                        else outputs
                    )
                    logits_b = disease_out.cpu().float().numpy()
                    probs_b = apply_temperature(logits_b, temperature)
                    preds.append(probs_b)
        return np.concatenate(preds, axis=0)

    # Convert NHWC format expected by robustness suite:
    # PyTorch inputs are NCHW, transpose to NHWC for the suite
    images_nhwc = np.transpose(images_subset, (0, 2, 3, 1))

    rob_suite = evaluate_robustness_suite(
        predictor=predictor_fn,
        images=images_nhwc,
        labels=labels_subset,
        blur_sigmas=[0.0, 1.0, 2.0, 3.0],
        brightness_factors=[0.6, 0.8, 1.0, 1.2, 1.4],
        noise_sigmas=[0.0, 0.05, 0.1, 0.15],
        fog_levels=[0.0, 0.1, 0.2, 0.3],
        occlusion_fracs=[0.0, 0.1, 0.2, 0.3],
    )

    # Save to reports/evaluation_report.json
    report_data = {
        "metrics": metrics,
        "bootstrap_confidence_intervals": bootstrap_cis,
        "calibration": {
            "uncalibrated": {
                "ece": float(uncal_calib["ece"]),
                "mce": float(uncal_calib["mce"]),
                "brier": float(uncal_calib["brier"]),
                "bin_edges": uncal_calib["bin_edges"],
                "bin_accuracy": uncal_calib["bin_accuracy"],
                "bin_confidence": uncal_calib["bin_confidence"],
                "bin_counts": uncal_calib["bin_counts"],
            },
            "temperature_scaled": {
                "ece": float(cal_calib["ece"]),
                "mce": float(cal_calib["mce"]),
                "brier": float(cal_calib["brier"]),
                "bin_edges": cal_calib["bin_edges"],
                "bin_accuracy": cal_calib["bin_accuracy"],
                "bin_confidence": cal_calib["bin_confidence"],
                "bin_counts": cal_calib["bin_counts"],
            },
            "temperature_scaling": float(temperature),
        },
        "rejection": {
            "confidence": {
                "coverages": coverages_conf,
                "accuracies": accuracies_conf,
            },
            "entropy": {
                "coverages": coverages_ent,
                "accuracies": accuracies_ent,
            },
        },
        "robustness": rob_suite,
    }

    report_dir = os.path.join(ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "evaluation_report.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"Evaluation report successfully saved to: {report_path}")


if __name__ == "__main__":
    main()
