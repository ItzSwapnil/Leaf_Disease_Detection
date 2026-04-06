from __future__ import annotations

import math
from typing import Callable

import numpy as np
import tensorflow as tf


def _to_label_indices(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim == 1:
        return arr.astype(np.int64)
    return np.argmax(arr, axis=1).astype(np.int64)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute ECE/MCE/Brier and per-bin reliability statistics."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array with shape [N, C].")

    y_true = _to_label_indices(labels)
    if y_true.shape[0] != probs.shape[0]:
        raise ValueError("labels and probs must have matching first dimension.")

    n_bins = max(2, int(n_bins))
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=True maps confidence=1.0 to the last bin.
    bin_ids = np.clip(np.digitize(confidences, bin_edges, right=True) - 1, 0, n_bins - 1)

    bin_accuracy: list[float] = []
    bin_confidence: list[float] = []
    bin_counts: list[int] = []

    ece = 0.0
    mce = 0.0
    total = float(len(y_true))

    for idx in range(n_bins):
        mask = bin_ids == idx
        count = int(np.sum(mask))
        if count == 0:
            bin_accuracy.append(0.0)
            bin_confidence.append(0.0)
            bin_counts.append(0)
            continue

        acc = float(np.mean(correct[mask]))
        conf = float(np.mean(confidences[mask]))
        gap = abs(acc - conf)

        bin_accuracy.append(acc)
        bin_confidence.append(conf)
        bin_counts.append(count)

        weight = float(count) / total
        ece += weight * gap
        mce = max(mce, gap)

    one_hot = np.eye(probs.shape[1], dtype=np.float64)[y_true]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": brier,
        "bin_edges": bin_edges.tolist(),
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_counts": bin_counts,
    }


def prediction_entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Shannon entropy in bits for each prediction row."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array with shape [N, C].")
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log2(p), axis=1)


def confidence_rejection_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """Compute coverage and accepted-set accuracy for confidence thresholding."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = _to_label_indices(labels)

    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accepted = conf >= float(threshold)
    accepted_count = int(np.sum(accepted))

    if accepted_count == 0:
        accepted_acc = 0.0
    else:
        accepted_acc = float(np.mean(preds[accepted] == y_true[accepted]))

    coverage = float(accepted_count / max(1, len(y_true)))
    rejection_rate = 1.0 - coverage

    return {
        "threshold": float(threshold),
        "coverage": coverage,
        "accepted_accuracy": accepted_acc,
        "rejection_rate": rejection_rate,
        "accepted_count": accepted_count,
        "total_count": int(len(y_true)),
    }


def entropy_rejection_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """Compute rejection metrics using entropy thresholding (reject high-entropy)."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = _to_label_indices(labels)

    entropy = prediction_entropy(probs)
    preds = np.argmax(probs, axis=1)
    accepted = entropy <= float(threshold)
    accepted_count = int(np.sum(accepted))

    if accepted_count == 0:
        accepted_acc = 0.0
    else:
        accepted_acc = float(np.mean(preds[accepted] == y_true[accepted]))

    coverage = float(accepted_count / max(1, len(y_true)))
    rejection_rate = 1.0 - coverage

    return {
        "threshold_bits": float(threshold),
        "coverage": coverage,
        "accepted_accuracy": accepted_acc,
        "rejection_rate": rejection_rate,
        "accepted_count": accepted_count,
        "total_count": int(len(y_true)),
        "mean_entropy_bits": float(np.mean(entropy)),
    }


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling to logits and return calibrated probabilities."""
    logits = np.asarray(logits, dtype=np.float64)
    temp = max(1e-6, float(temperature))
    scaled = logits / temp
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_scores = np.exp(scaled)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def optimize_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    steps: int = 400,
    learning_rate: float = 0.01,
) -> dict:
    """Fit a scalar temperature parameter on validation logits."""
    logits_np = np.asarray(logits, dtype=np.float32)
    y_true = _to_label_indices(labels).astype(np.int32)

    logits_tf = tf.convert_to_tensor(logits_np, dtype=tf.float32)
    labels_tf = tf.convert_to_tensor(y_true, dtype=tf.int32)

    # softplus(log_temp) guarantees a strictly positive temperature.
    log_temp = tf.Variable(math.log(math.e - 1.0), dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))

    def _nll(current_logits: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels_tf,
                current_logits,
                from_logits=True,
            )
        )

    nll_before = float(_nll(logits_tf).numpy())

    for _ in range(max(1, int(steps))):
        with tf.GradientTape() as tape:
            temperature = tf.nn.softplus(log_temp) + 1e-6
            scaled_logits = logits_tf / temperature
            loss = _nll(scaled_logits)
        grad = tape.gradient(loss, [log_temp])
        optimizer.apply_gradients(zip(grad, [log_temp]))

    temperature_value = float((tf.nn.softplus(log_temp) + 1e-6).numpy())
    nll_after = float(_nll(logits_tf / temperature_value).numpy())

    return {
        "temperature": temperature_value,
        "nll_before": nll_before,
        "nll_after": nll_after,
    }


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval with percentile bounds."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")

    n_samples = y_true.shape[0]
    n_boot = max(100, int(n_boot))
    rng = np.random.default_rng(int(seed))

    values = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, n_samples, size=n_samples)
        values[idx] = float(metric_fn(y_true[sample_idx], y_pred[sample_idx]))

    lower, upper = np.percentile(values, [2.5, 97.5])
    return {
        "mean": float(np.mean(values)),
        "lower": float(lower),
        "upper": float(upper),
        "n_boot": int(n_boot),
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar test comparing two classifiers on paired predictions.

    Model A is treated as the proposed model and Model B as the baseline.
    """
    y_true = _to_label_indices(y_true)
    pred_a = _to_label_indices(pred_a)
    pred_b = _to_label_indices(pred_b)

    if not (y_true.shape[0] == pred_a.shape[0] == pred_b.shape[0]):
        raise ValueError("Input arrays must have the same number of samples.")

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    n00 = int(np.sum(a_correct & b_correct))
    n01 = int(np.sum(~a_correct & b_correct))
    n10 = int(np.sum(a_correct & ~b_correct))
    n11 = int(np.sum(~a_correct & ~b_correct))

    table = np.array([[n00, n01], [n10, n11]], dtype=np.int64)

    try:
        from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

        result = sm_mcnemar(table, exact=False, correction=True)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        method = "statsmodels"
    except Exception:
        discordant = n01 + n10
        if discordant == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = float(((abs(n01 - n10) - 1.0) ** 2) / discordant)
            p_value = float(math.erfc(math.sqrt(max(0.0, statistic) / 2.0)))
        method = "chi_square_fallback"

    return {
        "table": table.tolist(),
        "n01_baseline_correct_proposed_wrong": n01,
        "n10_proposed_correct_baseline_wrong": n10,
        "statistic": statistic,
        "p_value": p_value,
        "method": method,
        "significant_at_0_05": bool(p_value < 0.05),
    }
