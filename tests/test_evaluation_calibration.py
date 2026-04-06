import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.calibration import (
    apply_temperature,
    bootstrap_ci,
    expected_calibration_error,
    mcnemar_test,
    optimize_temperature,
    prediction_entropy,
)


def test_expected_calibration_error_returns_complete_payload():
    probs = np.array(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.25, 0.75],
            [0.10, 0.90],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    result = expected_calibration_error(probs, labels, n_bins=5)

    assert set(result.keys()) >= {
        "ece",
        "mce",
        "brier",
        "bin_edges",
        "bin_accuracy",
        "bin_confidence",
        "bin_counts",
    }
    assert len(result["bin_accuracy"]) == 5
    assert len(result["bin_confidence"]) == 5
    assert len(result["bin_counts"]) == 5
    assert 0.0 <= result["ece"] <= 1.0


def test_temperature_scaling_functions_produce_valid_probabilities():
    logits = np.array(
        [
            [2.0, 0.0],
            [1.5, 0.1],
            [0.1, 1.3],
            [0.0, 2.2],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    fit = optimize_temperature(logits, labels, steps=80, learning_rate=0.05)
    probs = apply_temperature(logits, fit["temperature"])

    assert fit["temperature"] > 0.0
    assert probs.shape == logits.shape
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-6)


def test_bootstrap_ci_and_entropy_metrics_are_stable():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])

    ci = bootstrap_ci(
        metric_fn=lambda yt, yp: float(np.mean(yt == yp)),
        y_true=y_true,
        y_pred=y_pred,
        n_boot=500,
        seed=123,
    )

    assert ci["lower"] <= ci["mean"] <= ci["upper"]

    probs = np.array(
        [
            [0.95, 0.05],
            [0.10, 0.90],
            [0.55, 0.45],
            [0.60, 0.40],
        ]
    )
    entropy = prediction_entropy(probs)
    assert entropy.shape == (4,)
    assert np.all(entropy >= 0.0)


def test_mcnemar_test_returns_valid_statistics():
    y_true = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    pred_a = np.array([0, 1, 0, 0, 0, 1, 1, 1])  # proposed
    pred_b = np.array([0, 0, 1, 0, 1, 1, 0, 0])  # baseline

    result = mcnemar_test(y_true, pred_a, pred_b)

    assert "p_value" in result
    assert "statistic" in result
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["n01_baseline_correct_proposed_wrong"] >= 0
    assert result["n10_proposed_correct_baseline_wrong"] >= 0
