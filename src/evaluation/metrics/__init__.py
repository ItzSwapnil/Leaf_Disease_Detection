"""Evaluation utilities for calibration, uncertainty, and statistical testing."""

from .calibration import (
    apply_temperature,
    bootstrap_ci,
    confidence_rejection_metrics,
    entropy_rejection_metrics,
    expected_calibration_error,
    mcnemar_test,
    optimize_temperature,
    prediction_entropy,
)
from .reliability_plot import plot_reliability_diagram
from .robustness import evaluate_robustness_suite

__all__ = [
    "apply_temperature",
    "bootstrap_ci",
    "confidence_rejection_metrics",
    "entropy_rejection_metrics",
    "expected_calibration_error",
    "mcnemar_test",
    "optimize_temperature",
    "plot_reliability_diagram",
    "prediction_entropy",
    "evaluate_robustness_suite",
]
