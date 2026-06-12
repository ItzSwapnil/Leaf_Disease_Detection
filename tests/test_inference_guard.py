import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.inference_guard import (
    compute_prediction_diagnostics,
    evaluate_inference_safety,
)


def test_compute_prediction_diagnostics_reports_topk_margin_entropy():
    diagnostics = compute_prediction_diagnostics(
        np.array([0.8, 0.1, 0.1], dtype=np.float64)
    )

    assert diagnostics["top1_index"] == 0
    assert diagnostics["top2_index"] in {1, 2}
    assert np.isclose(diagnostics["top1_prob"], 0.8)
    assert np.isclose(diagnostics["confidence_margin"], 0.7)
    assert 0.0 <= diagnostics["entropy_bits"] <= np.log2(3.0)
    assert 0.0 <= diagnostics["entropy_ratio"] <= 1.0


def test_evaluate_inference_safety_rejects_non_leaf_signal():
    diagnostics = compute_prediction_diagnostics(
        np.array([0.97, 0.02, 0.01], dtype=np.float64)
    )
    leaf_validation = {"leaf_score": 0.1, "vegetation_ratio": 0.01}

    decision = evaluate_inference_safety(
        diagnostics=diagnostics,
        leaf_validation=leaf_validation,
        confidence_threshold=0.92,
        entropy_threshold_bits=0.7,
        msp_threshold=0.75,
    )

    assert decision["reject"] is True
    assert "image_appears_non_leaf" in decision["reasons"]


def test_evaluate_inference_safety_rejects_uncertain_predictions():
    diagnostics = compute_prediction_diagnostics(
        np.array([0.37, 0.33, 0.30], dtype=np.float64)
    )
    leaf_validation = {"leaf_score": 0.85, "vegetation_ratio": 0.42}

    decision = evaluate_inference_safety(
        diagnostics=diagnostics,
        leaf_validation=leaf_validation,
        confidence_threshold=0.92,
        entropy_threshold_bits=0.7,
        msp_threshold=0.75,
    )

    assert decision["reject"] is True
    assert decision["uncertainty_score"] >= 2
    assert "high_entropy" in decision["reasons"]


def test_evaluate_inference_safety_accepts_confident_leaf_prediction():
    diagnostics = compute_prediction_diagnostics(
        np.array([0.95, 0.03, 0.02], dtype=np.float64)
    )
    leaf_validation = {"leaf_score": 0.74, "vegetation_ratio": 0.34}

    decision = evaluate_inference_safety(
        diagnostics=diagnostics,
        leaf_validation=leaf_validation,
        confidence_threshold=0.92,
        entropy_threshold_bits=0.7,
        msp_threshold=0.75,
    )

    assert decision["reject"] is False
    assert decision["uncertainty_score"] <= 1


def test_evaluate_inference_safety_accepts_real_world_leaf_like_image():
    diagnostics = compute_prediction_diagnostics(
        np.array([0.88, 0.07, 0.05], dtype=np.float64)
    )
    leaf_validation = {"leaf_score": 0.24, "vegetation_ratio": 0.06}

    decision = evaluate_inference_safety(
        diagnostics=diagnostics,
        leaf_validation=leaf_validation,
        confidence_threshold=0.85,
        entropy_threshold_bits=0.8,
        msp_threshold=0.7,
    )

    assert decision["reject"] is False
