#!/usr/bin/env python3
"""
Simple unit test for backbone detection logic without loading full models.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MockLayer:
    """Mock Keras layer for testing."""

    def __init__(self, name):
        self.name = name


class MockModel:
    """Mock Keras model for testing."""

    def __init__(self, name, layer_names):
        self.name = name
        self.layers = [MockLayer(n) for n in layer_names]


def test_backbone_detection_logic():
    """Test the backbone detection logic without loading real models."""
    print("\n" + "=" * 70)
    print("UNIT TEST: Backbone Detection Logic")
    print("=" * 70)

    # Import the detection function from predict.py
    from predict import LeafDiseasePredictor

    # Create a fake predictor with mock model
    predictor = LeafDiseasePredictor.__new__(LeafDiseasePredictor)

    # Test cases: (model_name, layer_names, expected_backbone)
    test_cases = [
        ("vit_model", ["vit_patch_embed", "vit_transformer"], "DINOv3"),
        ("dinov3_classifier", ["dinov3_patching", "attention"], "DINOv3"),
        (
            "efficientnetv2b0_model",
            ["efficientnetv2b0_stem"],
            "EfficientNetV2B0",
        ),
        (
            "efficientnetv2b0_classifier",
            ["conv2d_1", "efficientnetv2b0_block"],
            "EfficientNetV2B0",
        ),
        (
            "efficientnetv2s_model",
            ["efficientnetv2s_expansion"],
            "EfficientNetV2S",
        ),
        (
            "custom_model",
            ["conv2d", "dense"],
            "EfficientNetV2B0",
        ),  # Default fallback
    ]

    for model_name, layer_names, expected_backbone in test_cases:
        predictor.model = MockModel(model_name, layer_names)
        detected = predictor._infer_backbone_from_model()

        status = "✓" if detected == expected_backbone else "✗"
        print(f"\n  {status} Model: {model_name}")
        print(f"    Layers: {layer_names}")
        print(f"    Expected: {expected_backbone}")
        print(f"    Detected: {detected}")

        if detected != expected_backbone:
            print("    ERROR: Mismatch!")


def test_preprocessing_logic():
    """Test preprocessing with detected backbone."""
    print("\n" + "=" * 70)
    print("UNIT TEST: Preprocessing with Backbone Detection")
    print("=" * 70)

    import numpy as np

    from preprocessing import preprocess_array_for_model

    # Create synthetic test image
    test_image = np.ones((1, 224, 224, 3), dtype=np.uint8) * 128

    print("\n  Testing EfficientNetV2 preprocessing:")
    result_effnet = preprocess_array_for_model(
        test_image, backbone_name="EfficientNetV2B0"
    )
    print(f"    Shape: {result_effnet.shape}")
    print(
        f"    Min: {result_effnet.min():.4f}, Max: {result_effnet.max():.4f}"
    )
    print(f"    Mean: {result_effnet.mean():.4f}")

    print("\n  Testing DINOv3 preprocessing:")
    result_dinov3 = preprocess_array_for_model(
        test_image, backbone_name="DINOv3"
    )
    print(f"    Shape: {result_dinov3.shape}")
    print(
        f"    Min: {result_dinov3.min():.4f}, Max: {result_dinov3.max():.4f}"
    )
    print(f"    Mean: {result_dinov3.mean():.4f}")

    print("\n  Comparing preprocessing methods:")
    if np.allclose(result_effnet, result_dinov3):
        print(
            "    ✗ ERROR: Preprocessing should differ between architectures!"
        )
    else:
        print(
            "    ✓ Good: Different preprocessing for different architectures"
        )

    # Check that preprocessing produces different results
    max_diff = np.abs(result_effnet - result_dinov3).max()
    print(f"    Max difference: {max_diff:.4f}")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# BACKBONE DETECTION UNIT TESTS")
    print("#" * 70)

    try:
        test_backbone_detection_logic()
    except Exception as e:
        print(f"\n✗ BACKBONE TEST FAILED: {e}")
        import traceback

        traceback.print_exc()

    try:
        test_preprocessing_logic()
    except Exception as e:
        print(f"\n✗ PREPROCESSING TEST FAILED: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "#" * 70)
    print("# UNIT TESTS COMPLETE")
    print("#" * 70 + "\n")
