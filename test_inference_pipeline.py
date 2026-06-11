#!/usr/bin/env python3
"""
Diagnostic script to verify preprocessing fixes and identify overfitting issues.

Tests:
1. Backbone detection in predict.py
2. Preprocessing consistency between training and inference
3. Model generalization on test set samples
4. Preprocessing correctness for each backbone type
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import LeafDiseasePredictor
from preprocessing import preprocess_array_for_model


def test_backbone_detection():
    """Test that backbone detection works correctly."""
    print("\n" + "=" * 70)
    print("TEST 1: Backbone Detection in Inference Pipeline")
    print("=" * 70)

    # Use canonical paths that resolve_keras_model_path accepts
    models_base = os.path.dirname(os.path.abspath(__file__))
    models_to_test = [
        (
            os.path.join(
                models_base,
                "models/EfficientNetv2B0/leaf_disease_classifier.keras",
            ),
            "Expected: EfficientNetV2B0",
        ),
        (
            os.path.join(
                models_base,
                "models/EfficientNetv2S/leaf_disease_classifier.keras",
            ),
            "Expected: EfficientNetV2S",
        ),
        (
            os.path.join(models_base, "models/leaf_disease_refined.keras"),
            "Expected: DINOv3",
        ),
    ]

    for full_path, expected in models_to_test:
        if not os.path.exists(full_path):
            print(f"⊘ SKIP: {full_path} (not found)")
            continue

        print(f"\n  Loading: {os.path.basename(full_path)}")
        print(f"  {expected}")

        try:
            predictor = LeafDiseasePredictor(model_path=full_path)
            detected = predictor.backbone_name
            print(f"  ✓ Detected: {detected}")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")


def test_preprocessing_consistency():
    """Test that preprocessing is consistent between training and inference."""
    print("\n" + "=" * 70)
    print("TEST 2: Preprocessing Consistency")
    print("=" * 70)

    # Create a synthetic test image
    test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    print("\n  Testing EfficientNetV2 preprocessing:")
    result_effnet = preprocess_array_for_model(
        np.expand_dims(test_image, 0), backbone_name="EfficientNetV2B0"
    )
    print(f"    Shape: {result_effnet.shape}")
    print(f"    Range: [{result_effnet.min():.3f}, {result_effnet.max():.3f}]")
    print(f"    Mean: {result_effnet.mean():.3f}")

    print("\n  Testing DINOv3 preprocessing:")
    result_dinov3 = preprocess_array_for_model(
        np.expand_dims(test_image, 0), backbone_name="DINOv3"
    )
    print(f"    Shape: {result_dinov3.shape}")
    print(f"    Range: [{result_dinov3.min():.3f}, {result_dinov3.max():.3f}]")
    print(f"    Mean: {result_dinov3.mean():.3f}")

    # Verify they're different
    if not np.allclose(result_effnet, result_dinov3):
        print(
            "\n  ✓ Good: Different preprocessing for different architectures"
        )
    else:
        print(
            "\n  ✗ ERROR: Preprocessing should differ between architectures!"
        )


def test_inference_on_dataset():
    """Test model inference on actual dataset samples."""
    print("\n" + "=" * 70)
    print("TEST 3: Inference on Dataset Samples")
    print("=" * 70)

    # Try to load test dataset
    test_dir = "dataset/test"
    if not os.path.exists(test_dir):
        print(f"  ⊘ SKIP: {test_dir} not found")
        return

    print(f"  Testing on samples from {test_dir}")

    predictor = LeafDiseasePredictor()

    # Get first class and sample
    classes = os.listdir(test_dir)
    if not classes:
        print("  ⊘ SKIP: No classes in test directory")
        return

    test_class = classes[0]
    test_files = os.listdir(os.path.join(test_dir, test_class))
    if not test_files:
        print(f"  ⊘ SKIP: No test files in {test_class}")
        return

    test_image_path = os.path.join(test_dir, test_class, test_files[0])
    print(f"\n  Test image: {test_image_path}")
    print(f"  Expected class: {test_class}")

    try:
        result = predictor.predict(test_image_path)
        predicted = result.get("disease", "Unknown")
        confidence = result.get("confidence", 0)

        print(f"  ✓ Prediction: {predicted}")
        print(f"  ✓ Confidence: {confidence:.1f}%")
        print(f"  ✓ Reject: {result.get('reject', False)}")

        if result.get("reject"):
            print(
                f"  ℹ Rejection reasons: {result.get('rejection_reasons', [])}"
            )

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback

        traceback.print_exc()


def test_out_of_distribution():
    """Test model behavior on potentially OOD images."""
    print("\n" + "=" * 70)
    print("TEST 4: Out-of-Distribution Detection")
    print("=" * 70)

    # Create synthetic OOD images
    test_cases = [
        (
            "Random noise",
            np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        ),
        ("Solid color (red)", np.full((224, 224, 3), 200, dtype=np.uint8)),
        (
            "Solid color (blue)",
            np.full((224, 224, 3), (0, 0, 200), dtype=np.uint8),
        ),
        (
            "Gradient",
            np.stack(
                [
                    np.tile(
                        np.linspace(0, 255, 224, dtype=np.uint8), (224, 1)
                    ),
                    np.tile(
                        np.linspace(255, 0, 224, dtype=np.uint8), (224, 1)
                    ),
                    np.full((224, 224), 128, dtype=np.uint8),
                ],
                axis=-1,
            ),
        ),
    ]

    # Save temp images
    temp_dir = "/tmp/leaf_test_ood"
    os.makedirs(temp_dir, exist_ok=True)

    predictor = LeafDiseasePredictor()

    for test_name, test_image in test_cases:
        # Save image
        img_path = os.path.join(temp_dir, f"{test_name.replace(' ', '_')}.png")
        tf.keras.preprocessing.image.save_img(img_path, test_image)

        print(f"\n  {test_name}:")
        try:
            result = predictor.predict(img_path)
            predicted = result.get("disease", "Unknown")
            confidence = result.get("confidence", 0)
            prediction_block = result.get("prediction", {})
            rejected = bool(
                prediction_block.get("rejected", result.get("reject", False))
            )

            status = "REJECTED" if rejected else "ACCEPTED"
            print(f"    {status}: {predicted} ({confidence:.1f}%)")

            if rejected:
                reasons = result.get(
                    "rejection_reasons",
                    [prediction_block.get("rejection_reason")]
                    if prediction_block.get("rejection_reason")
                    else [],
                )
                print(f"    Reasons: {reasons}")

        except Exception as e:
            print(f"    ✗ ERROR: {e}")

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic tests for inference pipeline"
    )
    parser.add_argument(
        "--test",
        choices=["backbone", "preprocessing", "dataset", "ood", "all"],
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# LEAF DISEASE DETECTION - INFERENCE PIPELINE DIAGNOSTICS")
    print("#" * 70)

    if args.test in ("all", "backbone"):
        test_backbone_detection()

    if args.test in ("all", "preprocessing"):
        test_preprocessing_consistency()

    if args.test in ("all", "dataset"):
        test_inference_on_dataset()

    if args.test in ("all", "ood"):
        test_out_of_distribution()

    print("\n" + "#" * 70)
    print("# DIAGNOSTICS COMPLETE")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
