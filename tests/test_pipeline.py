"""Test the multi-stage disease detection pipeline."""

from pathlib import Path

from src.pipeline.disease_detection_pipeline import create_pipeline


def _print_prediction(pred: dict, success: bool) -> None:
    """Print prediction result in formatted manner."""
    status_msg = (
        "✓ CLASSIFICATION SUCCESSFUL" if success else "[Model Best Guess]"
    )
    print(f"\n{status_msg}")
    print(f"  Class: {pred['class_name']}")
    print(f"  Confidence: {pred['confidence']:.1%}")
    if "disease_info" in pred:
        print(f"  Plant: {pred['disease_info']['plant']}")
        print(f"  Disease: {pred['disease_info']['disease']}")


def test_pipeline_on_sample_image():
    """Test the full pipeline on a sample image."""
    print("=" * 70)
    print("MULTI-STAGE LEAF DISEASE DETECTION PIPELINE TEST")
    print("=" * 70)

    # Find a test image
    dataset_dir = Path("dataset/test")
    if not dataset_dir.exists():
        print(f"[ERROR] Test dataset not found at {dataset_dir}")
        return

    test_images = list(dataset_dir.rglob("*.jpg")) + list(
        dataset_dir.rglob("*.png")
    )
    if not test_images:
        print(f"[ERROR] No test images found in {dataset_dir}")
        return

    test_image = test_images[0]
    print(f"\nTest Image: {test_image}")
    print(f"File Size: {test_image.stat().st_size / 1024:.1f} KB\n")

    # Create pipeline
    print("[*] Loading pipeline...")
    try:
        pipeline = create_pipeline()
    except Exception as e:
        print(f"[ERROR] Failed to create pipeline: {e}")
        return

    # Run prediction
    print("[*] Running prediction...\n")
    result = pipeline.predict(str(test_image))

    # Display results
    print("PIPELINE EXECUTION RESULTS")
    print("-" * 70)

    for stage_name, stage_result in result.get("pipeline_stages", []):
        print(f"\n[STAGE] {stage_name.upper()}")
        print("-" * 70)

        if isinstance(stage_result, dict):
            for key, value in stage_result.items():
                if key == "mask":  # Skip large arrays
                    continue
                if key == "preprocessed_image":  # Skip large arrays
                    continue

                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    - {k}: {v}")
                else:
                    print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("FINAL PREDICTION")
    print("=" * 70)

    if result.get("success"):
        _print_prediction(result["final_prediction"], success=True)
    else:
        print("\n✗ CLASSIFICATION FAILED")
        print(f"  Reason: {result.get('rejection_reason', 'Unknown')}")

        if result.get("final_prediction"):
            _print_prediction(result["final_prediction"], success=False)

    print("\n")


if __name__ == "__main__":
    test_pipeline_on_sample_image()
