"""
Quick-start guide for training the binary leaf detector model.

This script demonstrates how to train Stage 1 of the multi-stage pipeline.
"""

from src.training.train_leaf_detector import (
    BINARY_DETECTOR_FINAL,
    train_leaf_detector,
)


def _print_model_saved_info() -> None:
    """Print info about saved model."""
    if BINARY_DETECTOR_FINAL.exists():
        file_size_mb = BINARY_DETECTOR_FINAL.stat().st_size / (1024 * 1024)
        print(f"\n✓ Model successfully saved ({file_size_mb:.1f} MB)")
        print(f"  Path: {BINARY_DETECTOR_FINAL}")
        print("\nTo use this model in the pipeline:")
        print(
            "  1. Run your pipeline - it will automatically detect the model"
        )
        print("  2. Or explicitly pass it: ")
        print(
            "     from src.pipeline.disease_detection_pipeline import create_pipeline"
        )
        print("     pipeline = create_pipeline()")
        print(
            "\n✓ The pipeline will now use the trained model for leaf detection!"
        )
    else:
        print("[ERROR] Model file not found after training")


def main():
    """Train the leaf detector model."""
    print("\n" + "=" * 70)
    print("LEAF DETECTOR MODEL TRAINING")
    print("=" * 70)
    print("""
This script trains a binary classifier to distinguish between:
  ✓ Leaf images (from your disease dataset - all positives)
  ✓ Non-leaf images (backgrounds, text, etc. - currently not used)

The model uses the same EfficientNetV2B0 backbone as your disease classifier
and is trained in two phases:
  Phase 1: Frozen backbone, train head only (5 epochs)
  Phase 2: Full fine-tuning (5 epochs)

Time estimate: 5-10 minutes on GPU
Memory: ~4-6 GB VRAM

Output models saved to:
  - Checkpoint: models/leaf_detector_checkpoint.keras
  - Final:      models/leaf_detector_final.keras
""")

    input("\nPress Enter to start training... ")

    # Train
    model = train_leaf_detector()

    if model is None:
        print("\n[ERROR] Training failed. Check the error messages above.")
        return

    # Verify
    _print_model_saved_info()


if __name__ == "__main__":
    main()
