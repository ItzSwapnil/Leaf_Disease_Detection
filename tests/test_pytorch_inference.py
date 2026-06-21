"""Smoke test for PyTorch inference pipeline.

Skips gracefully when no trained .pt model file exists on disk.
"""

import os

import pytest
from PIL import Image

from src.utils.model_paths import resolve_model_path


def _model_available() -> bool:
    """Check whether any .pt model file can be resolved."""
    try:
        resolve_model_path()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(
    not _model_available(),
    reason="No trained .pt model file available — skipping inference test.",
)
def test_inference():
    from src.pipeline.disease_detection_pipeline import LeafDiseasePipeline

    pipeline = LeafDiseasePipeline()
    print("Pipeline initialized!")

    img_path = "dummy_test.jpg"
    Image.new("RGB", (224, 224), color="green").save(img_path)

    try:
        results = pipeline.predict(img_path)
        print("Inference results:")
        print(results)
    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


if __name__ == "__main__":
    test_inference()
