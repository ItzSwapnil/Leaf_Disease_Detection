# ruff: noqa: I001
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow as tf
import tensorflow.keras as keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.saliency_alignment import SaliencyAlignedModel
from src.pipeline.disease_detection_pipeline import LeafDiseasePipeline
from src.training import training_utils
from src.utils.config import IMG_SIZE


class _FixedFocusDetector:
    def get_focus_mask(self, image: np.ndarray) -> np.ndarray:
        mask = np.zeros((*image.shape[:2], 1), dtype=np.float32)
        mask[8:24, 10:30, 0] = 1.0
        return mask


class SpyImageModel(keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.seen_input = None

    def call(self, inputs, training=None):
        del training
        self.seen_input = inputs
        return inputs


def test_dynamic_yolo_focus_preserves_original_pixels(tmp_path, monkeypatch):
    """YOLO focus generation must not alter RGB pixels outside the bbox."""
    image_path = tmp_path / "leaf.png"
    bgr_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    bgr_image[:, :] = np.array([7, 19, 31], dtype=np.uint8)
    bgr_image[0, 0] = np.array([5, 17, 29], dtype=np.uint8)
    cv2.imwrite(str(image_path), bgr_image)

    monkeypatch.setattr(training_utils, "_yolo_leaf_detector", _FixedFocusDetector())

    image_tensor, yolo_mask = training_utils._dynamic_yolo_focus(
        tf.constant(str(image_path))
    )

    assert image_tensor.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert yolo_mask.shape == (IMG_SIZE, IMG_SIZE, 1)
    assert image_tensor.dtype == np.float32
    assert yolo_mask.dtype == np.float32
    assert np.array_equal(image_tensor[0, 0], np.array([29, 17, 5], dtype=np.float32))
    assert yolo_mask[0, 0, 0] == 0.0
    assert yolo_mask[16, 20, 0] == 1.0


def test_build_dynamic_yolo_dataset_returns_tuple_inputs(tmp_path, monkeypatch):
    """Dynamic YOLO datasets must emit ((image, focus_mask), label)."""
    class_dir = tmp_path / "Apple___healthy"
    class_dir.mkdir()
    image_path = class_dir / "sample.png"
    cv2.imwrite(str(image_path), np.full((20, 20, 3), 96, dtype=np.uint8))

    def fake_dynamic_yolo_focus_tf(path):
        del path
        image_tensor = tf.ones((IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32) * 96.0
        yolo_mask = tf.ones((IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32)
        return image_tensor, yolo_mask

    monkeypatch.setattr(
        training_utils, "dynamic_yolo_focus_tf", fake_dynamic_yolo_focus_tf
    )

    dataset = training_utils.build_dynamic_yolo_dataset(
        tmp_path,
        ["Apple___healthy"],
        batch_size=1,
        shuffle=False,
    )
    (image_batch, mask_batch), label_batch = next(iter(dataset))

    assert image_batch.shape == (1, IMG_SIZE, IMG_SIZE, 3)
    assert mask_batch.shape == (1, IMG_SIZE, IMG_SIZE, 1)
    assert label_batch.shape == (1, 1)
    assert float(image_batch[0, 0, 0, 0].numpy()) == 96.0
    assert float(mask_batch[0, 0, 0, 0].numpy()) == 1.0


def test_saliency_aligned_model_forwards_only_image_tensor():
    """The wrapped classifier should receive the image tensor, not the mask tuple."""
    functional_model = SpyImageModel()
    wrapped_model = SaliencyAlignedModel(functional_model)

    image_tensor = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    yolo_mask = tf.ones((1, IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32)

    output_tensor = wrapped_model((image_tensor, yolo_mask), training=False)

    assert output_tensor.shape == image_tensor.shape
    assert functional_model.seen_input is image_tensor


def test_pipeline_classification_receives_unmasked_image(tmp_path):
    """Pipeline stage 3 must receive original resized RGB pixels."""
    image_path = tmp_path / "leaf.png"
    rgb_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    rgb_image[:, :] = np.array([21, 43, 65], dtype=np.uint8)
    rgb_image[0, 0] = np.array([123, 45, 67], dtype=np.uint8)
    cv2.imwrite(str(image_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

    pipeline = LeafDiseasePipeline.__new__(LeafDiseasePipeline)
    pipeline.use_yolo = False
    pipeline.yolo_leaf_detector = None
    pipeline.stage_1_detect_leaf = lambda _path: {
        "is_leaf": True,
        "leaf_score": 1.0,
        "reason": "synthetic",
    }
    pipeline.stage_4_disease_analysis = lambda _class_name: {}

    captured: dict[str, np.ndarray] = {}

    def capture_classification(img_array: np.ndarray):
        captured["image"] = img_array.copy()
        return "Apple___healthy", 0.99, {"entropy_bits": 0.0}

    pipeline.stage_3_classify_leaf = capture_classification

    result = pipeline.predict(str(image_path), skip_safety_check=True)

    assert result["success"] is True
    assert captured["image"].shape == (IMG_SIZE, IMG_SIZE, 3)
    assert np.array_equal(captured["image"][0, 0], np.array([123, 45, 67]))
    assert np.array_equal(captured["image"][10, 10], np.array([21, 43, 65]))
