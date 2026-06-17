import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pytest

from src.core.yolo_leaf import YOLOLeafDetector


def test_yolo_leaf_detector_init():
    with patch("src.core.yolo_leaf._create_yolo_model") as mock_create_model:
        detector = YOLOLeafDetector(model_path="dummy_yolo.pt")
        mock_create_model.assert_called_once_with("dummy_yolo.pt")
        assert detector.model_path == "dummy_yolo.pt"


def test_get_masked_returns_untouched_rgb_for_legacy_callers(tmp_path):
    """Legacy get_masked() must not zero background pixels."""
    img_path = tmp_path / "leaf.jpg"
    dummy_img = np.full((200, 300, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy_img)

    mock_results = MagicMock()
    mock_box = MagicMock()
    mock_box.__len__ = lambda self: 1
    mock_box.conf.cpu().numpy.return_value = np.array([0.95])
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].cpu().numpy.return_value = np.array([50, 40, 250, 180])
    mock_results.boxes = mock_box

    with patch("src.core.yolo_leaf._create_yolo_model") as mock_create_model:
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_results]
        mock_create_model.return_value = mock_model

        detector = YOLOLeafDetector(model_path="dummy_yolo.pt")
        rgb_image = detector.get_masked(img_path)

    # Full image dimensions preserved (RGB)
    assert rgb_image.shape == (200, 300, 3)
    assert np.all(rgb_image == 128)


def test_detect_returns_bbox(tmp_path):
    """detect() should return found=True with correct bbox."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_results = MagicMock()
    mock_box = MagicMock()
    mock_box.__len__ = lambda self: 1
    mock_box.conf.cpu().numpy.return_value = np.array([0.88])
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].cpu().numpy.return_value = np.array([10, 20, 80, 90])
    mock_results.boxes = mock_box

    with patch("src.core.yolo_leaf._create_yolo_model") as mock_create_model:
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_results]
        mock_create_model.return_value = mock_model

        detector = YOLOLeafDetector(model_path="dummy_yolo.pt")
        result = detector.detect(img)

    assert result["found"] is True
    assert result["bbox"] == (10, 20, 80, 90)
    assert result["confidence"] == pytest.approx(0.88, abs=0.01)


def test_detect_no_leaf():
    """detect() with no detections should return found=False."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_results = MagicMock()
    mock_results.boxes = None

    with patch("src.core.yolo_leaf._create_yolo_model") as mock_create_model:
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_results]
        mock_create_model.return_value = mock_model

        detector = YOLOLeafDetector(model_path="dummy_yolo.pt")
        result = detector.detect(img)

    assert result["found"] is False
    assert result["confidence"] == 0.0


def test_get_focus_mask():
    """get_focus_mask() should return a binary mask with 1.0 inside detected bbox."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_results = MagicMock()
    mock_box = MagicMock()
    mock_box.__len__ = lambda self: 1
    mock_box.conf.cpu().numpy.return_value = np.array([0.90])
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].cpu().numpy.return_value = np.array([10, 20, 80, 90])
    mock_results.boxes = mock_box

    with patch("src.core.yolo_leaf._create_yolo_model") as mock_create_model:
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_results]
        mock_create_model.return_value = mock_model

        detector = YOLOLeafDetector(model_path="dummy_yolo.pt")
        mask = detector.get_focus_mask(img)

    assert mask.shape == (100, 100, 1)
    assert np.all(mask[20:90, 10:80, 0] == 1.0)
    assert np.all(mask[0:20, :, 0] == 0.0)
    assert np.all(mask[:, 0:10, 0] == 0.0)


def test_get_focus_mask_falls_back_to_all_ones_when_no_leaf():
    """No detection should produce a full-image focus mask."""
    img = np.zeros((64, 48, 3), dtype=np.uint8)

    mock_results = MagicMock()
    mock_results.boxes = None

    with patch("src.core.yolo_leaf._create_yolo_model") as mock_create_model:
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_results]
        mock_create_model.return_value = mock_model

        detector = YOLOLeafDetector(model_path="dummy_yolo.pt")
        mask = detector.get_focus_mask(img)

    assert mask.shape == (64, 48, 1)
    assert mask.dtype == np.float32
    assert np.all(mask == 1.0)
