from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow.keras as keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure model_paths is imported so custom layer registration triggers
from scripts.gradcam_check import (
    _compute_deletion_drop,
    _extract_leaf_mask,
    _simple_blur,
)
from src.utils import model_paths  # noqa: F401


def test_extract_leaf_mask():
    # Create a 224x224x3 image
    # Left half: green leaf (high channel variation: 50, 200, 50)
    # Right half: gray background (no channel variation: 128, 128, 128)
    img = np.zeros((224, 224, 3), dtype=np.float32)
    img[:, :112, :] = [50.0, 200.0, 50.0]
    img[:, 112:, :] = [128.0, 128.0, 128.0]

    mask = _extract_leaf_mask(img)
    assert mask.shape == (224, 224)
    # Leaf half should be 1.0 (std deviation is std([50, 200, 50]) ~ 70.7 > 8.0)
    assert np.all(mask[:, :112] == 1.0)
    # Background half should be 0.0 (std deviation is 0.0 < 8.0)
    assert np.all(mask[:, 112:] == 0.0)


def test_simple_blur():
    # Create a simple image with a single step edge
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[:, :16, :] = 255.0

    blurred = _simple_blur(img, size=5)
    assert blurred.shape == img.shape
    # At the step boundary (x=15, 16), values should be smoothed
    assert blurred[0, 15, 0] < 255.0
    assert blurred[0, 16, 0] > 0.0


def test_compute_deletion_drop():
    # Build a simple mock model using functional API
    inputs = keras.layers.Input(shape=(224, 224, 3), dtype="float32")
    # A dummy layer that outputs a prediction
    x = keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss="categorical_crossentropy")

    # Image and Heatmap
    img = np.zeros((1, 224, 224, 3), dtype=np.float32)
    heatmap = np.zeros((224, 224), dtype=np.float32)
    # Make top-left corner high-attention
    heatmap[:50, :50] = 1.0

    drop = _compute_deletion_drop(
        model=model,
        img_array=img,
        backbone_name="EfficientNetV2B0",
        pred_idx=0,
        heatmap=heatmap,
        fraction=0.15,
    )
    assert isinstance(drop, float)
    assert 0.0 <= drop <= 1.0


def test_randomize_background_batch_tf():
    import tensorflow as tf

    from src.training.training_utils import _randomize_background_batch_tf

    # Create batch of 2 images of size 32x32x3
    # Left half: green leaf (50, 200, 50)
    # Right half: gray background (128, 128, 128)
    batch = np.zeros((2, 32, 32, 3), dtype=np.float32)
    batch[:, :, :16, :] = [50.0, 200.0, 50.0]
    batch[:, :, 16:, :] = [128.0, 128.0, 128.0]

    batch_tf = tf.convert_to_tensor(batch)
    randomized = _randomize_background_batch_tf(batch_tf).numpy()

    # The green leaf half should remain unchanged
    np.testing.assert_allclose(
        randomized[:, :, :16, :], batch[:, :, :16, :], rtol=1e-5
    )
    # The gray background half should be changed to random colors
    assert not np.allclose(randomized[:, :, 16:, :], 128.0)


def test_saliency_aligned_model():

    from src.core.saliency_alignment import SaliencyAlignedModel

    # Build a simple mock model using functional API
    inputs = keras.layers.Input(shape=(224, 224, 3), dtype="float32")
    x = keras.layers.Conv2D(4, (3, 3), activation="relu", name="test_conv")(
        inputs
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    # Wrap it with SaliencyAlignedModel (for CNN backbone mock)
    aligned_model = SaliencyAlignedModel(
        functional_model=model,
        backbone_name="EfficientNetV2B0",
        bg_weight=0.05,
    )

    aligned_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Create dummy batch of images: batch size = 2
    images = np.zeros((2, 224, 224, 3), dtype=np.float32)
    images[0, :, :112, :] = [50.0, 200.0, 50.0]  # high variance leaf
    images[0, :, 112:, :] = [128.0, 128.0, 128.0]  # background
    images[1, :, :, :] = [128.0, 128.0, 128.0]  # all background

    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    aligned_model_frozen = SaliencyAlignedModel(
        functional_model=model,
        backbone_name="EfficientNetV2B0",
        bg_weight=0.05,
        enable_penalties=False,
    )
    aligned_model_frozen.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Run a single train step to check that it executes without errors
    metrics = aligned_model.train_step((images, labels))

    assert "loss" in metrics
    assert "cls_loss" in metrics
    assert "bg_penalty" in metrics
    assert "sparsity_penalty" in metrics
    assert "accuracy" in metrics

    metrics_frozen = aligned_model_frozen.train_step((images, labels))
    assert float(metrics_frozen["bg_penalty"]) == 0.0
    assert float(metrics_frozen["sparsity_penalty"]) == 0.0


def test_saliency_aligned_model_multiblock_config():
    """Verify that multi-block configuration is accepted and stored."""

    from src.core.saliency_alignment import SaliencyAlignedModel

    inputs = keras.layers.Input(shape=(224, 224, 3), dtype="float32")
    x = keras.layers.Conv2D(4, (3, 3), activation="relu", name="test_conv_mb")(
        inputs
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    # Test explicit multi-block indices
    aligned = SaliencyAlignedModel(
        functional_model=model,
        backbone_name="EfficientNetV2B0",
        bg_weight=2.0,
        sparsity_weight=0.3,
        vit_block_indices=[8, 10, 11],
    )

    assert aligned.vit_block_indices == (8, 10, 11)
    assert aligned.bg_weight == 2.0
    assert aligned.sparsity_weight == 0.3

    # Test fallback to single index
    aligned_single = SaliencyAlignedModel(
        functional_model=model,
        backbone_name="EfficientNetV2B0",
        vit_block_indices=None,
        vit_block_idx=5,
    )
    # Should use config default ATTENTION_VIT_BLOCK_INDICES
    # (which is [8, 10, 11] from src.utils.config)
    assert len(aligned_single.vit_block_indices) >= 1

    aligned.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Run a train step to confirm it works with multi-block config
    images = np.zeros((2, 224, 224, 3), dtype=np.float32)
    images[0, :, :112, :] = [50.0, 200.0, 50.0]
    images[0, :, 112:, :] = [128.0, 128.0, 128.0]
    images[1, :, :, :] = [128.0, 128.0, 128.0]
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    metrics = aligned.train_step((images, labels))
    assert "loss" in metrics
    assert "bg_penalty" in metrics
    assert "sparsity_penalty" in metrics
