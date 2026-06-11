import sys
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_utils import HierarchicalLoss


def test_hierarchical_loss_parsing_and_tensors():
    # 4 classes: 2 families (Apple, Grape), each has 1 healthy and 1 diseased class
    class_names = [
        "Apple___Apple_scab",
        "Apple___healthy",
        "Grape___Black_rot",
        "Grape___healthy",
    ]

    loss_fn = HierarchicalLoss(class_names=class_names, label_smoothing=0.1)

    # 2 unique families: Apple (0) and Grape (1)
    assert loss_fn.num_families == 2
    assert loss_fn.num_classes == 4

    # Healthy classes mapping: healthy is at idx 1 (Apple___healthy) and idx 3 (Grape___healthy)
    assert loss_fn.class_is_healthy == [0.0, 1.0, 0.0, 1.0]
    assert loss_fn.class_to_family_id == [0, 0, 1, 1]


def test_hierarchical_loss_execution():
    class_names = [
        "Apple___Apple_scab",
        "Apple___healthy",
        "Grape___Black_rot",
        "Grape___healthy",
    ]

    loss_fn = HierarchicalLoss(class_names=class_names, label_smoothing=0.1)

    # Batch size 2
    # y_true: class 1 (Apple___healthy) and class 2 (Grape___Black_rot)
    y_true = tf.constant(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=tf.float32
    )

    # y_pred: some soft probabilities
    y_pred = tf.constant(
        [[0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.6, 0.2]], dtype=tf.float32
    )

    loss_val = loss_fn(y_true, y_pred)

    assert loss_val.shape == ()
    assert float(loss_val.numpy()) > 0.0


def test_hierarchical_loss_get_config():
    class_names = [
        "Apple___Apple_scab",
        "Apple___healthy",
        "Grape___Black_rot",
        "Grape___healthy",
    ]

    loss_fn = HierarchicalLoss(class_names=class_names, label_smoothing=0.1)
    config = loss_fn.get_config()

    assert "class_names" in config
    assert config["class_names"] == class_names
    assert config["label_smoothing"] == 0.1

    # Deserialization test via standard keras mechanism
    reconstructed_loss = HierarchicalLoss.from_config(config)
    assert reconstructed_loss.num_families == 2
    assert reconstructed_loss.class_to_family_id == [0, 0, 1, 1]
