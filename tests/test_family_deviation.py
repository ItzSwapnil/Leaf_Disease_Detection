import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow.keras as keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure model_paths is imported so custom layer registration triggers
import model_paths  # noqa: F401
from training_utils import FamilyDeviationClassifier, parse_class_structure


def test_parse_class_structure():
    class_names = [
        "Apple___Apple_scab",  # 0 -> healthy partner: 2
        "Apple___Black_rot",  # 1 -> healthy partner: 2
        "Apple___healthy",  # 2 -> itself healthy: -1
        "Corn___Common_rust",  # 3 -> healthy partner: 4
        "Corn___healthy",  # 4 -> itself healthy: -1
        "Squash___Powdery_mildew",  # 5 -> no healthy in family: -1
        "Wheat brown spot",  # 6 -> no healthy in family: -1
    ]

    partners = parse_class_structure(class_names)
    assert partners == [2, 2, -1, 4, -1, -1, -1]


def test_family_deviation_classifier_compiles_and_serializes():
    num_classes = 4
    healthy_partners = [2, 2, -1, -1]  # Classes 0 and 1 deviate from 2.

    # Build model using Functional API
    inputs = keras.layers.Input(shape=(8,), dtype="float32")
    logits = FamilyDeviationClassifier(
        num_classes=num_classes, healthy_partners=healthy_partners
    )(inputs)
    outputs = keras.layers.Activation("softmax", dtype="float32")(logits)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Test forward pass
    dummy_input = np.ones((2, 8), dtype=np.float32)
    dummy_output = model.predict(dummy_input, verbose=0)
    assert dummy_output.shape == (2, num_classes)
    np.testing.assert_allclose(np.sum(dummy_output, axis=-1), 1.0, rtol=1e-5)

    # Test serialization and loading using robust method
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model.keras")
        model.save(model_path)

        # Load model using keras load_model (which resolves custom objects via model_paths import)
        loaded_model = keras.models.load_model(model_path, compile=False)
        loaded_output = loaded_model.predict(dummy_input, verbose=0)
        np.testing.assert_allclose(dummy_output, loaded_output, rtol=1e-5)
