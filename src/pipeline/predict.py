import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow.keras as keras
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from src.core.inference_guard import (
    assess_leaf_likelihood,
    compute_prediction_diagnostics,
    evaluate_inference_safety,
)
from src.core.preprocessing import preprocess_array_for_model
from src.training.training_utils import WarmupCosineSchedule
from src.utils.config import (
    CLASS_INDICES_PATH,
    CONFIDENCE_REJECT_THRESHOLD,
    ENTROPY_REJECT_THRESHOLD,
    IMG_SIZE,
    OOD_MSP_THRESHOLD,
)
from src.utils.hardware import configure_tensorflow
from src.utils.model_paths import resolve_keras_model_path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
configure_tensorflow()


def _load_model_robust(model_path: str):
    from src.training.training_utils import HierarchicalLoss

    custom_objects = {
        "WarmupCosineSchedule": WarmupCosineSchedule,
        "HierarchicalLoss": HierarchicalLoss,
    }
    try:
        return load_model(
            model_path, custom_objects=custom_objects, compile=False
        )
    except TypeError as exc:
        error_text = str(exc)
        if "ViTPatchingAndEmbedding" not in error_text:
            raise

        if not _patch_vit_layer_init_for_compat():
            raise RuntimeError(
                "Failed to load ViT/DINO checkpoint due to keras-hub version mismatch. "
                "Install a compatible keras-hub version or retrain with current stack."
            ) from exc

        print(
            "Detected KerasHub ViT checkpoint compatibility mismatch; "
            "retrying load with compatibility shim."
        )
        return load_model(
            model_path, custom_objects=custom_objects, compile=False
        )


def _patch_vit_layer_init_for_compat() -> bool:
    """Patch keras-hub ViT layer init to ignore legacy serialized kwargs."""
    try:
        from keras_hub.src.models.vit import vit_layers

        layer_cls = vit_layers.ViTPatchingAndEmbedding
    except Exception:
        return False

    if getattr(layer_cls, "_leaf_compat_patched", False):
        return True

    original_init = layer_cls.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.pop("num_patches", None)
        kwargs.pop("num_positions", None)
        return original_init(self, *args, **kwargs)

    layer_cls.__init__ = _patched_init
    layer_cls._leaf_compat_patched = True
    return True


class LeafDiseasePredictor:
    def __init__(
        self,
        model_paths: list[str | None] | str | None = None,
        class_indices_path: str | Path = CLASS_INDICES_PATH,
        img_size: int = IMG_SIZE,
        model_path: str | None = None,
    ):
        self.img_size = img_size
        self.models = []

        # Resolve model paths into a clean list of paths
        from src.utils.config import ENSEMBLE_MODEL_PATHS

        resolved_paths: list[str | None] = []
        if model_path is not None:
            resolved_paths = [model_path]
        elif not model_paths and ENSEMBLE_MODEL_PATHS:
            resolved_paths = list(ENSEMBLE_MODEL_PATHS)
        elif isinstance(model_paths, str):
            resolved_paths = [model_paths]
        elif model_paths is not None:
            resolved_paths = list(model_paths)
        else:
            resolved_paths = [None]

        print(
            f"Initializing predictor with {len(resolved_paths)} model(s) for ensembling..."
        )

        for path in resolved_paths:
            resolved_path = resolve_keras_model_path([path] if path else None)
            print(f"Loading model from {resolved_path}...")
            model = _load_model_robust(resolved_path)
            self.models.append(model)

        print(f"{len(self.models)} model(s) loaded successfully.")

        # Detect backbone architecture for correct preprocessing
        # For ensembles, we assume all models use the same input preprocessing (backbone family)
        self.backbone_name = self._infer_backbone_from_model(
            resolve_keras_model_path(
                [resolved_paths[0]] if resolved_paths[0] else None
            ),
            self.models[0],
        )
        print(f"Detected primary backbone architecture: {self.backbone_name}")

        if os.path.exists(class_indices_path):
            with open(class_indices_path, "r") as f:
                self.class_indices = json.load(f)
        else:
            self.class_indices = self._generate_class_indices()

        self.idx_to_class = {v: k for k, v in self.class_indices.items()}
        print(
            f"Loaded {len(self.class_indices)} total classes "
            "(13 healthy, 33 disease)."
        )

    def _infer_backbone_from_model(
        self, model_path: str | None = None, model: keras.Model | None = None
    ) -> str:
        """Best-effort backbone name detection from loaded model layer names."""
        if model is None:
            model = getattr(self, "model", None) or (
                self.models[0] if getattr(self, "models", None) else None
            )
        if model is None:
            return "EfficientNetV2B0"
        path_hint = (model_path or "").lower()
        if any(token in path_hint for token in ["dino", "vit", "refined"]):
            return "DINOv3"
        if "efficientnetv2s" in path_hint:
            return "EfficientNetV2S"
        if "efficientnetv2b0" in path_hint:
            return "EfficientNetV2B0"
        if "efficientnetv2b1" in path_hint:
            return "EfficientNetV2B1"
        if "efficientnetv2b2" in path_hint:
            return "EfficientNetV2B2"
        if "efficientnetv2b3" in path_hint:
            return "EfficientNetV2B3"
        if "efficientnetv2m" in path_hint:
            return "EfficientNetV2M"
        if "efficientnetv2l" in path_hint:
            return "EfficientNetV2L"

        model_name = str(getattr(model, "name", "")).lower()
        layer_names = [
            str(getattr(layer, "name", "")).lower() for layer in model.layers
        ]
        haystack = " ".join([model_name, *layer_names])

        # Check for DINOv3/ViT models
        if "dinov3" in haystack or "vit" in haystack:
            return "DINOv3"

        # Check for EfficientNetV2 variants
        if "efficientnetv2s" in haystack:
            return "EfficientNetV2S"
        if "efficientnetv2b0" in haystack:
            return "EfficientNetV2B0"
        if "efficientnetv2b1" in haystack:
            return "EfficientNetV2B1"
        if "efficientnetv2b2" in haystack:
            return "EfficientNetV2B2"
        if "efficientnetv2b3" in haystack:
            return "EfficientNetV2B3"
        if "efficientnetv2m" in haystack:
            return "EfficientNetV2M"
        if "efficientnetv2l" in haystack:
            return "EfficientNetV2L"

        # Default to EfficientNetV2B0 if detection fails
        print(
            "[WARNING] Could not detect backbone from model names. "
            "Defaulting to EfficientNetV2B0."
        )
        return "EfficientNetV2B0"

    def _generate_class_indices(self) -> dict:

        train_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "dataset", "train"
        )
        classes = sorted(os.listdir(train_dir))
        return {
            cls: i
            for i, cls in enumerate(classes)
            if os.path.isdir(os.path.join(train_dir, cls))
        }

    def preprocess_image(self, img_path: str) -> np.ndarray:

        img = image.load_img(
            img_path, target_size=(self.img_size, self.img_size)
        )
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # Use detected backbone for correct preprocessing
        return preprocess_array_for_model(
            img_array, backbone_name=self.backbone_name
        )

    def predict(self, img_path: str) -> dict:

        img_array = self.preprocess_image(img_path)

        # Ensemble inference: average probabilities across all models
        all_preds = []
        for model in self.models:
            all_preds.append(model.predict(img_array, verbose=0)[0])

        predictions = np.mean(all_preds, axis=0)

        diagnostics = compute_prediction_diagnostics(predictions)
        top_idx = int(diagnostics["top1_index"])
        class_name = self.idx_to_class[top_idx]
        confidence = float(diagnostics["top1_prob"])
        confidence_margin = float(diagnostics["confidence_margin"])
        entropy_bits = float(diagnostics["entropy_bits"])

        leaf_validation = assess_leaf_likelihood(img_path, self.img_size)
        safety = evaluate_inference_safety(
            diagnostics=diagnostics,
            leaf_validation=leaf_validation,
            confidence_threshold=CONFIDENCE_REJECT_THRESHOLD,
            entropy_threshold_bits=ENTROPY_REJECT_THRESHOLD,
            msp_threshold=OOD_MSP_THRESHOLD,
            min_margin=0.08,
        )

        if safety["reject"]:
            reason = (
                ", ".join(safety["reasons"])
                if safety["reasons"]
                else "low trust score"
            )
            parts = class_name.split("___")
            plant = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
            disease = (
                parts[1].replace("_", " ") if len(parts) > 1 else class_name
            )
            return {
                "image_path": img_path,
                "disease": class_name,
                "confidence": confidence * 100,
                "reject": True,
                "rejection_reasons": safety["reasons"],
                "prediction": {
                    "class": class_name,
                    "plant": plant,
                    "disease": disease,
                    "confidence": confidence,
                    "confidence_percent": f"{confidence * 100:.2f}%",
                    "rejected": True,
                    "rejection_reason": reason,
                    "raw_top_class": class_name,
                    "validation": {
                        "leaf_score": leaf_validation["leaf_score"],
                        "vegetation_ratio": leaf_validation[
                            "vegetation_ratio"
                        ],
                        "confidence_margin": round(confidence_margin * 100, 2),
                        "entropy_bits": round(entropy_bits, 4),
                        "uncertainty_score": int(safety["uncertainty_score"]),
                    },
                },
            }

        parts = class_name.split("___")
        plant = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
        disease = parts[1].replace("_", " ") if len(parts) > 1 else class_name

        return {
            "image_path": img_path,
            "disease": class_name,
            "confidence": confidence * 100,
            "reject": False,
            "rejection_reasons": [],
            "prediction": {
                "class": class_name,
                "plant": plant,
                "disease": disease,
                "confidence": confidence,
                "confidence_percent": f"{confidence * 100:.2f}%",
                "rejected": False,
                "validation": {
                    "leaf_score": leaf_validation["leaf_score"],
                    "vegetation_ratio": leaf_validation["vegetation_ratio"],
                    "confidence_margin": round(confidence_margin * 100, 2),
                    "entropy_bits": round(entropy_bits, 4),
                    "uncertainty_score": int(safety["uncertainty_score"]),
                },
            },
        }

    def predict_and_visualize(
        self, img_path: str, save_path: str | None = None
    ):
        import matplotlib.pyplot as plt

        result = self.predict(img_path)
        pred = result["prediction"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        img = Image.open(img_path)
        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title("Input Image", fontsize=14, fontweight="bold")

        display_name = pred["class"].replace("___", " - ").replace("_", " ")
        color = (
            "green"
            if pred["confidence"] > 0.9
            else "orange"
            if pred["confidence"] > 0.7
            else "red"
        )
        ax2.barh([display_name], [pred["confidence"]], color=[color])
        ax2.set_xlabel("Confidence", fontsize=12)
        ax2.set_title("Prediction", fontsize=14, fontweight="bold")
        ax2.set_xlim([0, 1])
        ax2.text(
            pred["confidence"] + 0.02,
            0,
            f"{pred['confidence'] * 100:.1f}%",
            va="center",
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")

        plt.show()

        print(f"\n{'=' * 80}")
        print("PREDICTION RESULT")
        print(f"{'=' * 80}")
        print(f"  {pred['class']}")
        print(f"  Confidence: {pred['confidence_percent']}")
        if pred.get("rejected"):
            print("  Status: Rejected by safety gate")
            print(
                f"  Reason: {pred.get('rejection_reason', 'low trust score')}"
            )
        print()

    def predict_batch(
        self, image_folder: str, output_file: str = "predictions.json"
    ):

        valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        image_files = [
            f for f in os.listdir(image_folder) if f.endswith(valid_extensions)
        ]

        print(f"Processing {len(image_files)} images...")
        results = []

        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            try:
                prediction = self.predict(img_path)
                results.append(
                    {
                        "filename": img_file,
                        "predicted_class": prediction["prediction"]["class"],
                        "confidence": prediction["prediction"][
                            "confidence_percent"
                        ],
                    }
                )
                print(f"  {img_file}: {prediction['prediction']['class']}")
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nBatch predictions saved to {output_file}")
        return results


def main():

    parser = argparse.ArgumentParser(
        description="Leaf Disease Detection — Inference CLI"
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        help="Path to a single image file or a directory of images.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Path to a saved .keras model file.",
    )
    parser.add_argument(
        "--save",
        "-s",
        type=str,
        help="Path to save the prediction visualization.",
    )
    args = parser.parse_args()

    predictor = LeafDiseasePredictor(model_path=args.model)

    if args.image:
        if os.path.isfile(args.image):
            predictor.predict_and_visualize(args.image, save_path=args.save)
        elif os.path.isdir(args.image):
            predictor.predict_batch(args.image)
        else:
            print(f"Error: {args.image} is not a valid file or directory")
    else:
        test_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "dataset", "test"
        )
        for subdir in os.listdir(test_dir):
            subdir_path = os.path.join(test_dir, subdir)
            if os.path.isdir(subdir_path):
                images = os.listdir(subdir_path)
                if images:
                    test_image = os.path.join(subdir_path, images[0])
                    print(f"\nExample prediction on: {test_image}")
                    predictor.predict_and_visualize(
                        test_image, save_path="plots/example_prediction.png"
                    )
                    break


if __name__ == "__main__":
    main()
