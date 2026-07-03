import argparse
import json
import os
import sys
from pathlib import Path
from typing import TypedDict, cast

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2

from src.core.inference_guard import (
    SafetyEvaluation,
    assess_leaf_likelihood,
    compute_prediction_diagnostics,
    evaluate_inference_safety,
)
from src.utils.config import (
    CLASS_INDICES_PATH,
    CONFIDENCE_REJECT_THRESHOLD,
    ENTROPY_REJECT_THRESHOLD,
    IMG_SIZE,
    OOD_MSP_THRESHOLD,
    TRAIN_DIR,
)
from src.utils.model_paths import resolve_pytorch_model_path


class ValidationMetrics(TypedDict):
    leaf_score: float
    vegetation_ratio: float
    confidence_margin: float
    entropy_bits: float
    uncertainty_score: int


DiseasePrediction = TypedDict(
    "DiseasePrediction",
    {
        "class": str,
        "plant": str,
        "disease": str,
        "confidence": float,
        "confidence_percent": str,
        "rejected": bool,
        "rejection_reason": str,
        "raw_top_class": str,
        "validation": ValidationMetrics,
    },
    total=False,
)


class PredictionResponse(TypedDict):
    image_path: str
    disease: str
    confidence: float
    reject: bool
    rejection_reasons: list[str]
    prediction: DiseasePrediction
    cropped_bbox: tuple[int, int, int, int] | None


def _extract_disease_predictions(
    predictions: dict[str, object]
    | list[object]
    | tuple[object, ...]
    | torch.Tensor
    | np.ndarray,
) -> np.ndarray:
    """Return disease probabilities from single-output or multi-output models."""
    val: object
    if isinstance(predictions, dict):
        preds_dict = cast(dict[str, object], predictions)
        if "disease_output" in preds_dict:
            val = preds_dict["disease_output"]
        else:
            val = next(iter(preds_dict.values()))
    elif isinstance(predictions, (list, tuple)):
        val = predictions[-1]
    else:
        val = predictions

    if isinstance(val, torch.Tensor):
        return val.cpu().numpy()
    return np.asarray(val)


def _load_model_robust(
    model_path: str | Path | None,
) -> tuple[torch.nn.Module, str]:
    import os

    os.environ["KERAS_BACKEND"] = "torch"

    from src.training.train_model import LeafDiseaseModel
    from src.training.training_utils import parse_class_structure
    from src.utils.config import TRAIN_DIR
    from src.utils.hardware import get_device

    device: torch.device = get_device()

    # Need class names to initialize model
    idx_to_class: dict[int, str]
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r") as f:
            idx_to_class = {int(v): k for k, v in json.load(f).items()}
    elif os.path.exists(TRAIN_DIR):
        train_class_names: list[str] = sorted(
            entry.name for entry in os.scandir(TRAIN_DIR) if entry.is_dir()
        )
        idx_to_class = {i: name for i, name in enumerate(train_class_names)}
        # Auto-save for next time
        os.makedirs(os.path.dirname(CLASS_INDICES_PATH), exist_ok=True)
        try:
            with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {name: idx for idx, name in idx_to_class.items()},
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"[!] Warning: Failed to save class indices: {e}")
    else:
        # Fallback if neither exists
        idx_to_class = {i: f"Class_{i}" for i in range(46)}

    class_names: list[str] = [
        idx_to_class[i] for i in range(len(idx_to_class))
    ]
    num_classes: int = len(class_names)
    crop_names: list[str] = sorted(
        list(set(name.split("___")[0] for name in class_names))
    )
    num_crops: int = len(crop_names)
    healthy_partners: list[int] = parse_class_structure(
        class_names
    )

    if model_path is None:
        from src.utils.model_paths import resolve_model_path
        try:
            model_path = resolve_model_path()
        except FileNotFoundError as e:
            raise ValueError(f"Model path must be specified or resolved: {e}")

    path_hint: str = str(model_path).lower()

    model: torch.nn.Module
    backbone_name: str
    if str(model_path).endswith(".keras"):
        if any(token in path_hint for token in ["dino", "vit", "refined"]):
            backbone_name = "DINOv3"
        else:
            backbone_name = "EfficientNetV2B0"

        import keras

        model = keras.models.load_model(model_path)
    else:
        state = torch.load(
            model_path, map_location=device, weights_only=True
        )
        actual_state: dict[str, torch.Tensor] = state.get(
            "model_state_dict", state
        )

        # Auto-detect backbone from state dict keys
        has_vit_keys = any(
            "class_token" in k or "encoder." in k
            for k in actual_state.keys()
        )
        has_efficientnet_keys = any(
            "features.0.0.weight" in k for k in actual_state.keys()
        )

        if has_vit_keys:
            backbone_name = "DINOv3"
        elif has_efficientnet_keys:
            backbone_name = "EfficientNetV2B0"
        else:
            if any(
                token in path_hint for token in ["dino", "vit", "refined"]
            ):
                backbone_name = "DINOv3"
            else:
                backbone_name = "EfficientNetV2B0"

        model = LeafDiseaseModel(
            backbone_name, num_classes, num_crops, healthy_partners
        )
        model.load_state_dict(actual_state)
        model.to(device)
        model.eval()

    return model, backbone_name


class LeafDiseasePredictor:
    def __init__(
        self,
        model_paths: list[str | None] | str | None = None,
        class_indices_path: str | Path = CLASS_INDICES_PATH,
        img_size: int = IMG_SIZE,
        model_path: str | None = None,
        use_background_removal: bool | None = None,
    ) -> None:
        from src.utils.hardware import get_device

        self.device: torch.device = get_device()
        self.img_size: int = img_size

        # CPU Thread optimization
        from src.utils.config import INTRA_OP_THREADS, INTER_OP_THREADS
        torch.set_num_threads(INTRA_OP_THREADS)
        try:
            torch.set_num_interop_threads(INTER_OP_THREADS)
        except RuntimeError:
            pass

        # CUDA benchmark & TF32 optimizations for RTX GPUs
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        self.models: list[torch.nn.Module] = []
        self.backbone_name: str = "EfficientNetV2B0"

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
            f"Initializing predictor with {len(resolved_paths)} model(s) "
            "for ensembling..."
        )

        for path in resolved_paths:
            resolved_path = resolve_pytorch_model_path(
                [path] if path else None
            )
            print(f"Loading model from {resolved_path}...")
            model, b_name = _load_model_robust(resolved_path)
            self.backbone_name = b_name
            self.models.append(model)

        print(f"{len(self.models)} model(s) loaded successfully.")

        if os.path.exists(class_indices_path):
            with open(class_indices_path, "r") as f:
                self.class_indices: dict[str, int] = json.load(f)
        else:
            self.class_indices = self._generate_class_indices()

        self.idx_to_class: dict[int, str] = {
            v: k for k, v in self.class_indices.items()
        }
        print(
            f"Loaded {len(self.class_indices)} total classes "
            "(13 healthy, 33 disease)."
        )

        self.transform: v2.Compose = v2.Compose(
            [
                v2.Resize((self.img_size, self.img_size)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Background removal configuration
        from src.utils.config import USE_YOLO_LEAF_DETECTION
        if use_background_removal is None:
            self.use_background_removal = USE_YOLO_LEAF_DETECTION
            if "LEAF_PIPELINE_YOLO_FOCUS" in os.environ:
                self.use_background_removal = USE_YOLO_LEAF_DETECTION and (
                    os.environ["LEAF_PIPELINE_YOLO_FOCUS"].strip().lower() in {"1", "true", "yes", "on"}
                )
        else:
            self.use_background_removal = use_background_removal

        self.yolo_leaf_detector = None

    def _get_yolo_leaf_detector(self):
        """Return a lazily initialized YOLO focus detector, if available."""
        if not self.use_background_removal:
            return None
        if self.yolo_leaf_detector is not None:
            return self.yolo_leaf_detector
        try:
            from src.core.yolo_leaf import YOLOLeafDetector
            self.yolo_leaf_detector = YOLOLeafDetector()
        except Exception as exc:
            print(f"[!] Failed to initialize YOLOLeafDetector: {exc}")
            self.yolo_leaf_detector = None
        return self.yolo_leaf_detector

    def _generate_class_indices(self) -> dict[str, int]:
        classes: list[str] = sorted(os.listdir(TRAIN_DIR))
        return {
            cls: i
            for i, cls in enumerate(classes)
            if os.path.isdir(os.path.join(TRAIN_DIR, cls))
        }

    def preprocess_image(self, img_path: str) -> torch.Tensor:
        img: Image.Image = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")

        # If background removal is enabled, run YOLO and crop
        detector = self._get_yolo_leaf_detector()
        if detector is not None:
            import cv2
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            detection = detector.detect(img_bgr)
            if detection["found"]:
                x1, y1, x2, y2 = detection["bbox"]
                img_cropped = img.crop((x1, y1, x2, y2))
                
                # Precise GrabCut segmentation inside the cropped bbox to mask out backgrounds/shadows
                from src.core.leaf_segmentation import segment_leaf_grabcut
                img_cropped_rgb = np.array(img_cropped)
                seg_res = segment_leaf_grabcut(img_cropped_rgb)
                if seg_res["success"]:
                    img = Image.fromarray(seg_res["masked_image"])
                else:
                    img = img_cropped
        else:
            if self.use_background_removal:
                # Fallback: precise GrabCut segmentation on the entire image
                from src.core.leaf_segmentation import segment_leaf_grabcut
                img_rgb = np.array(img)
                seg_res = segment_leaf_grabcut(img_rgb)
                if seg_res["success"]:
                    img = Image.fromarray(seg_res["masked_image"])

        tensor: torch.Tensor = self.transform(img).unsqueeze(0).to(self.device, non_blocking=True)
        return tensor

    @torch.no_grad()
    def predict(self, img_path: str) -> PredictionResponse:
        img: Image.Image = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")

        cropped_bbox = None
        detector = self._get_yolo_leaf_detector()
        if detector is not None:
            import cv2
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            detection = detector.detect(img_bgr)
            if detection["found"]:
                x1, y1, x2, y2 = detection["bbox"]
                cropped_bbox = (x1, y1, x2, y2)
                img_cropped = img.crop((x1, y1, x2, y2))
                
                # Precise GrabCut segmentation inside the cropped bbox to mask out backgrounds/shadows
                from src.core.leaf_segmentation import segment_leaf_grabcut
                img_cropped_rgb = np.array(img_cropped)
                seg_res = segment_leaf_grabcut(img_cropped_rgb)
                if seg_res["success"]:
                    img = Image.fromarray(seg_res["masked_image"])
                else:
                    img = img_cropped
        else:
            if self.use_background_removal:
                # Fallback: precise GrabCut segmentation on the entire image
                from src.core.leaf_segmentation import segment_leaf_grabcut
                img_rgb = np.array(img)
                seg_res = segment_leaf_grabcut(img_rgb)
                if seg_res["success"]:
                    img = Image.fromarray(seg_res["masked_image"])

        tensor: torch.Tensor = self.transform(img).unsqueeze(0).to(self.device, non_blocking=True)

        # Ensemble inference: average probabilities across all models
        all_preds: list[np.ndarray] = []
        device_type = (
            self.device.type
            if self.device.type in ("cuda", "cpu")
            else "cuda"
        )
        use_bf16 = (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            for model in self.models:
                output = model(tensor)
                if isinstance(output, dict) and "disease_output" in output:
                    logits = output["disease_output"]
                else:
                    logits = output
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                all_preds.append(probs)

        predictions: np.ndarray = np.mean(all_preds, axis=0)

        diagnostics: dict[str, float] = compute_prediction_diagnostics(
            predictions
        )
        top_idx: int = int(diagnostics["top1_index"])
        class_name: str = self.idx_to_class[top_idx]
        confidence: float = float(diagnostics["top1_prob"])
        confidence_margin: float = float(
            diagnostics["confidence_margin"]
        )
        entropy_bits: float = float(diagnostics["entropy_bits"])

        leaf_validation: dict[str, object] = assess_leaf_likelihood(
            img_path, self.img_size
        )
        safety: SafetyEvaluation = evaluate_inference_safety(
            diagnostics=diagnostics,
            leaf_validation=leaf_validation,
            confidence_threshold=CONFIDENCE_REJECT_THRESHOLD,
            entropy_threshold_bits=ENTROPY_REJECT_THRESHOLD,
            msp_threshold=OOD_MSP_THRESHOLD,
            min_margin=0.08,
        )

        rejection_reasons: list[str] = safety["reasons"]

        if safety["reject"]:
            reason = (
                ", ".join(rejection_reasons)
                if rejection_reasons
                else "low trust score"
            )
            parts = class_name.split("___")
            plant = (
                parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
            )
            disease = (
                parts[1].replace("_", " ")
                if len(parts) > 1
                else class_name
            )
            return {
                "image_path": img_path,
                "disease": class_name,
                "confidence": confidence * 100,
                "reject": True,
                "rejection_reasons": rejection_reasons,
                "cropped_bbox": cropped_bbox,
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
                        "leaf_score": float(leaf_validation["leaf_score"]),  # type: ignore
                        "vegetation_ratio": float(
                            leaf_validation["vegetation_ratio"]  # type: ignore
                        ),
                        "confidence_margin": round(
                            confidence_margin * 100, 2
                        ),
                        "entropy_bits": round(entropy_bits, 4),
                        "uncertainty_score": int(
                            safety["uncertainty_score"]
                        ),
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
            "cropped_bbox": cropped_bbox,
            "prediction": {
                "class": class_name,
                "plant": plant,
                "disease": disease,
                "confidence": confidence,
                "confidence_percent": f"{confidence * 100:.2f}%",
                "rejected": False,
                "validation": {
                    "leaf_score": float(leaf_validation["leaf_score"]),  # type: ignore
                    "vegetation_ratio": float(
                        leaf_validation["vegetation_ratio"]  # type: ignore
                    ),
                    "confidence_margin": round(confidence_margin * 100, 2),
                    "entropy_bits": round(entropy_bits, 4),
                    "uncertainty_score": int(
                        safety["uncertainty_score"]
                    ),
                },
            },
        }

    def predict_and_visualize(
        self, img_path: str, save_path: str | None = None
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        result: PredictionResponse = self.predict(img_path)
        pred = result["prediction"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        ax1.imshow(img)

        # Highlight the detected leaf focus bounding box if it was cropped
        cropped_bbox = result.get("cropped_bbox")
        if cropped_bbox is not None:
            x1, y1, x2, y2 = cropped_bbox
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=3,
                edgecolor="#22c55e",  # Vibrant green bounding box
                facecolor="none",
            )
            ax1.add_patch(rect)
            ax1.text(
                x1,
                max(y1 - 10, 20),
                "Leaf Focus",
                color="#22c55e",
                fontsize=12,
                weight="bold",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2),
            )

        ax1.axis("off")
        ax1.set_title("Input Image", fontsize=14, fontweight="bold")

        display_name: str = pred["class"].replace("___", " - ").replace(
            "_", " "
        )
        pred_conf = float(pred["confidence"])
        color = (
            "green"
            if pred_conf > 0.9
            else "orange"
            if pred_conf > 0.7
            else "red"
        )
        ax2.barh([display_name], [pred_conf], color=[color])
        ax2.set_xlabel("Confidence", fontsize=12)
        ax2.set_title("Prediction", fontsize=14, fontweight="bold")
        ax2.set_xlim([0, 1])
        ax2.text(
            pred_conf + 0.02,
            0,
            f"{pred_conf * 100:.1f}%",
            va="center",
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")
        else:
            os.makedirs("plots", exist_ok=True)
            default_save = "plots/latest_prediction.png"
            plt.savefig(default_save, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {default_save}")

        plt.close(fig)

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
    ) -> list[dict[str, str]]:
        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG",
        )
        image_files: list[str] = [
            f
            for f in os.listdir(image_folder)
            if f.endswith(valid_extensions)
        ]

        print(f"Processing {len(image_files)} images...")
        results: list[dict[str, str]] = []

        for img_file in image_files:
            img_path: str = os.path.join(image_folder, img_file)
            try:
                prediction: PredictionResponse = self.predict(img_path)
                pred = prediction["prediction"]
                results.append(
                    {
                        "filename": img_file,
                        "predicted_class": str(pred["class"]),
                        "confidence": str(pred["confidence_percent"]),
                    }
                )
                print(f"  {img_file}: {pred['class']}")
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nBatch predictions saved to {output_file}")
        return results


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
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
        help="Path to a saved model file.",
    )
    parser.add_argument(
        "--save",
        "-s",
        type=str,
        help="Path to save the prediction visualization.",
    )
    args: argparse.Namespace = parser.parse_args()

    predictor: LeafDiseasePredictor = LeafDiseasePredictor(
        model_path=args.model
    )

    if args.image:
        if os.path.isfile(args.image):
            predictor.predict_and_visualize(
                args.image, save_path=args.save
            )
        elif os.path.isdir(args.image):
            predictor.predict_batch(args.image)
        else:
            print(f"Error: {args.image} is not a valid file or directory")
    else:
        test_dir: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "dataset",
            "test",
        )
        for subdir in os.listdir(test_dir):
            subdir_path: str = os.path.join(test_dir, subdir)
            if os.path.isdir(subdir_path):
                images: list[str] = os.listdir(subdir_path)
                if images:
                    test_image: str = os.path.join(
                        subdir_path, images[0]
                    )
                    print(f"\nExample prediction on: {test_image}")
                    predictor.predict_and_visualize(
                        test_image,
                        save_path="plots/example_prediction.png",
                    )
                    break


if __name__ == "__main__":
    main()
