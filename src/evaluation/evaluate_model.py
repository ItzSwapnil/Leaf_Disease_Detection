from __future__ import annotations

import argparse
import os
import sys

# Add project root to sys.path to support running directly as a script
ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from sklearn.metrics import classification_report

from src.pipeline.predict import _load_model_robust
from src.training.training_utils import build_dynamic_yolo_dataset
from src.utils.config import BATCH_SIZE, VAL_DIR


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Evaluate leaf disease PyTorch model."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to the .pt model to evaluate.",
    )
    args: argparse.Namespace = parser.parse_args()

    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

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

    # Load Model
    model: torch.nn.Module
    backbone_name: str
    model, backbone_name = _load_model_robust(args.model_path)
    model.to(device)
    model.eval()

    # Load Dataset
    val_class_names: list[str] = sorted(
        entry.name for entry in os.scandir(VAL_DIR) if entry.is_dir()
    )
    _skip_yolo: bool = "dinov3" in backbone_name.lower()

    val_loader = build_dynamic_yolo_dataset(
        VAL_DIR,
        val_class_names,
        int(BATCH_SIZE),
        shuffle=False,
        use_yolo=not _skip_yolo,
    )

    y_true: list[int] = []
    y_pred: list[int] = []

    print("Starting Evaluation...")
    device_type = (
        device.type if device.type in ("cuda", "cpu") else "cuda"
    )
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            for images, _, labels in val_loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                disease_out = (
                    outputs["disease_output"]
                    if isinstance(outputs, dict)
                    else outputs
                )
                _, predicted = torch.max(disease_out, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

    print("\n--- Classification Report ---")
    report_str = classification_report(
        y_true, y_pred, target_names=val_class_names, zero_division=0
    )
    print(report_str)


if __name__ == "__main__":
    main()


