import argparse
import os
import sys
import numpy as np
import torch

# Add project root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluation.metrics.calibration import optimize_temperature
from src.pipeline.predict import _load_model_robust
from src.training.training_progress import ProgressEmitter
from src.training.training_utils import build_dynamic_yolo_dataset
from src.utils.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLASSIFIER_MODEL_PATH,
    REFINED_MODEL_PATH,
    VAL_DIR,
    NUM_WORKERS,
    INTRA_OP_THREADS,
    INTER_OP_THREADS,
    TEMPERATURE_SCALING_STEPS,
    TEMPERATURE_SCALING_LR,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    # CPU Thread optimization
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

    # Resolve active model path to refine
    model_path = CLASSIFIER_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = CHECKPOINT_PATH
        if not os.path.exists(model_path):
            msg = (
                f"[Error] Neither classifier checkpoint ({CLASSIFIER_MODEL_PATH}) "
                f"nor base checkpoint ({CHECKPOINT_PATH}) was found. "
                "Train the model first."
            )
            print(msg)
            sys.exit(1)

    print(f"Loading model to refine from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, backbone_name = _load_model_robust(model_path)
    model.to(device)
    model.eval()

    # Load Dataset
    val_class_names = sorted(
        entry.name for entry in os.scandir(VAL_DIR) if entry.is_dir()
    )
    _skip_yolo = "dinov3" in backbone_name.lower()

    num_workers = args.num_workers if args.num_workers is not None else NUM_WORKERS
    val_loader = build_dynamic_yolo_dataset(
        VAL_DIR,
        val_class_names,
        int(BATCH_SIZE),
        shuffle=False,
        use_yolo=not _skip_yolo,
        num_workers=num_workers,
    )

    all_logits = []
    all_labels = []

    progress_emitter = ProgressEmitter("refining", total_epochs=1)
    progress_emitter.on_train_begin(len(val_loader), initial_epoch=0)
    progress_emitter.on_epoch_begin(0)

    print("Collecting validation set logits...")
    device_type = device.type if device.type in ("cuda", "cpu") else "cuda"
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            for batch_idx, (images, _, labels) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                disease_out = (
                    outputs["disease_output"]
                    if isinstance(outputs, dict)
                    else outputs
                )
                all_logits.append(disease_out.cpu().float().numpy())
                all_labels.extend(labels.numpy())
                progress_emitter.on_train_batch_end(batch_idx)

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.array(all_labels)

    print("Optimizing temperature scaling parameter...")
    steps = int(os.getenv("LEAF_TEMPERATURE_SCALING_STEPS", TEMPERATURE_SCALING_STEPS))
    lr = float(os.getenv("LEAF_TEMPERATURE_SCALING_LR", TEMPERATURE_SCALING_LR))

    cal_res = optimize_temperature(
        logits_np, labels_np, steps=steps, learning_rate=lr
    )
    optimal_temp = cal_res["temperature"]
    print("Refinement Successful:")
    print(f"  Optimal Temperature: {optimal_temp:.4f}")
    print(f"  NLL Before Calibration: {cal_res['nll_before']:.4f}")
    print(f"  NLL After Calibration: {cal_res['nll_after']:.4f}")

    # Save the refined model state dict and temperature scaling parameter
    print(f"Saving refined model to: {REFINED_MODEL_PATH}")
    os.makedirs(os.path.dirname(REFINED_MODEL_PATH), exist_ok=True)

    # Save the complete state dict so it can be loaded robustly
    checkpoint_state = torch.load(model_path, map_location="cpu")
    checkpoint_state["temperature"] = optimal_temp

    # If the checkpoint is just a state dict, wrap it
    if "model_state_dict" not in checkpoint_state:
        checkpoint_state = {
            "model_state_dict": checkpoint_state,
            "temperature": optimal_temp,
            "backbone_name": backbone_name,
        }

    torch.save(checkpoint_state, REFINED_MODEL_PATH)
    print("Refined model saved successfully.")
    progress_emitter.on_epoch_end(0)


if __name__ == "__main__":
    main()
