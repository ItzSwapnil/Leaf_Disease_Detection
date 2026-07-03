"""Grad-CAM diagnostic tool for verifying model attention.

Generates Grad-CAM heatmaps overlaid on sample images to verify whether
the model is looking at the leaf/disease features or the background.
This is critical for diagnosing the confusion between similar disease
classes (e.g. Apple Black Rot vs Apple Rust) on clinical plain-background
images.

Usage:
    python scripts/gradcam_check.py
    python scripts/gradcam_check.py --model-path models/leaf_disease_classifier.keras
    python scripts/gradcam_check.py --num-samples 20 --output-dir plots/gradcam
"""

from __future__ import annotations

from typing import Any

import argparse
import json
import os
import random
import sys

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from PIL import Image, ImageOps

from src.pipeline.predict import _load_model_robust
from src.utils.config import (
    CLASS_INDICES_PATH,
    FINAL_MODEL_PATH,
    IMG_SIZE,
    PLOTS_DIR,
    VAL_DIR,
)
from src.utils.model_paths import resolve_pytorch_model_path

# background_remover import bypassed


def _load_class_indices(path: str) -> dict[int, str]:
    """Load class_indices.json and return idx -> label mapping."""
    with open(path, "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)
    return {int(v): k for k, v in label_to_idx.items()}


class HookContainer:
    def __init__(self) -> None:
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None

    def __call__(self, module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
        if isinstance(output, tuple):
            self.activation = output[0]
        else:
            self.activation = output

        def backward_hook(grad: torch.Tensor) -> None:
            self.gradient = grad.detach()

        self.activation.register_hook(backward_hook)


def _find_last_conv_layer(module: nn.Module) -> nn.Module | None:
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def _process_heatmap_tensor(
    act: torch.Tensor,
    grad: torch.Tensor,
    backbone_name: str,
) -> np.ndarray:
    act = act.detach().cpu()
    grad = grad.detach().cpu()

    if "vit" in backbone_name.lower() or backbone_name == "DINOv3":
        # Average gradients over sequence dimension: shape (1, 768)
        pooled = torch.mean(grad, dim=1)
        # Weighted sum: shape (197,)
        hm = torch.sum(act[0] * pooled[0], dim=-1)
        # Slice off class token at index 0: shape (196,)
        hm = hm[1:]
        # Apply ReLU
        hm = torch.clamp(hm, min=0.0)
        # Reshape to (14, 14)
        gs = int(np.sqrt(hm.shape[0]))
        hm = hm.view(gs, gs)
        heatmap = hm.numpy()
    else:
        # CNN backbone: shape (1, C, H, W)
        # Average gradients over spatial dimensions (height, width): shape (C,)
        pooled = torch.mean(grad, dim=(0, 2, 3))
        # Weighted sum over channels: shape (H, W)
        hm = torch.sum(act[0] * pooled.view(-1, 1, 1), dim=0)
        hm = torch.clamp(hm, min=0.0)
        heatmap = hm.numpy()

    # Normalise heatmap to [0, 1]
    denom = np.max(heatmap) + 1e-8
    heatmap = heatmap / denom

    import cv2
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    return heatmap


def _make_gradcam_heatmap(
    model: nn.Module,
    img_tensor: torch.Tensor,
    target_layer_name: str | None = None,
    pred_index: int | None = None,
    backbone_name: str = "DINOv3",
    vit_block_idx: int = 10,
    healthy_partner_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Grad-CAM heatmap for a single image using PyTorch hooks.

    Args:
        model: PyTorch LeafDiseaseModel.
        img_tensor: Preprocessed image tensor on device of shape (1, 3, 224, 224).
        target_layer_name: Name of the target layer (used for CNN/Conv2D models).
        pred_index: Class index to visualize. None = top predicted class.
        backbone_name: Detected backbone name (e.g. "DINOv3" or "EfficientNetV2B0").
        vit_block_idx: The block index of the ViT backbone to visualize (default: 10).
        healthy_partner_idx: Healthy baseline index for FBDL deviation computation.

    Returns:
        tuple containing (crop_heatmap, disease_heatmap) of shape (H, W) in [0, 1].
    """
    # 1. Resolve target module
    backbone_wrapper = getattr(model, "backbone", None)
    if backbone_wrapper is None:
        raise ValueError("Model does not have a backbone attribute.")
    underlying_backbone = getattr(backbone_wrapper, "backbone", None)
    if underlying_backbone is None:
        raise ValueError("Backbone wrapper does not have a backbone attribute.")

    target_module: nn.Module
    if backbone_name == "DINOv3":
        encoder = getattr(underlying_backbone, "encoder", None)
        if encoder is None:
            raise ValueError("ViT backbone has no encoder.")
        layers = getattr(encoder, "layers", None)
        if layers is None:
            raise ValueError("ViT encoder has no layers.")
        try:
            target_module = layers[vit_block_idx]
        except (AttributeError, IndexError) as e:
            raise ValueError(
                f"Failed to find ViT block index {vit_block_idx} in model backbone: {e}"
            )
    else:
        if target_layer_name:
            resolved_module = None
            for name, module in model.named_modules():
                if name == target_layer_name:
                    resolved_module = module
                    break
            if resolved_module is None:
                raise ValueError(f"Target layer name '{target_layer_name}' not found.")
            target_module = resolved_module
        else:
            resolved_module = _find_last_conv_layer(underlying_backbone)
            if resolved_module is None:
                raise ValueError("Could not find any Conv2d layer in backbone.")
            target_module = resolved_module

    # 2. Register hooks
    container = HookContainer()
    handle = target_module.register_forward_hook(container)

    # Enable grad to compute Grad-CAM gradients
    with torch.set_grad_enabled(True):
        outputs = model(img_tensor)
        crop_logits = outputs["crop_output"]
        disease_logits = outputs["disease_output"]

        if pred_index is None:
            pred_index = int(torch.argmax(disease_logits[0]).item())
        crop_pred_index = int(torch.argmax(crop_logits[0]).item())

        if healthy_partner_idx is not None and healthy_partner_idx != -1:
            disease_score = disease_logits[0, pred_index] - disease_logits[0, healthy_partner_idx]
        else:
            disease_score = disease_logits[0, pred_index]

        crop_score = crop_logits[0, crop_pred_index]

        # Backward for disease score
        model.zero_grad()
        disease_score.backward(retain_graph=True)
        disease_grad = container.gradient
        disease_act = container.activation

        # Clear gradient state in hook container
        container.gradient = None

        # Backward for crop score
        model.zero_grad()
        crop_score.backward()
        crop_grad = container.gradient
        crop_act = container.activation

    # Clean up hook handle
    handle.remove()

    # Process heatmaps
    if disease_grad is None or disease_act is None:
        disease_heatmap = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    else:
        disease_heatmap = _process_heatmap_tensor(disease_act, disease_grad, backbone_name)

    if crop_grad is None or crop_act is None:
        crop_heatmap = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    else:
        crop_heatmap = _process_heatmap_tensor(crop_act, crop_grad, backbone_name)

    return crop_heatmap, disease_heatmap


def _overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay a heatmap on the original image."""
    if colormap == "jet":
        r = np.clip(1.5 - np.abs(4 * heatmap - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * heatmap - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * heatmap - 1), 0, 1)
    else:  # "viridis" approx (blue-green-yellow)
        r = heatmap
        g = heatmap * 0.8
        b = 1.0 - heatmap

    colored_heatmap = np.stack([r, g, b], axis=-1) * 255

    # Overlay
    original = original_img.astype(np.float32)
    overlay = original * (1 - alpha) + colored_heatmap * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _extract_leaf_mask(img_array: np.ndarray) -> np.ndarray:
    """Segment the leaf (foreground) from the neutral background.

    Uses standard deviation over RGB channels to distinguish colorful leaf tissue
    from neutral clinical backgrounds (gray, black, etc.).
    """
    # img_array shape is (224, 224, 3) in [0, 255]
    rgb_std = np.std(img_array, axis=-1)
    mean_val = np.mean(img_array, axis=-1)
    # Leaf pixels typically have high standard deviation across color channels
    leaf_mask = (rgb_std > 8.0) & (mean_val > 20.0)
    return leaf_mask.astype(np.float32)


def _simple_blur(img_array: np.ndarray, size: int = 9) -> np.ndarray:
    """Fast, 100% vectorized 2D prefix-sum box filter box blur in pure NumPy."""
    h, w, c = img_array.shape
    pad = size // 2
    padded = np.pad(img_array, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

    # Prepend zeros to make index subtraction clean and out-of-bounds safe
    cum = np.zeros((h + 2 * pad + 1, w + 2 * pad + 1, c), dtype=np.float64)
    cum[1:, 1:, :] = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    # Prefix sum calculation indices
    y2, x2 = np.arange(size, h + size), np.arange(size, w + size)
    y1, x1 = y2 - size, x2 - size

    box_sum = (
        cum[y2[:, None], x2]
        - cum[y1[:, None], x2]
        - cum[y2[:, None], x1]
        + cum[y1[:, None], x1]
    )
    return (box_sum / (size * size)).astype(img_array.dtype)


def _compute_deletion_drop(
    model: nn.Module,
    img_tensor: torch.Tensor,
    backbone_name: str,
    pred_idx: int,
    heatmap: np.ndarray,
    fraction: float = 0.15,
) -> float:
    """Calculate relative confidence drop when blurring the top 'fraction' of attended pixels."""
    # 1. Predict original probability
    with torch.no_grad():
        outputs = model(img_tensor)
        disease_out = outputs["disease_output"] if isinstance(outputs, dict) else outputs
        disease_probs = torch.softmax(disease_out, dim=-1)
        orig_prob = float(disease_probs[0][pred_idx].item())

    if orig_prob < 1e-8:
        return 0.0

    # 2. Extract original RGB image in [0, 255] from img_tensor
    rgb_img = img_tensor[0].detach().cpu().numpy().transpose((1, 2, 0)) * 255.0

    # 3. Create blurred image
    blurred_img = _simple_blur(rgb_img, size=15)

    # 4. Identify high-attention mask
    threshold = np.percentile(heatmap, (1.0 - fraction) * 100)
    mask = heatmap >= threshold

    # 5. Replace high attention areas with blurred areas
    perturbed_img = rgb_img.copy()
    perturbed_img[mask] = blurred_img[mask]

    # 6. Re-preprocess and predict
    perturbed_tensor = torch.from_numpy(perturbed_img.transpose((2, 0, 1))).unsqueeze(0).float() / 255.0
    perturbed_tensor = perturbed_tensor.to(img_tensor.device)

    with torch.no_grad():
        perturbed_outputs = model(perturbed_tensor)
        perturbed_disease_out = perturbed_outputs["disease_output"] if isinstance(perturbed_outputs, dict) else perturbed_outputs
        perturbed_probs = torch.softmax(perturbed_disease_out, dim=-1)
        perturbed_prob = float(perturbed_probs[0][pred_idx].item())

    # 7. Compute relative drop
    return float(max(0.0, (orig_prob - perturbed_prob) / orig_prob))


def _collect_sample_images(
    val_dir: str, num_samples: int, seed: int = 42
) -> list[tuple[str, str]]:
    """Collect random sample images from validation directory.

    Returns list of (file_path, class_name) tuples.
    """
    all_samples = []
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    for class_name in sorted(os.listdir(val_dir)):
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                all_samples.append(
                    (os.path.join(class_dir, fname), class_name)
                )

    rng = random.Random(seed)
    rng.shuffle(all_samples)
    return all_samples[:num_samples]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps to verify model attention."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to model to analyze (default: auto-detect best model).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sample images to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(str(PLOTS_DIR), "gradcam"),
        help="Directory to save Grad-CAM overlays.",
    )
    parser.add_argument(
        "--conv-layer",
        default=None,
        help="Name of conv layer to visualize (default: auto-detect last conv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--bg-removal",
        action="store_true",
        help="Apply background removal to images before running Grad-CAM.",
    )
    parser.add_argument(
        "--eis-threshold",
        type=float,
        default=80.0,
        help="Minimum required average Energy in Saliency (EiS) percentage to pass.",
    )
    parser.add_argument(
        "--deletion-threshold",
        type=float,
        default=30.0,
        help="Minimum required average Deletion Drop percentage to pass.",
    )
    parser.add_argument(
        "--hac-threshold",
        type=float,
        default=80.0,
        help="Minimum required Hierarchical Attention Consistency (HAC) passing rate percentage to pass.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any explainability validation threshold is not met.",
    )
    args = parser.parse_args()

    # Load model
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = resolve_pytorch_model_path([FINAL_MODEL_PATH])
    print(f"Loading model: {model_path}")

    model, backbone_name = _load_model_robust(model_path)
    print(f"Detected backbone: {backbone_name}")

    from src.utils.hardware import get_device
    device = get_device()
    model.to(device)

    # Load class indices and parse family structure
    idx_to_label = _load_class_indices(str(CLASS_INDICES_PATH))
    print(f"Loaded {len(idx_to_label)} class labels")
    class_names = [idx_to_label[idx] for idx in sorted(idx_to_label.keys())]
    from src.training.training_utils import parse_class_structure

    healthy_partners = parse_class_structure(class_names)

    # Set up targeting for Grad-CAM
    vit_block_idx = 10
    target_layer_name = args.conv_layer

    if backbone_name == "DINOv3":
        if args.conv_layer is not None:
            clean_layer = args.conv_layer.lower().strip()
            if clean_layer.startswith("block_"):
                try:
                    vit_block_idx = int(clean_layer.split("_")[1])
                    target_layer_name = None
                except ValueError:
                    pass
            else:
                try:
                    vit_block_idx = int(clean_layer)
                    target_layer_name = None
                except ValueError:
                    # Keep target_layer_name as is (e.g. "patch_embed")
                    pass
        else:
            target_layer_name = None

        target_desc = (
            f"ViT block {vit_block_idx}"
            if target_layer_name is None
            else f"ViT layer {target_layer_name}"
        )
        print(f"Using target layer for Grad-CAM: {target_desc}")
    else:
        if args.conv_layer:
            target_layer_name = args.conv_layer
            print(f"Using target layer for Grad-CAM: {target_layer_name}")
        else:
            target_layer_name = None
            print("Using target layer for Grad-CAM: last convolutional layer (auto-detected)")

    # Collect samples
    samples = _collect_sample_images(str(VAL_DIR), args.num_samples, args.seed)
    if not samples:
        print("ERROR: No sample images found in validation directory.")
        sys.exit(1)
    print(f"Collected {len(samples)} sample images")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize lists to store metrics for the dataset-level summary
    all_eis = []
    all_del_drops = []
    all_hac_satisfied = []

    # Process each sample
    for i, (img_path, true_class) in enumerate(samples):
        # Load image
        try:
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        except Exception as e:
            print(f"  Failed to load image {img_path}: {e}")
            continue
        original_img = np.array(img, dtype=np.float32)

        # Convert to tensor: shape (1, 3, 224, 224), range [0.0, 1.0]
        preprocessed_tensor = torch.from_numpy(original_img.transpose((2, 0, 1))).unsqueeze(0).float() / 255.0
        preprocessed_tensor = preprocessed_tensor.to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(preprocessed_tensor)
            disease_out = outputs["disease_output"] if isinstance(outputs, dict) else outputs
            disease_probs = torch.softmax(disease_out, dim=-1)
            pred_idx = int(torch.argmax(disease_probs[0]).item())
            pred_conf = float(disease_probs[0][pred_idx].item())
        pred_label = idx_to_label.get(pred_idx, f"class_{pred_idx}")

        # Generate Grad-CAM
        try:
            healthy_partner_idx = (
                healthy_partners[pred_idx]
                if pred_idx < len(healthy_partners)
                else -1
            )
            crop_heatmap, disease_heatmap = _make_gradcam_heatmap(
                model,
                preprocessed_tensor,
                target_layer_name=target_layer_name,
                pred_index=pred_idx,
                backbone_name=backbone_name,
                vit_block_idx=vit_block_idx,
                healthy_partner_idx=healthy_partner_idx,
            )
            # The metrics focus on the disease head for anomaly detection
            heatmap = disease_heatmap
        except Exception as exc:
            import traceback
            print(f"  Grad-CAM failed for {os.path.basename(img_path)}: {exc}")
            traceback.print_exc()
            continue

        # Segment leaf and background
        leaf_mask = _extract_leaf_mask(original_img)
        bg_mask = 1.0 - leaf_mask

        # Calculate Energy in Saliency (EiS)
        eis_score = float(
            np.sum(heatmap * leaf_mask) / (np.sum(heatmap) + 1e-8)
        )
        all_eis.append(eis_score)

        # Calculate Deletion Drop (blur 15% top pixels)
        del_drop = _compute_deletion_drop(
            model,
            preprocessed_tensor,
            backbone_name,
            pred_idx,
            heatmap,
            fraction=0.15,
        )
        all_del_drops.append(del_drop)

        # Calculate Hierarchical Attention Consistency (HAC) if applicable
        hac_info: dict[str, Any] | None = None
        healthy_partner_idx = (
            healthy_partners[pred_idx]
            if pred_idx < len(healthy_partners)
            else -1
        )
        is_healthy_class = "healthy" in pred_label.lower()

        if not is_healthy_class and healthy_partner_idx != -1:
            try:
                # Generate deviation heatmap
                _, deviation_heatmap = _make_gradcam_heatmap(
                    model,
                    preprocessed_tensor,
                    target_layer_name=target_layer_name,
                    pred_index=pred_idx,
                    backbone_name=backbone_name,
                    vit_block_idx=vit_block_idx,
                    healthy_partner_idx=healthy_partner_idx,
                )

                # Binarize deviation heatmap to form lesion mask
                lesion_mask = (deviation_heatmap > 0.4).astype(
                    np.float32
                ) * leaf_mask
                healthy_leaf_mask = leaf_mask * (1.0 - lesion_mask)

                sum_lesion = np.sum(lesion_mask)
                sum_healthy_leaf = np.sum(healthy_leaf_mask)
                sum_bg = np.sum(bg_mask)

                a_lesion = float(
                    np.sum(heatmap * lesion_mask) / (sum_lesion + 1e-8)
                )
                a_healthy = float(
                    np.sum(heatmap * healthy_leaf_mask)
                    / (sum_healthy_leaf + 1e-8)
                )
                a_bg = float(np.sum(heatmap * bg_mask) / (sum_bg + 1e-8))

                max_a = max(a_lesion, 1e-8)
                hac_ratio = (1.0, a_healthy / max_a, a_bg / max_a)
                hac_satisfied = a_lesion > a_healthy > a_bg

                hac_info = {
                    "a_lesion": a_lesion,
                    "a_healthy": a_healthy,
                    "a_bg": a_bg,
                    "hac_ratio": hac_ratio,
                    "satisfied": hac_satisfied,
                }
            except Exception as exc:
                print(f"    Failed to calculate HAC for {pred_label}: {exc}")
                hac_satisfied = False
        else:
            # For healthy class, define HAC using leaf vs background
            sum_leaf = np.sum(leaf_mask)
            sum_bg = np.sum(bg_mask)
            a_leaf = float(np.sum(heatmap * leaf_mask) / (sum_leaf + 1e-8))
            a_bg = float(np.sum(heatmap * bg_mask) / (sum_bg + 1e-8))
            max_a = max(a_leaf, 1e-8)
            hac_ratio = (1.0, a_bg / max_a, 0.0)
            hac_satisfied = a_leaf > a_bg
            hac_info = {
                "a_lesion": None,
                "a_healthy": a_leaf,
                "a_bg": a_bg,
                "hac_ratio": hac_ratio,
                "satisfied": hac_satisfied,
            }

        all_hac_satisfied.append(hac_satisfied)

        # Create overlay
        overlay = _overlay_heatmap(original_img, heatmap, alpha=0.4)

        # Determine if correct
        is_correct = true_class == pred_label
        status = "CORRECT" if is_correct else "WRONG"

        # Save
        safe_class = true_class.replace("/", "_").replace("\\", "_")
        out_name = f"{i:03d}_{safe_class}_{status}_{pred_conf:.2f}.png"
        out_path = os.path.join(args.output_dir, out_name)

        # Save overlay image
        overlay_img = Image.fromarray(overlay)
        overlay_img.save(out_path)

        # Print metrics
        hac_str = "N/A"
        if hac_info is not None:
            if hac_info["a_lesion"] is not None:
                ratio_healthy = hac_info["hac_ratio"][1]
                ratio_bg = hac_info["hac_ratio"][2]
                hac_str = (
                    f"Lesion: {hac_info['a_lesion']:.3f}, "
                    f"Healthy-Leaf: {hac_info['a_healthy']:.3f}, "
                    f"BG: {hac_info['a_bg']:.3f} (Ratio: 1.00 : "
                    f"{ratio_healthy:.2f} : {ratio_bg:.2f})"
                )
            else:
                ratio_bg = hac_info["hac_ratio"][1]
                hac_str = (
                    f"Leaf: {hac_info['a_healthy']:.3f}, "
                    f"BG: {hac_info['a_bg']:.3f} "
                    f"(Ratio: 1.00 : {ratio_bg:.2f})"
                )

        # Split log line to keep length under 79 chars
        print(
            f"  [{i + 1}/{len(samples)}] {status}: "
            f"true={true_class}, pred={pred_label} "
            f"({pred_conf:.3f}) -> {out_name}"
        )
        print(f"    - EiS (Energy in Saliency): {eis_score * 100:.2f}%")
        print(f"    - Deletion Drop: {del_drop * 100:.2f}%")
        hac_status_str = "PASSED" if hac_satisfied else "FAILED"
        print(f"    - HAC Score: {hac_status_str} | {hac_str}")

    mean_eis = np.mean(all_eis) * 100.0
    mean_del_drop = np.mean(all_del_drops) * 100.0
    hac_passing_rate = np.mean(all_hac_satisfied) * 100.0

    eis_passed = mean_eis >= args.eis_threshold
    deletion_passed = mean_del_drop >= args.deletion_threshold
    hac_passed = hac_passing_rate >= args.hac_threshold

    model_accepted = eis_passed and deletion_passed and hac_passed

    print(f"\nGrad-CAM overlays saved to: {args.output_dir}")
    print("\n==================================================")
    print("        EXPLAINABILITY VALIDATION SUMMARY         ")
    print("==================================================")
    print(
        f"Average Energy in Saliency (EiS): {mean_eis:.2f}% "
        f"(Threshold: {args.eis_threshold:.1f}%) -> "
        f"{'PASSED' if eis_passed else 'FAILED'}"
    )
    print(
        f"Average Deletion Drop: {mean_del_drop:.2f}% "
        f"(Threshold: {args.deletion_threshold:.1f}%) -> "
        f"{'PASSED' if deletion_passed else 'FAILED'}"
    )
    print(
        f"HAC Passing Rate: {hac_passing_rate:.2f}% "
        f"(Threshold: {args.hac_threshold:.1f}%) -> "
        f"{'PASSED' if hac_passed else 'FAILED'}"
    )
    print("--------------------------------------------------")
    print(
        f"EXPLAINABILITY VERIFICATION: "
        f"{'ACCEPTED' if model_accepted else 'REJECTED'}"
    )
    print("==================================================")
    print(
        "\nInterpretation guide:\n"
        "  - RED/WARM areas = high attention (what the model looks at)\n"
        "  - BLUE/COOL areas = low attention\n"
        "  - Background focus = shortcut learning (unacceptable)\n"
        "  - Leaf/disease focus = correct learning"
    )

    if args.strict and not model_accepted:
        print("\nERROR: Model failed explainability validation requirements.")
        sys.exit(1)


if __name__ == "__main__":
    main()
