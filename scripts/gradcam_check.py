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

import argparse
import json
import os
import random
import sys

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import tensorflow.keras as keras

from config import (
    CLASS_INDICES_PATH,
    FINAL_MODEL_PATH,
    IMG_SIZE,
    PLOTS_DIR,
    VAL_DIR,
)
from model_paths import resolve_keras_model_path
from preprocessing import preprocess_array_for_model

# background_remover import bypassed


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


def _load_model_robust(model_path: str):
    """Load model with compatibility fallbacks for DINO/KerasHub checkpoints."""
    from training_utils import WarmupCosineSchedule as _WCS

    custom_objects = {"WarmupCosineSchedule": _WCS}
    try:
        return keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False
        )
    except TypeError as exc:
        error_text = str(exc)
        if "ViTPatchingAndEmbedding" not in error_text:
            raise

        if not _patch_vit_layer_init_for_compat():
            raise RuntimeError(
                "Failed to load ViT/DINO checkpoint due to keras-hub version mismatch."
            ) from exc

        print(
            "Detected KerasHub ViT checkpoint compatibility mismatch; "
            "retrying load with compatibility shim."
        )
        return keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False
        )


def _load_class_indices(path: str) -> dict[int, str]:
    """Load class_indices.json and return idx -> label mapping."""
    with open(path, "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)
    return {int(v): k for k, v in label_to_idx.items()}


def _infer_backbone_name(model) -> str:
    """Best-effort backbone detection from model layers."""
    for layer in getattr(model, "layers", []):
        layer_name = (getattr(layer, "name", "") or "").lower()
        if any(tok in layer_name for tok in ["vit", "dino", "transformer"]):
            return "DINOv3"
    return "EfficientNetV2B0"


def _find_target_layer(model) -> str | None:
    """Find the name of the target layer (last conv or vit_encoder)."""
    # Check for ViT encoder first
    for layer in model.layers:
        if layer.name == "vit_encoder":
            return layer.name

    # Fallback to last conv
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Conv2D,)):
            last_conv = layer.name
        # Also check inside nested models (like the backbone)
        if hasattr(layer, "layers"):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, (keras.layers.Conv2D,)):
                    last_conv = sub_layer.name
    return last_conv


def _find_conv_layer_in_model(model, layer_name: str):
    """Find a layer by name, searching nested models too."""
    try:
        return model.get_layer(layer_name)
    except ValueError:
        pass
    for layer in model.layers:
        if hasattr(layer, "get_layer"):
            try:
                return layer.get_layer(layer_name)
            except ValueError:
                continue
    return None


def _make_gradcam_heatmap(
    model,
    img_array: np.ndarray,
    target_layer_name: str | None = None,
    pred_index: int | None = None,
    backbone_name: str = "DINOv3",
    vit_block_idx: int = 6,
    healthy_partner_idx: int | None = None,
) -> np.ndarray:
    """Generate Grad-CAM heatmap for a single image.

    Args:
        model: Keras model.
        img_array: Preprocessed image array of shape (1, H, W, 3).
        target_layer_name: Name of the target layer (used for CNN/Conv2D models).
        pred_index: Class index to visualize. None = top predicted class.
        backbone_name: Detected backbone name (e.g. "DINOv3" or "EfficientNetV2B0").
        vit_block_idx: The block index of the ViT backbone to visualize (default: 6).
        healthy_partner_idx: Healthy baseline index for FBDL deviation computation.

    Returns:
        Heatmap array of shape (H, W) with values in [0, 1].
    """
    if backbone_name == "DINOv3":
        # ViT Grad-CAM using custom forward pass to capture intermediate block activations
        # and compute gradients of raw class logits (before softmax)
        vit_encoder = model.get_layer("vit_encoder")
        patch_embed = model.get_layer("vit_patching_and_embedding")
        pool = model.get_layer("global_average_pooling1d")

        with tf.GradientTape() as tape:
            # Step 1: Patching & Embedding
            pe_out = patch_embed(img_array)
            tape.watch(pe_out)

            # Step 2: ViT Encoder Blocks
            x = vit_encoder.dropout(pe_out)

            target_activation = None
            if target_layer_name == "patch_embed":
                target_activation = pe_out

            for i in range(vit_encoder.num_layers):
                x = vit_encoder.encoder_layers[i](x)
                if target_layer_name != "patch_embed" and i == vit_block_idx:
                    target_activation = x
                    tape.watch(target_activation)

            final_vit_out = vit_encoder.layer_norm(x)
            if target_layer_name == "vit_encoder_final":
                target_activation = final_vit_out
                tape.watch(target_activation)

            # Step 3: Classification Head
            # Dynamically run all remaining layers sequentially to compute logits
            x_head = pool(final_vit_out)
            for layer in model.layers[4:10]:
                x_head = layer(x_head)
            logits = x_head

            if pred_index is None:
                pred_index = tf.argmax(logits[0])

            if healthy_partner_idx is not None and healthy_partner_idx != -1:
                score = logits[:, pred_index] - logits[:, healthy_partner_idx]
            else:
                score = logits[:, pred_index]

        grads = tape.gradient(score, target_activation)
        if grads is None:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        # ViT standard Grad-CAM: average gradients over token dimension
        pooled_grads = tf.reduce_mean(grads, axis=1)  # (1, 768)
        heatmap = tf.reduce_sum(
            target_activation[0] * pooled_grads[0], axis=-1
        )  # (197,)

        # Discard CLS token if present (197 -> 196)
        if heatmap.shape[0] % 2 != 0:
            heatmap = heatmap[1:]

        grid_size = int(np.sqrt(heatmap.shape[0]))
        heatmap = tf.reshape(heatmap, (grid_size, grid_size))

    else:
        # Standard CNN (Conv2D) Grad-CAM
        target_layer = _find_conv_layer_in_model(model, target_layer_name)
        if target_layer is None:
            raise ValueError(
                f"Layer '{target_layer_name}' not found in model."
            )

        original_activation = model.layers[-1].activation
        model.layers[-1].activation = tf.keras.activations.linear

        try:
            grad_model = keras.Model(
                inputs=model.input,
                outputs=[target_layer.output, model.output],
            )

            with tf.GradientTape() as tape:
                outputs, predictions = grad_model(img_array, training=False)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])

                if (
                    healthy_partner_idx is not None
                    and healthy_partner_idx != -1
                ):
                    class_channel = (
                        predictions[:, pred_index]
                        - predictions[:, healthy_partner_idx]
                    )
                else:
                    class_channel = predictions[:, pred_index]

            grads = tape.gradient(class_channel, outputs)
        finally:
            model.layers[-1].activation = original_activation

        if grads is None:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        outputs = outputs[0]
        heatmap = outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-5)
    heatmap = heatmap.numpy()

    heatmap = tf.image.resize(
        heatmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE)
    ).numpy()[:, :, 0]

    return heatmap


def _overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a heatmap on the original image.

    Args:
        original_img: Original image (H, W, 3) in [0, 255].
        heatmap: Heatmap (H, W) in [0, 1].
        alpha: Overlay transparency.

    Returns:
        Overlaid image (H, W, 3) in [0, 255] as uint8.
    """
    # Create a simple jet colormap
    jet_r = np.clip(1.5 - np.abs(4 * heatmap - 3), 0, 1)
    jet_g = np.clip(1.5 - np.abs(4 * heatmap - 2), 0, 1)
    jet_b = np.clip(1.5 - np.abs(4 * heatmap - 1), 0, 1)
    colored_heatmap = np.stack([jet_r, jet_g, jet_b], axis=-1) * 255

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
    model,
    img_array: np.ndarray,
    backbone_name: str,
    pred_idx: int,
    heatmap: np.ndarray,
    fraction: float = 0.15,
) -> float:
    """Calculate relative confidence drop when blurring the top 'fraction' of attended pixels."""
    # 1. Predict original probability
    original_preds = model.predict(img_array, verbose=0)
    orig_prob = float(original_preds[0][pred_idx])
    if orig_prob < 1e-8:
        return 0.0

    # 2. Extract or reconstruct de-preprocessed RGB image in [0, 255]
    if backbone_name == "DINOv3":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_img = (img_array[0] * std + mean) * 255.0
    else:
        rgb_img = img_array[0].copy()

    # 3. Create blurred image
    blurred_img = _simple_blur(rgb_img, size=15)

    # 4. Identify high-attention mask
    threshold = np.percentile(heatmap, (1.0 - fraction) * 100)
    mask = heatmap >= threshold

    # 5. Replace high attention areas with blurred areas
    perturbed_img = rgb_img.copy()
    perturbed_img[mask] = blurred_img[mask]

    # 6. Re-preprocess and predict
    if backbone_name == "DINOv3":
        perturbed_img_prep = perturbed_img / 255.0
        perturbed_img_prep = (perturbed_img_prep - mean) / std
    else:
        perturbed_img_prep = perturbed_img

    perturbed_batch = perturbed_img_prep[np.newaxis, ...]
    perturbed_preds = model.predict(perturbed_batch, verbose=0)
    perturbed_prob = float(perturbed_preds[0][pred_idx])

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
        model_path = resolve_keras_model_path([FINAL_MODEL_PATH])
    print(f"Loading model: {model_path}")

    model = _load_model_robust(model_path)

    backbone_name = _infer_backbone_name(model)
    print(f"Detected backbone: {backbone_name}")

    # Load class indices and parse family structure
    idx_to_label = _load_class_indices(str(CLASS_INDICES_PATH))
    print(f"Loaded {len(idx_to_label)} class labels")
    class_names = [idx_to_label[idx] for idx in sorted(idx_to_label.keys())]
    from training_utils import parse_class_structure

    healthy_partners = parse_class_structure(class_names)

    # Set up targeting for Grad-CAM
    vit_block_idx = 6
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
        target_layer_name = args.conv_layer or _find_target_layer(model)
        if target_layer_name is None:
            print("ERROR: Could not find a convolutional layer in the model.")
            sys.exit(1)
        print(f"Using target layer for Grad-CAM: {target_layer_name}")

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
        # Load and preprocess image
        img = keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = keras.utils.img_to_array(img)
        original_img = img_array.copy()

        # Preprocess for model
        preprocessed = preprocess_array_for_model(
            img_array[np.newaxis, ...], backbone_name=backbone_name
        )

        # Get prediction
        preds = model.predict(preprocessed, verbose=0)
        pred_idx = int(np.argmax(preds[0]))
        pred_conf = float(preds[0][pred_idx])
        pred_label = idx_to_label.get(pred_idx, f"class_{pred_idx}")

        # Generate Grad-CAM
        try:
            heatmap = _make_gradcam_heatmap(
                model,
                preprocessed,
                target_layer_name=target_layer_name,
                pred_index=pred_idx,
                backbone_name=backbone_name,
                vit_block_idx=vit_block_idx,
            )
        except Exception as exc:
            print(f"  Grad-CAM failed for {os.path.basename(img_path)}: {exc}")
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
            preprocessed,
            backbone_name,
            pred_idx,
            heatmap,
            fraction=0.15,
        )
        all_del_drops.append(del_drop)

        # Calculate Hierarchical Attention Consistency (HAC) if applicable
        hac_info = None
        healthy_partner_idx = (
            healthy_partners[pred_idx]
            if pred_idx < len(healthy_partners)
            else -1
        )
        is_healthy_class = "healthy" in pred_label.lower()

        if not is_healthy_class and healthy_partner_idx != -1:
            try:
                # Generate deviation heatmap
                deviation_heatmap = _make_gradcam_heatmap(
                    model,
                    preprocessed,
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

        # Save using tf.io to avoid matplotlib dependency
        overlay_tensor = tf.constant(overlay, dtype=tf.uint8)
        encoded = tf.io.encode_png(overlay_tensor)
        tf.io.write_file(out_path, encoded)

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
