#!/usr/bin/env python3
"""Generate side-by-side comparative architecture visualization for EfficientNetV2 vs DINOv3."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def create_comparative_architecture():
    """Generate side-by-side comparison of EfficientNetV2 vs DINOv3."""
    fig = plt.figure(figsize=(16, 11.2))

    # Create two subplots side by side
    ax_eff = fig.add_subplot(1, 2, 1)
    ax_dino = fig.add_subplot(1, 2, 2)

    # ==================== EfficientNetV2-B0 (Left) ====================
    ax_eff.set_xlim(0, 10)
    ax_eff.set_ylim(0, 14)
    ax_eff.axis("off")

    eff_layers = [
        ("Input\n224×224×3 RGB", "#87CEEB", 0.7, "Input"),
        (
            "MBConv Blocks\n(Efficient Inverted Bottleneck)\n6 blocks, 128 filters",
            "#90EE90",
            1.2,
            "Backbone\nStart",
        ),
        (
            "MBConv Blocks\n9 blocks, 160 filters",
            "#90EE90",
            1.2,
            "Backbone\nMid",
        ),
        (
            "MBConv Blocks\n15 blocks, 256 filters",
            "#90EE90",
            1.2,
            "Backbone\nEnd",
        ),
        ("Conv 1×1 Head\n1280 filters", "#FFE4B5", 0.8, "Head"),
        ("GlobalAvgPool\n1280-dim", "#FFD700", 0.6, "Pool"),
        ("BatchNormalization\n1280-dim", "#FFFACD", 0.6, "Norm"),
        ("Dense(512) + Swish\nDropout(0.4)", "#F08080", 0.8, "Dense 1"),
        ("Dense(256) + Swish\nDropout(0.2)", "#F08080", 0.8, "Dense 2"),
        (
            "Dense(46) + Softmax\n46 classes (13 healthy, 33 disease)",
            "#98FB98",
            0.7,
            "Output",
        ),
    ]

    y_start = 13.2
    y_spacing = 1.3

    for i, (desc, color, height, label) in enumerate(eff_layers):
        y = y_start - i * y_spacing

        # Draw box
        rect = FancyBboxPatch(
            (1, y - height / 2),
            8,
            height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="darkblue",
            linewidth=2,
            alpha=0.85,
        )
        ax_eff.add_patch(rect)

        # Add text
        ax_eff.text(
            5,
            y,
            desc,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            wrap=True,
        )

        # Draw arrow to next layer
        if i < len(eff_layers) - 1:
            ax_eff.annotate(
                "",
                xy=(5, y - height / 2 - 0.1),
                xytext=(5, y - height / 2 - 0.4),
                arrowprops=dict(arrowstyle="->", color="darkblue", lw=2.5),
            )

    # Add title and info box for EfficientNetV2
    ax_eff.text(
        5,
        13.8,
        "EfficientNetV2-B0 (CNN)",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

    info_eff = (
        "Backbone: Efficient Inverted Bottleneck (MBConv)\n"
        "Total Parameters: ~7.2M\n"
        "Inference Speed: ~50ms per image\n"
        "Model Size: ~29MB\n"
        "Head: BatchNorm -> Dense(512, swish) -> Dropout(0.4)\n"
        "      Dense(256, swish) -> Dropout(0.2) -> Softmax\n"
        "Pre-training: ImageNet-1k supervised"
    )

    ax_eff.text(
        5,
        0.8,
        info_eff,
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.7),
        family="monospace",
    )

    # ==================== DINOv3 ViT (Right) ====================
    ax_dino.set_xlim(0, 10)
    ax_dino.set_ylim(0, 14)
    ax_dino.axis("off")

    dino_layers = [
        ("Input\n224×224×3 RGB", "#87CEEB", 0.7, "Input"),
        (
            "Patch Embedding\n16×16 patches -> token sequence",
            "#FFB6C1",
            1.0,
            "Tokenize",
        ),
        (
            "Transformer Blocks ×12\n(Multi-Head Attention + FFN)",
            "#90EE90",
            1.4,
            "Transformer",
        ),
        ("Backbone Output Embedding\nfeature vector", "#FFD700", 0.7, "Pool"),
        ("BatchNormalization", "#FFFACD", 0.6, "Norm"),
        ("Dense(512) + Swish\nDropout(0.4)", "#F08080", 0.8, "Dense 1"),
        ("Dense(256) + Swish\nDropout(0.2)", "#F08080", 0.8, "Dense 2"),
        (
            "Dense(46) + Softmax\n46 classes (13 healthy, 33 disease)",
            "#98FB98",
            0.7,
            "Output",
        ),
    ]

    y_start = 13.2
    y_spacing = 1.22

    for i, (desc, color, height, label) in enumerate(dino_layers):
        y = y_start - i * y_spacing

        # Draw box
        rect = FancyBboxPatch(
            (1, y - height / 2),
            8,
            height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="darkgreen",
            linewidth=2,
            alpha=0.85,
        )
        ax_dino.add_patch(rect)

        # Add text
        ax_dino.text(
            5,
            y,
            desc,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            wrap=True,
        )

        # Draw arrow to next layer
        if i < len(dino_layers) - 1:
            ax_dino.annotate(
                "",
                xy=(5, y - height / 2 - 0.1),
                xytext=(5, y - height / 2 - 0.4),
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2.5),
            )

    # Add title and info box for DINOv3
    ax_dino.text(
        5,
        13.8,
        "DINOv3 Vision Transformer",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

    info_dino = (
        "Backbone: Vision Transformer (ViT-Base)\n"
        "Total Parameters: ~87M (about 4.1x larger)\n"
        "Inference Speed: ~150-200ms per image\n"
        "Model Size: ~350MB (about 23x larger)\n"
        "Head: BatchNorm -> Dense(512, swish) -> Dropout(0.4)\n"
        "      Dense(256, swish) -> Dropout(0.2) -> Softmax\n"
        "Pre-training: DINO-style self-supervision"
    )

    ax_dino.text(
        5,
        0.8,
        info_dino,
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgreen", alpha=0.7),
        family="monospace",
    )

    # ==================== Overall Figure ====================
    fig.suptitle(
        "Backbone Architecture Comparison: EfficientNetV2-B0 vs DINOv3",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Keep footer compact to avoid cramped rendering.
    comparison_text = (
        "Key differences: CNN vs ViT | Params: ~7.2M vs ~87M | "
        "Latency: ~50ms vs ~150-200ms | Head: shared BatchNorm + Swish MLP"
    )

    fig.text(
        0.5,
        0.035,
        comparison_text,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8
        ),
    )

    plt.tight_layout(rect=[0, 0.09, 1, 0.96])
    output_path = PLOTS_DIR / "backbone_architecture_comparison.png"
    plt.savefig(output_path, dpi=800, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_comparative_architecture()
    print("\n" + "=" * 70)
    print("Comparative architecture visualization created successfully")
    print("=" * 70)
