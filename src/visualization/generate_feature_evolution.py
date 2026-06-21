#!/usr/bin/env python3
"""Generate feature evolution and capabilities matrix visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def create_feature_evolution_plot():
    """Generate feature evolution from old model (v1) to new model (v2+)."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # ==================== LEFT: Feature Timeline ====================
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 15)
    ax1.axis("off")

    ax1.text(
        5,
        14.5,
        "Feature Evolution Timeline",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    # Version 1 (Old)
    v1_box = Rectangle(
        (0.5, 11), 4, 2.5, facecolor="#FFE4E1", edgecolor="red", linewidth=2.5
    )
    ax1.add_patch(v1_box)
    ax1.text(
        2.5,
        13,
        "V1 (Old Model)\n2a8a3f3 commit",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    ax1.text(
        0.7,
        12.3,
        "✓ EfficientNetV2-S only\n✓ Basic classification head\n✓ Validation accuracy\n✓ Confusion matrix",
        fontsize=9,
        va="top",
        family="monospace",
    )

    # Arrow down
    ax1.annotate(
        "",
        xy=(2.5, 10.8),
        xytext=(2.5, 11),
        arrowprops=dict(arrowstyle="->", color="gray", lw=3),
    )

    # Version 2+ (New)
    v2_box = Rectangle(
        (0.5, 2), 9, 8.5, facecolor="#E1F5E1", edgecolor="green", linewidth=2.5
    )
    ax1.add_patch(v2_box)
    ax1.text(
        5,
        10,
        "V2+ (Current Model)",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    new_features = [
        (
            "Architecture",
            [
                "✅ 7 EfficientNetV2 variants (B0-L)",
                "✅ DINOv3 Vision Transformer",
                "✅ Flexible backbone factory",
            ],
        ),
        (
            "Calibration & Reliability",
            [
                "✅ Expected Calibration Error (ECE)",
                "✅ Maximum Calibration Error (MCE)",
                "✅ Brier Score",
                "✅ Temperature scaling optimization",
            ],
        ),
        (
            "Uncertainty Quantification",
            [
                "✅ MC Dropout for uncertainty",
                "✅ Prediction entropy calculation",
                "✅ Confidence rejection threshold",
            ],
        ),
        (
            "Robustness Testing",
            [
                "✅ Blur degradation testing",
                "✅ Brightness shift tolerance",
                "✅ Gaussian noise robustness",
                "✅ Fog simulation stress test",
                "✅ Occlusion robustness",
            ],
        ),
        (
            "Out-of-Distribution Detection",
            [
                "✅ Mahalanobis distance scoring",
                "✅ MSP (Maximum Softmax Prob)",
                "✅ OOD rejection capability",
            ],
        ),
        (
            "Statistical Validation",
            [
                "✅ Bootstrap confidence intervals",
                "✅ McNemar significance testing",
                "✅ Per-class performance metrics",
                "✅ Per-crop accuracy analysis",
            ],
        ),
        (
            "Safety & Inference Guards",
            [
                "✅ 5+ rejection strategies",
                "✅ Leaf likelihood assessment",
                "✅ Prediction diagnostics",
            ],
        ),
    ]

    y_pos = 9.5
    for category, items in new_features:
        ax1.text(
            0.8,
            y_pos,
            f"📦 {category}",
            fontsize=9.5,
            fontweight="bold",
            color="darkgreen",
        )
        y_pos -= 0.4
        for item in items:
            ax1.text(1.2, y_pos, item, fontsize=8.5, family="monospace")
            y_pos -= 0.3
        y_pos -= 0.1

    # ==================== RIGHT: Capability Matrix ====================
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 15)
    ax2.axis("off")

    ax2.text(
        5,
        14.5,
        "Capability Comparison Matrix",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    # Create comparison table
    capabilities = [
        ("Multi-backbone support", "❌", "✅", "EfficientNetV2 + DINOv3"),
        ("Basic inference", "✅", "✅", "Works in both"),
        ("Calibration", "❌", "✅", "ECE, MCE, Brier Score"),
        ("Uncertainty quantification", "❌", "✅", "MC Dropout + entropy"),
        ("Robustness testing", "❌", "✅", "5 stress tests"),
        ("OOD detection", "❌", "✅", "Mahalanobis + MSP"),
        ("Statistical validation", "❌", "✅", "Bootstrap + McNemar"),
        ("Inference guards", "❌", "✅", "5+ safety strategies"),
        ("Per-class metrics", "❌", "✅", "Detailed performance"),
        ("Production-grade", "⚠️", "✅", "Enterprise-ready"),
    ]

    # Header
    header_y = 13.5
    ax2.text(1.2, header_y, "Feature", fontsize=10, fontweight="bold")
    ax2.text(
        3.2, header_y, "V1 (Old)", fontsize=10, fontweight="bold", ha="center"
    )
    ax2.text(
        5.2, header_y, "V2+ (New)", fontsize=10, fontweight="bold", ha="center"
    )
    ax2.text(7.5, header_y, "Details", fontsize=10, fontweight="bold")

    # Draw separator line
    ax2.plot([0.5, 9.5], [13.2, 13.2], "k-", linewidth=1.5)

    # Rows
    y = 12.8
    for feature, v1, v2, detail in capabilities:
        # Feature name
        ax2.text(0.7, y, feature, fontsize=8.5, va="center")

        # V1 status
        color_v1 = (
            "#FFE4E1" if v1 == "❌" else "#FFE4E1" if v1 == "⚠️" else "#E1F5E1"
        )
        ax2.text(
            3.2,
            y,
            v1,
            fontsize=11,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color_v1),
        )

        # V2 status
        ax2.text(
            5.2,
            y,
            v2,
            fontsize=11,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E1F5E1"),
        )

        # Details
        ax2.text(7.5, y, detail, fontsize=8, va="center", style="italic")

        y -= 0.9

    # Add legend
    legend_y = 1.2
    ax2.text(
        0.7,
        legend_y,
        "✅ = Implemented  |  ⚠️ = Partial  |  ❌ = Not Available",
        fontsize=9,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7
        ),
    )

    # ==================== Overall Figure ====================
    fig.suptitle(
        "Leaf Disease Detection: Model Evolution & Capability Matrix\nNew Features Unlock Production-Grade Deployment",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = PLOTS_DIR / "feature_evolution_and_capabilities.png"
    plt.savefig(output_path, dpi=800, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_feature_evolution_plot()
    print("\n" + "=" * 70)
    print("📊 Feature evolution and capabilities visualization created!")
    print("=" * 70)
