from __future__ import annotations

import os
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_reliability_diagram(
    calibration: Mapping[str, object],
    out_path: str,
    title: str = "Reliability Diagram",
) -> str:
    """Render and save a reliability diagram from calibration statistics."""
    bin_edges = np.asarray(calibration.get("bin_edges", []), dtype=np.float64)
    bin_accuracy = np.asarray(calibration.get("bin_accuracy", []), dtype=np.float64)
    bin_confidence = np.asarray(calibration.get("bin_confidence", []), dtype=np.float64)
    bin_counts = np.asarray(calibration.get("bin_counts", []), dtype=np.float64)
    ece = float(calibration.get("ece", 0.0))

    if bin_edges.size < 2:
        raise ValueError("Calibration payload must include non-empty bin_edges.")

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)

    fig, (ax_main, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(8.5, 9.0),
        gridspec_kw={"height_ratios": [3.0, 1.2]},
        constrained_layout=True,
    )

    ax_main.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", linewidth=1.2)
    ax_main.bar(
        centers,
        bin_accuracy,
        width=widths * 0.9,
        align="center",
        alpha=0.75,
        color="#2a9d8f",
        edgecolor="#1d6f67",
        label="Empirical accuracy",
    )
    ax_main.plot(
        centers,
        bin_confidence,
        color="#e76f51",
        marker="o",
        linewidth=1.8,
        label="Mean confidence",
    )
    ax_main.set_xlim(0.0, 1.0)
    ax_main.set_ylim(0.0, 1.05)
    ax_main.set_xlabel("Confidence")
    ax_main.set_ylabel("Accuracy")
    ax_main.set_title(f"{title} (ECE={ece:.4f})")
    ax_main.legend(frameon=False)

    total = max(1.0, float(np.sum(bin_counts)))
    proportions = bin_counts / total
    ax_hist.bar(
        centers,
        proportions,
        width=widths * 0.9,
        align="center",
        color="#457b9d",
        alpha=0.75,
        edgecolor="#2d4f66",
    )
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_xlabel("Confidence")
    ax_hist.set_ylabel("Sample proportion")
    ax_hist.set_title("Prediction confidence histogram")

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
