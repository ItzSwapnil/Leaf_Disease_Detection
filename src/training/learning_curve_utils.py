from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MetricSeries = Dict[str, List[float]]
MetricMarker = Tuple[float, str, str, str]
METRIC_KEYS = ("accuracy", "val_accuracy", "loss", "val_loss")


def clone_metric_series(metrics: MetricSeries) -> MetricSeries:

    return {key: list(metrics.get(key, [])) for key in METRIC_KEYS}


def trim_train_metrics_to_restore_epoch(
    train_metrics: MetricSeries,
    restore_epochs: Sequence[int],
) -> Tuple[MetricSeries, int, int]:

    trimmed = clone_metric_series(train_metrics)
    original_len = len(trimmed["accuracy"])
    effective_len = original_len

    if restore_epochs:
        candidate = int(restore_epochs[-1])
        if 1 <= candidate <= original_len:
            effective_len = candidate

    for key in METRIC_KEYS:
        trimmed[key] = trimmed[key][:effective_len]

    dropped = max(0, original_len - effective_len)
    return trimmed, effective_len, dropped


def combine_train_and_fine_metrics(
    train_metrics: MetricSeries,
    fine_metrics: MetricSeries,
    epochs_phase1: int,
) -> Tuple[MetricSeries, int]:

    train_len = len(train_metrics["accuracy"])
    fine_len = len(fine_metrics["accuracy"])
    combined: MetricSeries = {
        key: train_metrics[key] + fine_metrics[key] for key in METRIC_KEYS
    }

    if fine_len > 0:
        phase_boundary = train_len
    else:
        phase_boundary = max(
            0, min(int(epochs_phase1), len(combined["accuracy"]))
        )

    return combined, phase_boundary


def best_epoch_from_values(values: Sequence[float]) -> Optional[int]:

    arr = np.array(values, dtype=np.float64)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None
    return int(np.nanargmax(arr)) + 1


def build_best_epoch_markers(
    train_val_acc: Sequence[float],
    fine_val_acc: Sequence[float],
    phase_boundary: int,
) -> Tuple[
    List[MetricMarker],
    Optional[str],
    Optional[int],
    Optional[int],
    Optional[int],
]:

    train_best_local = best_epoch_from_values(train_val_acc)
    fine_best_local = best_epoch_from_values(fine_val_acc)

    marker_lines: List[MetricMarker] = []
    marker_note: Optional[str] = None
    fine_best_global: Optional[int] = None

    if train_best_local is not None:
        marker_lines.append(
            (
                float(train_best_local),
                "darkorange",
                ":",
                f"Train Best (epoch {train_best_local})",
            )
        )

    if fine_best_local is not None and phase_boundary > 0:
        fine_best_global = int(phase_boundary) + int(fine_best_local)
        marker_lines.append(
            (
                float(fine_best_global),
                "green",
                ":",
                f"Fine-Tune Best (epoch {fine_best_global})",
            )
        )
        marker_note = (
            f"Fine-tune local best: {fine_best_local} -> "
            f"global best marker: {fine_best_global}"
        )

    return (
        marker_lines,
        marker_note,
        train_best_local,
        fine_best_local,
        fine_best_global,
    )
