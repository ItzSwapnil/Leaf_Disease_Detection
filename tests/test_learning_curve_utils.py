from learning_curve_utils import (
    best_epoch_from_values,
    build_best_epoch_markers,
    combine_train_and_fine_metrics,
    trim_train_metrics_to_restore_epoch,
)


def _metrics(acc, val_acc, loss, val_loss):
    return {
        "accuracy": list(acc),
        "val_accuracy": list(val_acc),
        "loss": list(loss),
        "val_loss": list(val_loss),
    }


def test_trim_train_metrics_to_restore_epoch_trims_tail():
    train = _metrics(
        acc=[0.1, 0.2, 0.3, 0.4],
        val_acc=[0.15, 0.25, 0.35, 0.45],
        loss=[1.0, 0.8, 0.7, 0.6],
        val_loss=[1.1, 0.9, 0.8, 0.7],
    )

    trimmed, effective_len, dropped = trim_train_metrics_to_restore_epoch(
        train, [3]
    )

    assert effective_len == 3
    assert dropped == 1
    assert trimmed["accuracy"] == [0.1, 0.2, 0.3]
    assert trimmed["val_accuracy"] == [0.15, 0.25, 0.35]


def test_trim_train_metrics_to_restore_epoch_ignores_out_of_range_restore():
    train = _metrics(
        acc=[0.1, 0.2, 0.3],
        val_acc=[0.2, 0.3, 0.4],
        loss=[1.0, 0.9, 0.8],
        val_loss=[1.1, 1.0, 0.9],
    )

    trimmed, effective_len, dropped = trim_train_metrics_to_restore_epoch(
        train, [99]
    )

    assert effective_len == 3
    assert dropped == 0
    assert trimmed["accuracy"] == [0.1, 0.2, 0.3]


def test_combine_train_and_fine_metrics_uses_train_len_as_phase_boundary():
    train = _metrics(
        acc=[0.1, 0.2, 0.3],
        val_acc=[0.1, 0.2, 0.25],
        loss=[1.0, 0.9, 0.8],
        val_loss=[1.1, 1.0, 0.9],
    )
    fine = _metrics(
        acc=[0.31, 0.32],
        val_acc=[0.26, 0.27],
        loss=[0.79, 0.78],
        val_loss=[0.89, 0.88],
    )

    combined, phase_boundary = combine_train_and_fine_metrics(
        train, fine, epochs_phase1=25
    )

    assert phase_boundary == 3
    assert combined["accuracy"] == [0.1, 0.2, 0.3, 0.31, 0.32]


def test_build_best_epoch_markers_maps_fine_best_to_global_epoch():
    marker_lines, marker_note, train_best, fine_best, fine_global = (
        build_best_epoch_markers(
            train_val_acc=[0.6, 0.7, 0.8, 0.75],
            fine_val_acc=[0.81, 0.83, 0.82],
            phase_boundary=21,
        )
    )

    assert train_best == 3
    assert fine_best == 2
    assert fine_global == 23
    assert marker_note is not None
    labels = [entry[3] for entry in marker_lines]
    assert "Train Best (epoch 3)" in labels
    assert "Fine-Tune Best (epoch 23)" in labels


def test_best_epoch_from_values_handles_empty_and_nan():
    assert best_epoch_from_values([]) is None
    assert best_epoch_from_values([float("nan"), float("nan")]) is None
