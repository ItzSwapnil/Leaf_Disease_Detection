"""Utilities for emitting machine-readable training progress to stdout."""

import json
import time
import keras


class ProgressEmitter(keras.callbacks.Callback):
    """Emit line-delimited JSON progress records for external job monitors."""

    def __init__(self, stage, total_epochs, completed_epochs_before=0, run_start_time=None, min_emit_interval=4.0):
        super().__init__()
        self.stage = stage
        self.total_epochs = max(1, int(total_epochs))
        self.completed_epochs_before = max(0, int(completed_epochs_before))
        self.run_start_time = run_start_time or time.time()
        self.min_emit_interval = float(min_emit_interval)
        self._current_epoch = 0
        self._steps_in_epoch = None
        self._last_emit_time = 0.0

    def _emit(self, progress_pct, eta_seconds, epoch_done):
        payload = {
            "stage": self.stage,
            "progress_pct": round(float(progress_pct), 2),
            "eta_seconds": max(0.0, float(eta_seconds)),
            "eta_scope": "whole_process",
            "epoch_done": int(epoch_done),
            "total_epochs": int(self.total_epochs),
            "timestamp": round(time.time(), 2),
        }
        print(f"COPILOT_PROGRESS {json.dumps(payload)}", flush=True)

    def _estimate_eta(self, completed_units):
        if completed_units <= 0.0:
            return 0.0
        elapsed = max(0.0, time.time() - self.run_start_time)
        avg_seconds_per_unit = elapsed / float(completed_units)
        remaining_units = max(0.0, float(self.total_epochs) - completed_units)
        return avg_seconds_per_unit * remaining_units

    def on_train_begin(self, logs=None):
        self._steps_in_epoch = self.params.get("steps")
        self._last_emit_time = 0.0
        self._emit((self.completed_epochs_before / self.total_epochs) * 100.0, 0.0, self.completed_epochs_before)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = int(epoch)

    def on_train_batch_end(self, batch, logs=None):
        if not self._steps_in_epoch:
            return
        now = time.time()
        if now - self._last_emit_time < self.min_emit_interval:
            return

        epoch_fraction = min(1.0, float(batch + 1) / float(self._steps_in_epoch))
        completed_units = self.completed_epochs_before + self._current_epoch + epoch_fraction
        progress_pct = (completed_units / self.total_epochs) * 100.0
        eta_seconds = self._estimate_eta(completed_units)
        self._emit(progress_pct, eta_seconds, int(completed_units))
        self._last_emit_time = now

    def on_epoch_end(self, epoch, logs=None):
        epoch_done = self.completed_epochs_before + epoch + 1
        progress_pct = (epoch_done / self.total_epochs) * 100.0
        eta_seconds = self._estimate_eta(float(epoch_done))
        self._emit(progress_pct, eta_seconds, epoch_done)
        self._last_emit_time = time.time()
