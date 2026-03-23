import csv
import json
import os
import time

import keras

class ProgressEmitter(keras.callbacks.Callback):
    

    def __init__(
        self,
        stage: str,
        total_epochs: int,
        completed_epochs_before: int = 0,
        run_start_time: float = None,
        min_emit_interval: float = 4.0,
    ):
        super().__init__()
        self.stage = stage
        self.total_epochs = max(1, int(total_epochs))
        self.completed_epochs_before = max(0, int(completed_epochs_before))
        self.run_start_time = run_start_time or time.time()
        self.min_emit_interval = float(min_emit_interval)
        self._current_epoch = 0
        self._steps_in_epoch = None
        self._last_emit_time = 0.0

    def _emit(self, progress_pct: float, eta_seconds: float, epoch_done: int):
        payload = {
            "stage": self.stage,
            "progress_pct": round(float(progress_pct), 2),
            "eta_seconds": max(0.0, float(eta_seconds)),
            "eta_scope": "whole_process",
            "epoch_done": int(epoch_done),
            "total_epochs": int(self.total_epochs),
            "timestamp": round(time.time(), 2),
        }
        print(f"TRAINING_PROGRESS {json.dumps(payload)}", flush=True)

    def _estimate_eta(self, completed_units: float) -> float:
        if completed_units <= 0.0:
            return 0.0
        elapsed = max(0.0, time.time() - self.run_start_time)
        avg_per_unit = elapsed / float(completed_units)
        remaining = max(0.0, float(self.total_epochs) - completed_units)
        return avg_per_unit * remaining

    def on_train_begin(self, logs=None):
        self._steps_in_epoch = self.params.get("steps")
        self._last_emit_time = 0.0
        initial_pct = (self.completed_epochs_before / self.total_epochs) * 100.0
        self._emit(initial_pct, 0.0, self.completed_epochs_before)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = int(epoch)

    def on_train_batch_end(self, batch, logs=None):
        if not self._steps_in_epoch:
            return
        now = time.time()
        if now - self._last_emit_time < self.min_emit_interval:
            return

        epoch_fraction = min(1.0, float(batch + 1) / float(self._steps_in_epoch))
        completed = self.completed_epochs_before + self._current_epoch + epoch_fraction
        progress_pct = (completed / self.total_epochs) * 100.0
        eta = self._estimate_eta(completed)
        self._emit(progress_pct, eta, int(completed))
        self._last_emit_time = now

    def on_epoch_end(self, epoch, logs=None):
        epoch_done = self.completed_epochs_before + epoch + 1
        progress_pct = (epoch_done / self.total_epochs) * 100.0
        eta = self._estimate_eta(float(epoch_done))
        self._emit(progress_pct, eta, epoch_done)
        self._last_emit_time = time.time()

class IntervalMetricsLogger(keras.callbacks.Callback):
    

    def __init__(
        self,
        file_path: str,
        points_per_epoch: int = 12,
        stage: str = "train",
        append: bool = False,
        run_id: str = None,
    ):
        super().__init__()
        self.file_path = file_path
        self.points_per_epoch = max(1, int(points_per_epoch))
        self.stage = stage
        self.append = bool(append)
        self.run_id = run_id or ""
        self._steps = None
        self._interval = 1
        self._global_step = 0
        self._epoch = 0
        self._writer = None
        self._fp = None

    def on_train_begin(self, logs=None):
        self._steps = int(self.params.get("steps") or 0)
        if self._steps > 0:
            self._interval = max(1, self._steps // self.points_per_epoch)
        else:
            self._interval = 1

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        mode = "a" if self.append else "w"
        file_exists = os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0
        self._fp = open(self.file_path, mode, newline="", encoding="utf-8")
        self._writer = csv.writer(self._fp)
        if (not self.append) or (not file_exists):
            self._writer.writerow([
                "run_id", "stage", "row_type", "global_step",
                "epoch", "epoch_progress", "loss", "accuracy",
                "val_loss", "val_accuracy", "learning_rate", "timestamp",
            ])
        self._fp.flush()

    def _safe_float(self, value):
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _current_lr(self):
        try:
            lr = self.model.optimizer.learning_rate
            if callable(lr):
                lr = lr(self.model.optimizer.iterations)
            return self._safe_float(keras.ops.convert_to_numpy(lr))
        except Exception:
            return None

    def _write_row(self, row_type: str, epoch_progress: float, logs):
        if self._writer is None:
            return
        logs = logs or {}
        self._writer.writerow([
            self.run_id, self.stage, row_type, self._global_step,
            self._epoch + 1, round(float(epoch_progress), 6),
            self._safe_float(logs.get("loss")),
            self._safe_float(logs.get("accuracy")),
            self._safe_float(logs.get("val_loss")),
            self._safe_float(logs.get("val_accuracy")),
            self._current_lr(),
            round(time.time(), 3),
        ])
        self._fp.flush()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = int(epoch)

    def on_train_batch_end(self, batch, logs=None):
        self._global_step += 1
        if self._steps and ((batch + 1) % self._interval != 0) and ((batch + 1) != self._steps):
            return
        epoch_progress = min(1.0, float(batch + 1) / float(self._steps)) if self._steps else 0.0
        self._write_row("batch", epoch_progress, logs)

    def on_epoch_end(self, epoch, logs=None):
        self._write_row("epoch_end", 1.0, logs)

    def on_train_end(self, logs=None):
        if self._fp is not None:
            self._fp.flush()
            self._fp.close()
            self._fp = None
