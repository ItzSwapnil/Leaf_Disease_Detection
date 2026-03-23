import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

import model_paths
from training_progress import ProgressEmitter


def test_resolve_keras_model_path_prefers_existing_preferred(tmp_path, monkeypatch):
    preferred = tmp_path / "preferred.keras"
    preferred.write_text("x", encoding="utf-8")

    monkeypatch.setattr(model_paths, "FINAL_MODEL_PATH", str(tmp_path / "final.keras"))
    monkeypatch.setattr(model_paths, "CHECKPOINT_PATH", str(tmp_path / "checkpoint.keras"))
    monkeypatch.setattr(model_paths, "MODELS_DIR", str(tmp_path / "models"))

    resolved = model_paths.resolve_keras_model_path([str(preferred)])
    assert resolved == str(preferred)


def test_resolve_keras_model_path_falls_back_to_discovered_sorted(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "z_last.keras").write_text("z", encoding="utf-8")
    (models_dir / "a_first.keras").write_text("a", encoding="utf-8")

    monkeypatch.setattr(model_paths, "FINAL_MODEL_PATH", str(tmp_path / "missing_final.keras"))
    monkeypatch.setattr(model_paths, "CHECKPOINT_PATH", str(tmp_path / "missing_checkpoint.keras"))
    monkeypatch.setattr(model_paths, "MODELS_DIR", str(models_dir))

    resolved = model_paths.resolve_keras_model_path([])
    assert Path(resolved).name == "a_first.keras"


def test_resolve_keras_model_path_raises_when_nothing_exists(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    monkeypatch.setattr(model_paths, "FINAL_MODEL_PATH", str(tmp_path / "missing_final.keras"))
    monkeypatch.setattr(model_paths, "CHECKPOINT_PATH", str(tmp_path / "missing_checkpoint.keras"))
    monkeypatch.setattr(model_paths, "MODELS_DIR", str(models_dir))

    with pytest.raises(FileNotFoundError):
        model_paths.resolve_keras_model_path([])


def test_progress_emitter_estimate_eta_math():
    emitter = ProgressEmitter(
        stage="phase1",
        total_epochs=10,
        completed_epochs_before=0,
        run_start_time=100.0,
    )

    # If 5 units are completed in 10 seconds, 5 units remain -> ETA should be 10 seconds.
    emitter.run_start_time = 100.0
    import time
    original = time.time
    try:
        time.time = lambda: 110.0
        assert emitter._estimate_eta(5.0) == pytest.approx(10.0)
    finally:
        time.time = original


def test_progress_emitter_emits_progress_lines(capsys):
    emitter = ProgressEmitter(
        stage="phase2",
        total_epochs=4,
        completed_epochs_before=1,
        min_emit_interval=0.0,
    )
    emitter.params = {"steps": 10}

    emitter.on_train_begin()
    emitter.on_epoch_begin(0)
    emitter.on_train_batch_end(4)
    emitter.on_epoch_end(0)

    out = capsys.readouterr().out
    assert "TRAINING_PROGRESS" in out
    assert '"stage": "phase2"' in out