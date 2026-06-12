import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import config


def _reload_config():
    return importlib.reload(config)


def test_logging_flags_default_latest_only(monkeypatch):
    monkeypatch.delenv("LEAF_SAVE_LOG_ARCHIVE", raising=False)
    monkeypatch.delenv("LEAF_SAVE_RUN_MANIFESTS", raising=False)

    cfg = _reload_config()
    assert cfg.SAVE_LOG_ARCHIVE is False
    assert cfg.SAVE_RUN_MANIFESTS is False


def test_logging_flags_enable_archive_enables_manifests_by_default(
    monkeypatch,
):
    monkeypatch.setenv("LEAF_SAVE_LOG_ARCHIVE", "1")
    monkeypatch.delenv("LEAF_SAVE_RUN_MANIFESTS", raising=False)

    cfg = _reload_config()
    assert cfg.SAVE_LOG_ARCHIVE is True
    assert cfg.SAVE_RUN_MANIFESTS is True


def test_logging_flags_manifest_override(monkeypatch):
    monkeypatch.setenv("LEAF_SAVE_LOG_ARCHIVE", "1")
    monkeypatch.setenv("LEAF_SAVE_RUN_MANIFESTS", "0")

    cfg = _reload_config()
    assert cfg.SAVE_LOG_ARCHIVE is True
    assert cfg.SAVE_RUN_MANIFESTS is False
