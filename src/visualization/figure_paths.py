from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLOTS_ROOT = ROOT / "plots"
OTHERS_PLOTS_DIR = PLOTS_ROOT / "others"
EFFICIENTNET_PLOTS_DIR = PLOTS_ROOT / "EfficientNetV2"
EFFICIENTNET_B0_PLOTS_DIR = PLOTS_ROOT / "EfficientNetV2-B0"
EFFICIENTNET_S_PLOTS_DIR = PLOTS_ROOT / "EfficientNetV2-S"
DINO_PLOTS_DIR = PLOTS_ROOT / "DINOv3"
EVIDENCE_PLOTS_DIR = PLOTS_ROOT / "evidence"
ARCHIVED_PLOTS_DIR = PLOTS_ROOT / "archived"

_SHARED_ALLOWLIST_PREFIXES = (
    "architecture",
    "class_distribution",
    "system",
    "pipeline",
    "preprocessing",
)

_EFFICIENTNET_KEYWORDS = (
    "efficientnet",
    "effnet",
)

_DINO_KEYWORDS = (
    "dino",
    "vit",
)


def prepare_plot_directories() -> None:
    """Create the plot directory tree if it was deleted."""
    for directory in [
        PLOTS_ROOT,
        OTHERS_PLOTS_DIR,
        EFFICIENTNET_PLOTS_DIR,
        EFFICIENTNET_B0_PLOTS_DIR,
        EFFICIENTNET_S_PLOTS_DIR,
        DINO_PLOTS_DIR,
        EVIDENCE_PLOTS_DIR,
        ARCHIVED_PLOTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def reset_plot_directories() -> None:
    """Delete and recreate the full plot tree."""
    if PLOTS_ROOT.exists():
        shutil.rmtree(PLOTS_ROOT)
    prepare_plot_directories()


def normalize_backbone_name(backbone_name: str | None) -> str:
    name = (backbone_name or "").strip().lower()
    if "dino" in name or "vit" in name:
        return "DINOv3"
    if "efficientnet" in name or "effnet" in name:
        # Check for specific variant (B0 or S)
        if "b0" in name:
            return "EfficientNetV2-B0"
        if "efficientnetv2s" in name or "-s" in name:
            return "EfficientNetV2-S"
        # Default fallback for unspecified EfficientNet
        return "EfficientNetV2"
    return "others"


def backbone_plot_dir(backbone_name: str | None) -> Path:
    normalized = normalize_backbone_name(backbone_name)
    if normalized == "DINOv3":
        return DINO_PLOTS_DIR
    if normalized == "EfficientNetV2-B0":
        return EFFICIENTNET_B0_PLOTS_DIR
    if normalized == "EfficientNetV2-S":
        return EFFICIENTNET_S_PLOTS_DIR
    if normalized == "EfficientNetV2":
        return EFFICIENTNET_PLOTS_DIR
    return OTHERS_PLOTS_DIR


def relocate_misplaced_shared_plots() -> list[tuple[Path, Path]]:
    """Move model-specific files out of others into their backbone folders."""
    moved: list[tuple[Path, Path]] = []
    if not OTHERS_PLOTS_DIR.exists():
        return moved

    for file_path in OTHERS_PLOTS_DIR.glob("*"):
        if not file_path.is_file():
            continue

        stem = file_path.stem.lower()
        if stem.startswith(_SHARED_ALLOWLIST_PREFIXES):
            continue

        destination_dir: Path | None = None
        if any(keyword in stem for keyword in _EFFICIENTNET_KEYWORDS):
            destination_dir = EFFICIENTNET_PLOTS_DIR
        elif any(keyword in stem for keyword in _DINO_KEYWORDS):
            destination_dir = DINO_PLOTS_DIR

        if destination_dir is None:
            continue

        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / file_path.name
        file_path.replace(destination)
        moved.append((file_path, destination))

    return moved
