"""
Master Figure Generation Orchestrator.

Runs all figure generation scripts in the correct order with error handling
and comprehensive logging.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.figure_paths import (
    DINO_PLOTS_DIR,
    EFFICIENTNET_B0_PLOTS_DIR,
    EFFICIENTNET_S_PLOTS_DIR,
    OTHERS_PLOTS_DIR,
    prepare_plot_directories,
    relocate_misplaced_shared_plots,
    reset_plot_directories,
)


def run_script(script_name, description, timeout_seconds=5400, args=None):
    """Run a generation script with live output streaming and error handling."""
    script_path = os.path.join(ROOT, "scripts", script_name)

    if not os.path.exists(script_path):
        print(f"\n⚠ SKIPPED: {description}")
        print(f"   Script not found: {script_path}")
        return False

    print(f"\n{'=' * 70}")
    print(f"▶ RUNNING: {description}")
    print(f"⏱ Timeout: {timeout_seconds // 60} minutes")
    print(f"{'=' * 70}")

    started = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, script_path]
    if args:
        command.extend(args)

    try:
        # Stream child output directly so long-running work doesn't look stuck.
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            timeout=timeout_seconds,
            check=False,
        )

        elapsed = time.time() - started
        if result.returncode == 0:
            print(f"✓ SUCCESS: {description} ({elapsed / 60:.1f} min)")
            return True

        print(
            f"✗ FAILED: {description} (exit code {result.returncode}, {elapsed / 60:.1f} min)"
        )
        return False

    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        print(
            f"✗ TIMEOUT: {description} (exceeded {timeout_seconds / 60:.1f} min, ran {elapsed / 60:.1f} min)"
        )
        return False
    except Exception as e:
        elapsed = time.time() - started
        print(f"✗ ERROR: {description} ({elapsed / 60:.1f} min)")
        print(f"  {type(e).__name__}: {e}")
        return False


def main():
    """Main orchestration."""
    print(f"\n{'#' * 70}")
    print("#  LEAF DISEASE DETECTION - COMPREHENSIVE FIGURE GENERATION")
    print(f"{'#' * 70}")

    print("\nResetting plots directory tree...")
    reset_plot_directories()
    prepare_plot_directories()

    scripts_to_run = [
        (
            "generate_figures.py",
            "Core Validation & Results Figures (DINOv3)",
            5400,
            [
                "--model-path",
                str(ROOT / "models" / "leaf_disease_refined.keras"),
                "--output-dir",
                str(DINO_PLOTS_DIR),
            ],
        ),
        (
            "generate_robustness_figures.py",
            "Robustness & Perturbation Analysis (DINOv3)",
            1800,
            [
                "--model-path",
                str(ROOT / "models" / "leaf_disease_refined.keras"),
                "--output-dir",
                str(DINO_PLOTS_DIR),
            ],
        ),
        (
            "generate_statistical_figures.py",
            "Statistical Validation & Uncertainty (DINOv3)",
            1800,
            [
                "--model-path",
                str(ROOT / "models" / "leaf_disease_refined.keras"),
                "--output-dir",
                str(DINO_PLOTS_DIR),
            ],
        ),
        (
            "generate_ablation_figures.py",
            "Ablation Study Comparisons (DINOv3)",
            900,
            ["--output-dir", str(DINO_PLOTS_DIR)],
        ),
    ]

    effnet_b0_dir = ROOT / "models" / "EfficientNetv2B0"
    effnet_b0_candidates = [
        effnet_b0_dir / "leaf_disease_EfficientNetV2-B0.keras",
        effnet_b0_dir / "leaf_disease_classifier.keras",
        effnet_b0_dir / "leaf_disease_checkpoint.keras",
    ]
    effnet_b0_model = next(
        (path for path in effnet_b0_candidates if path.exists()), None
    )

    if effnet_b0_dir.exists() and effnet_b0_model is not None:
        scripts_to_run.extend(
            [
                (
                    "generate_figures.py",
                    "Core Validation & Results Figures (EfficientNetV2-B0)",
                    5400,
                    [
                        "--model-path",
                        str(effnet_b0_model),
                        "--output-dir",
                        str(EFFICIENTNET_B0_PLOTS_DIR),
                    ],
                ),
                (
                    "generate_robustness_figures.py",
                    "Robustness & Perturbation Analysis (EfficientNetV2-B0)",
                    1800,
                    [
                        "--model-path",
                        str(effnet_b0_model),
                        "--output-dir",
                        str(EFFICIENTNET_B0_PLOTS_DIR),
                    ],
                ),
                (
                    "generate_statistical_figures.py",
                    "Statistical Validation & Uncertainty (EfficientNetV2-B0)",
                    1800,
                    [
                        "--model-path",
                        str(effnet_b0_model),
                        "--output-dir",
                        str(EFFICIENTNET_B0_PLOTS_DIR),
                    ],
                ),
                (
                    "generate_ablation_figures.py",
                    "Ablation Study Comparisons (EfficientNetV2-B0)",
                    900,
                    ["--output-dir", str(EFFICIENTNET_B0_PLOTS_DIR)],
                ),
            ]
        )
    else:
        print("\n⚠ SKIPPING EfficientNetV2-B0 core figures")
        print(f"   Required folder/model not found under: {effnet_b0_dir}")

    # Handle EfficientNetV2-S model
    effnet_s_dir = ROOT / "models" / "EfficientNetv2S"
    effnet_s_candidates = [
        effnet_s_dir / "leaf_disease_EfficientNetV2-S.keras",
        effnet_s_dir / "leaf_disease_classifier.keras",
        effnet_s_dir / "leaf_disease_checkpoint.keras",
    ]
    effnet_s_model = next(
        (path for path in effnet_s_candidates if path.exists()), None
    )

    if effnet_s_dir.exists() and effnet_s_model is not None:
        scripts_to_run.extend(
            [
                (
                    "generate_figures.py",
                    "Core Validation & Results Figures (EfficientNetV2-S)",
                    5400,
                    [
                        "--model-path",
                        str(effnet_s_model),
                        "--output-dir",
                        str(EFFICIENTNET_S_PLOTS_DIR),
                    ],
                ),
                (
                    "generate_robustness_figures.py",
                    "Robustness & Perturbation Analysis (EfficientNetV2-S)",
                    1800,
                    [
                        "--model-path",
                        str(effnet_s_model),
                        "--output-dir",
                        str(EFFICIENTNET_S_PLOTS_DIR),
                    ],
                ),
                (
                    "generate_statistical_figures.py",
                    "Statistical Validation & Uncertainty (EfficientNetV2-S)",
                    1800,
                    [
                        "--model-path",
                        str(effnet_s_model),
                        "--output-dir",
                        str(EFFICIENTNET_S_PLOTS_DIR),
                    ],
                ),
                (
                    "generate_ablation_figures.py",
                    "Ablation Study Comparisons (EfficientNetV2-S)",
                    900,
                    ["--output-dir", str(EFFICIENTNET_S_PLOTS_DIR)],
                ),
            ]
        )
    else:
        print("\n⚠ SKIPPING EfficientNetV2-S core figures")
        print(f"   Required folder/model not found under: {effnet_s_dir}")

    scripts_to_run.append(
        (
            "generate_preprocessing_figures.py",
            "Preprocessing & Data Quality (Shared)",
            1800,
            ["--output-dir", str(OTHERS_PLOTS_DIR)],
        )
    )

    results = {}
    start_time = time.time()

    for script_name, description, timeout_seconds, args in scripts_to_run:
        results[description] = run_script(
            script_name,
            description,
            timeout_seconds=timeout_seconds,
            args=args,
        )

    moved_files = relocate_misplaced_shared_plots()
    if moved_files:
        print(
            f"\nMoved {len(moved_files)} misplaced model-specific shared plot(s):"
        )
        for source, destination in moved_files:
            print(
                f"  - {source.name}: {source.parent.name} -> {destination.parent.name}"
            )

    elapsed_time = time.time() - start_time

    # Summary report
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    successful = sum(1 for v in results.values() if v)
    total = len(results)

    for description, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {description}")

    print(f"\n{'=' * 70}")
    print(f"Results: {successful}/{total} suites completed successfully")
    print(f"Total time: {elapsed_time / 60:.1f} minutes")
    print(f"{'=' * 70}")

    if successful == total:
        print("\n✓✓✓ ALL FIGURE GENERATION COMPLETE ✓✓✓")
        return 0
    else:
        print(
            f"\n⚠ {total - successful} suite(s) failed. Review errors above."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
