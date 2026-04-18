import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

TASK_CHOICES = [
    "serve",
    "train",
    "fine_tune",
    "refine",
    "evaluate",
    "visualize",
    "resume",
    "validate",
]

# Keep command-runner lightweight and independent from TensorFlow imports.
BACKBONE_CHOICES = [
    "EfficientNetV2B0",
    "EfficientNetV2B1",
    "EfficientNetV2B2",
    "EfficientNetV2B3",
    "EfficientNetV2M",
    "EfficientNetV2L",
    "DINOv3",
]


def _run_command(command, cwd, env=None):

    try:
        completed = subprocess.run(command, cwd=str(cwd), check=False, env=env)
    except KeyboardInterrupt:
        return 130
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Leaf Disease Detection — Command Runner"
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="serve",
        choices=TASK_CHOICES,
        help="Task to execute (default: serve)",
    )
    parser.add_argument(
        "--archive-logs",
        action="store_true",
        help="Keep timestamped archive logs and per-run manifests.",
    )
    parser.add_argument(
        "--latest-logs-only",
        action="store_true",
        help="Keep only latest logs (default behaviour).",
    )
    parser.add_argument(
        "--base-model",
        choices=BACKBONE_CHOICES,
        default=None,
        help="Backbone to use for train/resume tasks (passed through to train_model.py).",
    )
    args = parser.parse_args()

    if args.archive_logs and args.latest_logs_only:
        parser.error("Use either --archive-logs or --latest-logs-only, not both.")

    if args.base_model and args.task not in {"train", "resume"}:
        parser.error("--base-model can only be used with train or resume.")

    command_map = {
        "serve": PROJECT_ROOT / "app.py",
        "train": PROJECT_ROOT / "train_model.py",
        "fine_tune": PROJECT_ROOT / "fine_tune_model.py",
        "refine": PROJECT_ROOT / "refine_model.py",
        "evaluate": PROJECT_ROOT / "evaluate_model.py",
        "visualize": PROJECT_ROOT / "scripts" / "generate_figures.py",
        "resume": PROJECT_ROOT / "fine_tune_model.py",
        "validate": PROJECT_ROOT / "evaluate_model.py",
    }
    script_path = command_map[args.task]
    if not script_path.exists():
        parser.error(f"Missing script for task '{args.task}': {script_path}")

    command = [sys.executable, str(script_path)]
    child_env = dict(os.environ)
    child_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if args.archive_logs:
        child_env["LEAF_SAVE_LOG_ARCHIVE"] = "1"
        child_env["LEAF_SAVE_RUN_MANIFESTS"] = "1"
    elif args.latest_logs_only:
        child_env["LEAF_SAVE_LOG_ARCHIVE"] = "0"
        child_env["LEAF_SAVE_RUN_MANIFESTS"] = "0"

    if args.base_model:
        child_env["LEAF_BASE_MODEL"] = args.base_model

    raise SystemExit(_run_command(command, PROJECT_ROOT, env=child_env))


if __name__ == "__main__":
    main()
