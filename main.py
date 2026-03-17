"""Project entrypoint for common operational tasks."""

import argparse
import subprocess
import sys


def _run_command(command):
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description="Leaf disease detection command runner")
    parser.add_argument(
        "task",
        choices=[
            "serve",
            "train",
            "fine_tune",
            "evaluate",
            "visualize",
            "resume",
            "validate",
        ],
        help="Task to execute",
    )
    args = parser.parse_args()

    command_map = {
        "serve": [sys.executable, "app.py"],
        "train": [sys.executable, "train_model.py"],
        "fine_tune": [sys.executable, "fine_tune_model.py"],
        "evaluate": [sys.executable, "evaluate_model.py"],
        "visualize": [sys.executable, "generate_figures.py"],
        # Backward-compatible aliases.
        "resume": [sys.executable, "fine_tune_model.py"],
        "validate": [sys.executable, "evaluate_model.py"],
    }
    raise SystemExit(_run_command(command_map[args.task]))


if __name__ == "__main__":
    main()
