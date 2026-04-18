#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_SEEDS = [42, 123, 999, 2024, 7]


def _run_step(script_path: Path, seed: int, root: Path) -> None:
    env = os.environ.copy()
    env["RUN_SEED"] = str(seed)
    command = [sys.executable, str(script_path)]
    print(f"Running {' '.join(command)} with RUN_SEED={seed}")
    subprocess.run(command, cwd=str(root), env=env, check=True)


def _collect_metric(report: dict, key: str) -> float:
    return float((report.get("metrics") or {}).get(key, 0.0))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run train/fine-tune/evaluate across multiple seeds and aggregate metrics."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="List of integer seeds (default: 42 123 999 2024 7).",
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip train_model.py")
    parser.add_argument(
        "--skip-fine-tune",
        action="store_true",
        help="Skip fine_tune_model.py",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip evaluate_model.py (not recommended).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to store per-seed and aggregate JSON files.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_dir = (root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    train_script = root / "train_model.py"
    fine_tune_script = root / "fine_tune_model.py"
    evaluate_script = root / "evaluate_model.py"
    eval_report_path = root / "reports" / "evaluation_report.json"

    per_seed_metrics: list[dict] = []

    for seed in args.seeds:
        if not args.skip_train:
            _run_step(train_script, seed, root)
        if not args.skip_fine_tune:
            _run_step(fine_tune_script, seed, root)
        if not args.skip_evaluate:
            _run_step(evaluate_script, seed, root)

        if not eval_report_path.exists():
            raise FileNotFoundError(
                f"Expected evaluation report not found: {eval_report_path}"
            )

        report = json.loads(eval_report_path.read_text(encoding="utf-8"))
        output = {
            "seed": int(seed),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model_path": report.get("model_path"),
            "validation_accuracy": _collect_metric(report, "validation_accuracy"),
            "macro_precision": _collect_metric(report, "macro_precision"),
            "macro_recall": _collect_metric(report, "macro_recall"),
            "macro_f1": _collect_metric(report, "macro_f1"),
            "test_accuracy": _collect_metric(report, "test_accuracy"),
            "temperature": float(
                (
                    (report.get("calibration") or {}).get("temperature_scaling") or {}
                ).get("temperature", 1.0)
            ),
        }
        per_seed_metrics.append(output)

        seed_path = results_dir / f"seed_{seed}.json"
        seed_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Saved {seed_path.relative_to(root)}")

    f1_values = [entry["macro_f1"] for entry in per_seed_metrics]
    accuracy_values = [entry["validation_accuracy"] for entry in per_seed_metrics]

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": [int(seed) for seed in args.seeds],
        "num_runs": int(len(per_seed_metrics)),
        "aggregate": {
            "validation_accuracy_mean": float(statistics.mean(accuracy_values)),
            "validation_accuracy_std": float(statistics.pstdev(accuracy_values)),
            "macro_f1_mean": float(statistics.mean(f1_values)),
            "macro_f1_std": float(statistics.pstdev(f1_values)),
        },
        "runs": per_seed_metrics,
    }

    summary_path = results_dir / "multi_seed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {summary_path.relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
