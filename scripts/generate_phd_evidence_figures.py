from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _clamp01(value: float) -> bool:
    return 0.0 <= float(value) <= 1.0


def _metric_close(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(float(a) - float(b)) <= tol


def _append_status(
    report: dict, name: str, status: str, reason: str | None = None
) -> None:
    entry = {"plot": name, "status": status}
    if reason:
        entry["reason"] = reason
    report.setdefault("plots", []).append(entry)


def validate_artifacts(eval_report: dict) -> list[str]:
    issues: list[str] = []
    metrics = eval_report.get("metrics", {})
    val_acc = metrics.get("validation_accuracy")
    test_acc = metrics.get("test_accuracy")
    macro_f1 = metrics.get("macro_f1")
    if val_acc is None or test_acc is None or macro_f1 is None:
        issues.append(
            "Missing one or more core metrics (validation_accuracy, test_accuracy, macro_f1)."
        )
    else:
        for name, value in [
            ("validation_accuracy", val_acc),
            ("test_accuracy", test_acc),
            ("macro_f1", macro_f1),
        ]:
            if not _clamp01(value):
                issues.append(f"Core metric out of range [0,1]: {name}={value}")

    robust = eval_report.get("robustness", {}) or {}
    robust_base = (robust.get("base") or {}).get("accuracy")
    if robust_base is not None and val_acc is not None:
        if val_acc - float(robust_base) > 0.3:
            issues.append(
                "robustness base accuracy is far below validation accuracy; "
                "possible split/protocol mismatch."
            )

    return issues


def plot_calibration(eval_report: dict, out_dir: Path) -> None:
    unc = eval_report["calibration"]["uncalibrated"]
    tsc = eval_report["calibration"]["temperature_scaled"]
    temp = eval_report["calibration"]["temperature_scaling"]

    labels = ["ECE", "MCE", "NLL before", "NLL after"]
    values = [unc["ece"], unc["mce"], temp["nll_before"], temp["nll_after"]]

    x = range(len(labels))

    plt.figure(figsize=(8.0, 4.2))
    colors = ["#3b82f6", "#ef4444", "#6b7280", "#10b981"]
    plt.bar(list(x), values, color=colors)
    plt.xticks(list(x), labels)
    plt.ylabel("Metric value")
    plt.title("Calibration Summary from evaluation_report.json")
    _save_fig(out_dir / "calibration_metrics_reported.png")


def plot_bootstrap_ci(eval_report: dict, out_dir: Path) -> None:
    ci = eval_report["bootstrap_confidence_intervals"]
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]

    means = [ci[m]["mean"] for m in metrics]
    lowers = [ci[m]["mean"] - ci[m]["lower"] for m in metrics]
    uppers = [ci[m]["upper"] - ci[m]["mean"] for m in metrics]

    x = list(range(len(metrics)))

    plt.figure(figsize=(8.2, 4.4))
    plt.errorbar(x, means, yerr=[lowers, uppers], fmt="o", capsize=4)
    plt.xticks(x, labels)
    plt.ylabel("Metric value")
    plt.title("Bootstrap 95% Confidence Intervals (n_boot=2000)")
    _save_fig(out_dir / "bootstrap_ci_metrics.png")


def plot_selective_risk(eval_report: dict, out_dir: Path) -> None:
    conf = eval_report["rejection"]["confidence"]
    ent = eval_report["rejection"]["entropy"]

    names = ["Confidence gate", "Entropy gate"]
    coverage = [conf["coverage"], ent["coverage"]]
    accepted_acc = [conf["accepted_accuracy"], ent["accepted_accuracy"]]

    plt.figure(figsize=(8.2, 4.5))
    plt.scatter(coverage, accepted_acc, s=120)
    for i, name in enumerate(names):
        plt.annotate(
            name,
            (coverage[i], accepted_acc[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    plt.xlabel("Coverage")
    plt.ylabel("Accepted-set accuracy")
    plt.ylim(min(accepted_acc) - 0.01, 1.0)
    plt.xlim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.title("Selective Risk-Coverage Points from evaluation_report.json")
    _save_fig(out_dir / "selective_risk_coverage_points.png")


def plot_robustness(eval_report: dict, out_dir: Path) -> None:
    rob = eval_report.get("robustness", {})
    base_acc = rob.get("base", {}).get("accuracy")
    if base_acc is None:
        return

    labels = ["base"]
    values = [base_acc]

    for row in rob.get("gaussian_blur", []):
        labels.append(f"blur_sigma_{row['sigma']}")
        values.append(row["accuracy"])
    for row in rob.get("brightness_shift", []):
        labels.append(f"bright_{row['factor']}")
        values.append(row["accuracy"])
    for row in rob.get("gaussian_noise", []):
        labels.append(f"noise_{row['sigma']}")
        values.append(row["accuracy"])

    plt.figure(figsize=(11.0, 4.8))
    plt.plot(range(len(values)), values, marker="o")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Robustness Stress Accuracy (as reported)")
    plt.grid(alpha=0.25)
    _save_fig(out_dir / "robustness_accuracy_reported.png")


def plot_run_timeline(latest_runs: dict, out_dir: Path) -> None:
    items = []
    for stage in ["train", "fine_tune", "refine"]:
        row = latest_runs.get(stage)
        if not row:
            continue
        created = row.get("created_at")
        if not created:
            continue
        items.append((stage, datetime.fromisoformat(created)))

    if not items:
        return

    items.sort(key=lambda x: x[1])
    x0 = items[0][1]
    xs = [(dt - x0).total_seconds() / 3600.0 for _, dt in items]
    ys = list(range(len(items)))
    labels = [f"{stage} ({dt.strftime('%Y-%m-%d %H:%M')})" for stage, dt in items]

    plt.figure(figsize=(9.0, 3.8))
    plt.scatter(xs, ys, s=120)
    plt.yticks(ys, labels)
    plt.xlabel("Hours since first recorded stage")
    plt.title("Recorded Run Lineage Timeline")
    plt.grid(alpha=0.25)
    _save_fig(out_dir / "run_lineage_timeline.png")


def write_provenance_report(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "evidence_plot_report.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis evidence plots from real run artifacts."
    )
    parser.add_argument(
        "--evaluation-report",
        default="reports/evaluation_report.json",
        help="Path to evaluation_report.json",
    )
    parser.add_argument(
        "--latest-runs",
        default="models/logs/latest_runs.json",
        help="Path to latest_runs.json",
    )
    parser.add_argument(
        "--output-dir",
        default="Thesis/plots/evidence",
        help="Output directory for generated evidence plots",
    )
    parser.add_argument(
        "--include-timeline",
        action="store_true",
        help="Include run timeline figure (off by default to keep thesis figures technically focused).",
    )
    parser.add_argument(
        "--allow-unsafe-robustness-plot",
        action="store_true",
        help="Force robustness plot generation even when robustness/validation mismatch is detected.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if major artifact consistency issues are detected.",
    )
    args = parser.parse_args()

    eval_report = _read_json(Path(args.evaluation_report))
    latest_runs = _read_json(Path(args.latest_runs))
    out_dir = Path(args.output_dir)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_report": str(Path(args.evaluation_report)),
        "latest_runs": str(Path(args.latest_runs)),
        "issues": [],
        "plots": [],
    }

    issues = validate_artifacts(eval_report)
    report["issues"] = issues
    if args.strict and issues:
        raise ValueError(
            "Artifact consistency checks failed in strict mode: " + " | ".join(issues)
        )

    plot_calibration(eval_report, out_dir)
    _append_status(report, "calibration_metrics_reported.png", "generated")

    plot_bootstrap_ci(eval_report, out_dir)
    _append_status(report, "bootstrap_ci_metrics.png", "generated")

    plot_selective_risk(eval_report, out_dir)
    _append_status(report, "selective_risk_coverage_points.png", "generated")

    robustness_issue = any("robustness base accuracy" in text for text in issues)
    if robustness_issue and not args.allow_unsafe_robustness_plot:
        _append_status(
            report,
            "robustness_accuracy_reported.png",
            "skipped",
            "robustness-validation mismatch detected; use --allow-unsafe-robustness-plot to force generation",
        )
    else:
        plot_robustness(eval_report, out_dir)
        _append_status(report, "robustness_accuracy_reported.png", "generated")

    if args.include_timeline:
        plot_run_timeline(latest_runs, out_dir)
        _append_status(report, "run_lineage_timeline.png", "generated")
    else:
        _append_status(
            report, "run_lineage_timeline.png", "skipped", "disabled by default"
        )

    write_provenance_report(out_dir, report)

    print(f"Saved evidence plots to: {out_dir}")
    if issues:
        print("Consistency warnings detected:")
        for msg in issues:
            print(f"- {msg}")
        print("See evidence_plot_report.json for full details.")


if __name__ == "__main__":
    main()
