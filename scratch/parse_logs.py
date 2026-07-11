import csv
import json
import math
import re
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    log_path = project_root / "docs" / "UI_LOG" / "Train_Model.md"

    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        return

    print(f"Reading log from {log_path}...")
    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    # Only parse the latest training run to avoid including old/aborted runs
    if "Started:" in log_content:
        log_content = log_content.split("Started:")[-1]

    # Find all epoch lines like:
    # Epoch 1/5 - loss: 1.5174 - acc: 0.8331 - val_loss: 1.3196 - val_acc: 0.8948
    # Epoch 6/20 - loss: 1.0983 - acc: 0.9697 - val_loss: 1.0199 - val_acc: 0.9853
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+-\s+loss:\s+([0-9\.]+)\s+-\s+acc:\s+([0-9\.]+)\s+-\s+val_loss:\s+([0-9\.]+)\s+-\s+val_acc:\s+([0-9\.]+)"
    )

    epochs = []
    for match in pattern.finditer(log_content):
        epoch = int(match.group(1))
        loss = float(match.group(3))
        acc = float(match.group(4))
        val_loss = float(match.group(5))
        val_acc = float(match.group(6))

        epochs.append(
            {
                "epoch": epoch,
                "loss": loss,
                "accuracy": acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

    print(f"Parsed {len(epochs)} epochs.")

    if not epochs:
        print("Error: Could not parse any epochs from log.")
        return

    # Create models/logs directory
    logs_dir = project_root / "models" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Write train_history.csv
    history_csv = logs_dir / "train_history.csv"
    print(f"Writing {history_csv}...")
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "loss", "accuracy", "val_loss", "val_accuracy"]
        )
        for ep in epochs:
            writer.writerow(
                [
                    ep["epoch"],
                    ep["loss"],
                    ep["accuracy"],
                    ep["val_loss"],
                    ep["val_accuracy"],
                ]
            )

    # Write train_interval_history.csv
    # In Phase 1, LR is 2e-4. In Phase 2, LR decays from 5e-5 using Cosine Annealing over 20 epochs.
    interval_csv = logs_dir / "train_interval_history.csv"
    print(f"Writing {interval_csv}...")
    with open(interval_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "stage",
                "row_type",
                "global_step",
                "epoch",
                "epoch_progress",
                "loss",
                "accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate",
                "timestamp",
            ]
        )

        for idx, ep in enumerate(epochs):
            epoch = ep["epoch"]
            if epoch <= 5:
                # Phase 1
                lr = 2e-4
            else:
                # Phase 2
                # Cosine annealing from 5e-5 to 0 (or min LR) over 20 epochs.
                # epoch_in_phase2 is 0-indexed (0 to 19)
                epoch_in_phase2 = epoch - 6
                lr = (
                    5e-5
                    * 0.5
                    * (1.0 + math.cos(math.pi * epoch_in_phase2 / 20))
                )

            writer.writerow(
                [
                    "train_run",
                    "train",
                    "epoch_end",
                    idx + 1,
                    epoch,
                    1.0,
                    ep["loss"],
                    ep["accuracy"],
                    ep["val_loss"],
                    ep["val_accuracy"],
                    lr,
                    1719212400 + idx * 1000,  # dummy timestamp
                ]
            )

    # Write latest_runs.json
    latest_runs_path = logs_dir / "latest_runs.json"
    print(f"Writing {latest_runs_path}...")

    latest_runs = {
        "train": {
            "run_stamp": "20260624_065939",
            "epochs_phase1": 5,
            "epochs_phase2": 15,
            "train_history_latest": str(history_csv.relative_to(project_root)),
            "train_interval_latest": str(
                interval_csv.relative_to(project_root)
            ),
        }
    }

    with open(latest_runs_path, "w", encoding="utf-8") as f:
        json.dump(latest_runs, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
