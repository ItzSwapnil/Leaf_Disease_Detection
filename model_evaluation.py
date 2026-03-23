import gc
import json
import os
from datetime import datetime

import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import FINAL_MODEL_PATH, IMG_SIZE, SAVE_LOG_ARCHIVE, VAL_DIR
from hardware import configure_tensorflow
from model_paths import resolve_keras_model_path
from preprocessing import preprocess_batch_for_model
from training_utils import WarmupCosineSchedule

def _save_evaluation_report(report: dict):
    
    reports_dir = os.path.join("docs", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    latest_json = os.path.join(reports_dir, "evaluation_report.json")
    archive_json = os.path.join(reports_dir, f"evaluation_report_{stamp}.json")
    latest_md = os.path.join(reports_dir, "evaluation_report.md")
    archive_md = os.path.join(reports_dir, f"evaluation_report_{stamp}.md")

    target_json = [latest_json]
    if SAVE_LOG_ARCHIVE:
        target_json.append(archive_json)

    for path in target_json:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    md_lines = [
        "# Evaluation Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Model: {report['model_path']}",
        f"- Validation samples: {report['validation_samples']}",
        f"- Classes: {report['num_classes']}",
        "",
        "## Aggregate Metrics",
        "",
        f"- Loss: {report['metrics']['loss']:.4f}",
        f"- Accuracy: {report['metrics']['accuracy'] * 100:.2f}%",
        f"- Macro Precision: {report['metrics']['macro_precision']:.4f}",
        f"- Macro Recall: {report['metrics']['macro_recall']:.4f}",
        f"- Macro F1: {report['metrics']['macro_f1']:.4f}",
        "",
        "## Top Confused Class Pairs",
        "",
    ]
    if report.get("top_confused_pairs"):
        for pair in report["top_confused_pairs"][:10]:
            md_lines.append(
                f"- {pair['true_class']} -> {pair['pred_class']}: {pair['count']}"
            )
    else:
        md_lines.append("- No notable confusion pairs found.")
    md_lines.append("")
    md_content = "\n".join(md_lines)

    target_md = [latest_md]
    if SAVE_LOG_ARCHIVE:
        target_md.append(archive_md)

    for path in target_md:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_content)

    print(f"Saved evaluation report: {latest_json}")
    print(f"Saved evaluation report: {latest_md}")
    if SAVE_LOG_ARCHIVE:
        print(f"Saved archive report: {archive_json}")
        print(f"Saved archive report: {archive_md}")

def main():
    
    configure_tensorflow()

    model_path = resolve_keras_model_path([FINAL_MODEL_PATH])
    print(f"Loading model: {model_path}")
    model = load_model(
        model_path,
        custom_objects={"WarmupCosineSchedule": WarmupCosineSchedule}
    )

    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        shuffle=False,
    )
    val_gen = val_ds.map(
        lambda x, y: (preprocess_batch_for_model(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    print("\n--- Evaluating saved model ---")
    metrics = model.evaluate(val_gen, verbose=1, return_dict=True)
    loss = float(metrics.get("loss", 0.0))
    acc = float(metrics.get("accuracy", metrics.get("acc", 0.0)))

    predictions = model.predict(val_gen, verbose=1)
    y_pred = predictions.argmax(axis=1)
    y_true_cat = np.concatenate([labels.numpy() for _, labels in val_ds], axis=0)
    y_true = y_true_cat.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)
    class_names = list(val_ds.class_names)
    class_report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Identify most confused class pairs from off-diagonal confusion matrix
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)
    flat_idx = np.argsort(cm_offdiag, axis=None)[::-1]
    top_confused_pairs = []
    for idx in flat_idx:
        count = int(cm_offdiag.flat[idx])
        if count <= 0:
            break
        true_idx, pred_idx = np.unravel_index(idx, cm_offdiag.shape)
        top_confused_pairs.append({
            "true_class": class_names[int(true_idx)],
            "pred_class": class_names[int(pred_idx)],
            "count": count,
        })
        if len(top_confused_pairs) >= 15:
            break

    print(f"\nValidation loss: {loss:.4f}")
    print(f"Validation accuracy: {acc * 100:.2f}%")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1-score: {f1:.4f}")
    if top_confused_pairs:
        print("Top confused class pairs:")
        for pair in top_confused_pairs[:5]:
            print(f"  {pair['true_class']} -> {pair['pred_class']}: {pair['count']}")

    per_class_metrics = {}
    for class_name in class_names:
        row = class_report.get(class_name, {})
        per_class_metrics[class_name] = {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1-score", 0.0)),
            "support": int(row.get("support", 0)),
        }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": model_path,
        "validation_samples": int(len(y_true)),
        "num_classes": int(len(val_ds.class_names)),
        "metrics": {
            "loss": loss,
            "accuracy": acc,
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        },
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "top_confused_pairs": top_confused_pairs,
    }
    _save_evaluation_report(report)

    gc.collect()

if __name__ == "__main__":
    main()
