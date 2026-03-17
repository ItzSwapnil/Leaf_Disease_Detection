# Experimental Run Log and Outcome Summary

This document records a complete successful workflow execution consisting of base training, extended fine-tuning, evaluation, and visualization generation.

## Workflow Timeline

```mermaid
flowchart LR
    A[Base Training] --> B[Fine-Tuning]
    B --> C[Evaluation]
    C --> D[Figure Generation]
```

## 1. Run Summary

| Stage | Script | Runtime | Exit Code | Status |
| ----- | ------ | ------- | --------- | ------ |
| Base Training | `train_model.py` | 53m 56s | 0 | Completed |
| Fine-Tuning | `fine_tune_model.py` | 148m 21s | 0 | Completed |
| Evaluation | `evaluate_model.py` | 2m 36s | 0 | Completed |
| Figure Generation | `generate_figures.py` | 2m 54s | 0 | Completed |

## 2. Base Training Observations

- Best checkpoint saved at epoch 13.
- Peak validation accuracy during this run segment: 96.69%.
- Final epochs showed stable high top-3 validation accuracy (>99.5%).

### 2.1 Terminal Excerpt

```text
Epoch 13: val_accuracy improved from 0.96188 to 0.96688, saving model to .../models/leaf_disease_checkpoint.keras
750/750 - 132s - accuracy: 0.9558 - loss: 0.9643 - top3_acc: 0.9952 - val_accuracy: 0.9669 - val_loss: 0.9491 - val_top3_acc: 0.9956
Epoch 14: val_accuracy did not improve from 0.96688
750/750 - 110s - accuracy: 0.9569 - loss: 0.9613 - top3_acc: 0.9948 - val_accuracy: 0.9631 - val_loss: 0.9542 - val_top3_acc: 0.9969
Epoch 15: val_accuracy did not improve from 0.96688
750/750 - 109s - accuracy: 0.9586 - loss: 0.9595 - top3_acc: 0.9955 - val_accuracy: 0.9550 - val_loss: 0.9645 - val_top3_acc: 0.9962
Restoring model weights from the end of the best epoch: 13.
```

## 3. Fine-Tuning Observations

- Fine-tuning executed with schedule-based AdamW optimization.
- Early stopping triggered at epoch 12 and restored best epoch 6.
- Final classifier artifact saved successfully.

### 3.1 Terminal Excerpt

```text
Trainable layers: 50
Found 220498 files belonging to 46 classes.
Found 19419 files belonging to 46 classes.
Using cosine LR schedule; ReduceLROnPlateau callback disabled.
TensorBoard package not found; continuing without TensorBoard callback.
Training configuration:
Batch size: 4
Optimizer: AdamW + CosineDecayRestarts
Epoch 12: val_accuracy did not improve from 0.97399
55125/55125 - 725s - accuracy: 0.9905 - loss: 0.7832 - top3_acc: 0.9987 - val_accuracy: 0.9586 - val_loss: 1.1385 - val_top3_acc: 0.9849
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 6.
Training complete
Best model saved to: .../models/leaf_disease_classifier.keras
```

## 4. Evaluation Metrics

Final evaluation metrics from the successful run:

1. Validation loss: 1.0340
2. Validation accuracy: 97.40%
3. Validation top-3 accuracy: 99.73%
4. Macro precision: 0.9686
5. Macro recall: 0.9668
6. Macro F1-score: 0.9657

### 4.1 Terminal Excerpt

```text
607/607  24s 29ms/step
Validation loss: 1.0340
Validation accuracy: 97.40%
Validation top3 accuracy: 99.73%
Macro precision: 0.9686
Macro recall: 0.9668
Macro F1 score: 0.9657
```

## 5. Visualization Outputs

The following artifacts were generated successfully under `plots/`:

1. class_distribution.png
2. learning_curves.png
3. model_architecture.png
4. confusion_matrix.png
5. sample_predictions.png

### 5.2 Generated Figure Previews

![Learning Curves](plots/learning_curves.png)
![Confusion Matrix](plots/confusion_matrix.png)
![Sample Predictions](plots/sample_predictions.png)

### 5.1 Terminal Excerpt

```text
ALL VISUALIZATIONS GENERATED SUCCESSFULLY!
Check the '.../plots/' directory for output files:
- class_distribution.png
- learning_curves.png
- model_architecture.png
- confusion_matrix.png
- sample_predictions.png
```

## 6. Reproducibility Note

All stages completed with exit code 0, indicating a stable end-to-end pipeline for this execution snapshot.
