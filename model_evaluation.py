import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from keras.applications.efficientnet_v2 import preprocess_input
from config import FINAL_MODEL_PATH, VAL_DIR, IMG_SIZE
from hardware import configure_tensorflow
from model_paths import resolve_keras_model_path
import os
from sklearn.metrics import precision_recall_fscore_support

def main():
    configure_tensorflow()

    model_path = resolve_keras_model_path([FINAL_MODEL_PATH])
    model = load_model(model_path)

    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels='inferred',
        label_mode='int',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        shuffle=False,
    )
    val_gen = val_ds.map(
        lambda x, y: (preprocess_input(x), tf.one_hot(y, depth=len(val_ds.class_names))),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    print("\n--- validating saved model ---")
    metrics = model.evaluate(val_gen, verbose=1, return_dict=True)
    loss = float(metrics.get('loss', 0.0))
    acc = float(metrics.get('accuracy', metrics.get('acc', 0.0)))
    top3_acc = metrics.get('top3_acc')

    predictions = model.predict(val_gen, verbose=1)
    y_pred = predictions.argmax(axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in val_ds], axis=0)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='macro',
        zero_division=0,
    )

    print(f"\nValidation loss: {loss:.4f}")
    print(f"Validation accuracy: {acc * 100:.2f}%")
    if top3_acc is not None:
        print(f"Validation top3 accuracy: {float(top3_acc) * 100:.2f}%")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1 score: {f1:.4f}")


if __name__ == "__main__":
    main()
