"""
TensorFlow hardware helpers to prefer GPU and enable memory growth.
"""

import tensorflow as tf


def configure_tensorflow():
    """Enable GPU memory growth and log visible devices."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                # Continue even if a single device fails; log below.
                pass
        print(f"TensorFlow devices (GPU preferred): {gpus if gpus else 'none'}")
    except Exception as exc:
        print(f"TensorFlow hardware configuration failed: {exc}")
    return tf.config.list_physical_devices()

