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


def get_compute_info():
    """Return best-effort compute backend info for API/UI usage (GPU/CPU)."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')

        backend = 'GPU' if gpus else 'CPU'

        gpu_names = []
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            gpu_names.append(details.get('device_name', gpu.name))

        return {
            'backend': backend,
            'gpu_count': len(gpus),
            'cpu_count': len(cpus),
            'gpu_names': gpu_names,
            'is_gpu_active': len(gpus) > 0,
            'tf_version': tf.__version__,
        }
    except Exception as exc:
        return {
            'backend': 'CPU',
            'gpu_count': 0,
            'cpu_count': 0,
            'gpu_names': [],
            'is_gpu_active': False,
            'tf_version': tf.__version__,
            'error': str(exc),
        }


def get_training_strategy():
    """Select GPU/CPU distribution strategy."""
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        print(f"Using MirroredStrategy across {len(gpus)} GPUs")
        return tf.distribute.MirroredStrategy()

    if len(gpus) == 1:
        print("Using single GPU with default strategy")
    else:
        print("Using CPU with default strategy")

    return tf.distribute.get_strategy()

