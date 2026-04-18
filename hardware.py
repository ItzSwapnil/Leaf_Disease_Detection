import tensorflow as tf


def configure_tensorflow():

    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

        if gpus:
            gpu_details = []
            for gpu in gpus:
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    name = details.get("device_name", gpu.name)
                    gpu_details.append(name)
                except Exception:
                    gpu_details.append(gpu.name)
            print(f"GPU devices detected ({len(gpus)}): {', '.join(gpu_details)}")
        else:
            print("No GPU detected; using CPU backend.")
    except Exception as exc:
        print(f"Hardware configuration failed: {exc}")
    return tf.config.list_physical_devices()


def get_compute_info() -> dict:

    try:
        gpus = tf.config.list_physical_devices("GPU")
        cpus = tf.config.list_physical_devices("CPU")

        backend = "GPU" if gpus else "CPU"

        gpu_names = []
        for gpu in gpus:
            try:
                details = tf.config.experimental.get_device_details(gpu)
                gpu_names.append(details.get("device_name", gpu.name))
            except Exception:
                gpu_names.append(gpu.name)

        return {
            "backend": backend,
            "gpu_count": len(gpus),
            "cpu_count": len(cpus),
            "gpu_names": gpu_names,
            "is_gpu_active": len(gpus) > 0,
            "tf_version": tf.__version__,
        }
    except Exception as exc:
        return {
            "backend": "CPU",
            "gpu_count": 0,
            "cpu_count": 0,
            "gpu_names": [],
            "is_gpu_active": False,
            "tf_version": tf.__version__,
            "error": str(exc),
        }


def get_training_strategy():

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        print(f"Using MirroredStrategy across {len(gpus)} GPUs")
        return tf.distribute.MirroredStrategy()

    if len(gpus) == 1:
        print("Using single GPU with default strategy")
    else:
        print("Using CPU with default strategy")

    return tf.distribute.get_strategy()
