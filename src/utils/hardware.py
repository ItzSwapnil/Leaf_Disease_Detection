from __future__ import annotations

import os
import subprocess


def _tf():
    import tensorflow as tf

    return tf


def _blackwell_gpu_present() -> bool:
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
    except Exception:
        return False
    gpu_info_lower = gpu_info.lower()
    return "rtx 50" in gpu_info_lower or "blackwell" in gpu_info_lower


def configure_tensorflow():
    tf = _tf()

    try:
        gpu_mode = os.getenv("LEAF_TF_GPU_MODE", "gpu").strip().lower()
        if gpu_mode in {"cpu", "off", "disable", "disabled"}:
            print("[*] LEAF_TF_GPU_MODE=cpu; disabling TensorFlow GPU backend.")
            try:
                tf.config.set_visible_devices([], "GPU")
            except RuntimeError as exc:
                print(f"TensorFlow GPU visibility was already initialized: {exc}")
        elif gpu_mode == "auto" and _blackwell_gpu_present():
            print("[*] Blackwell GPU detected via nvidia-smi.")
            print(
                "[*] LEAF_TF_GPU_MODE=auto; disabling TensorFlow GPU backend "
                "to prevent JIT compilation compatibility issues."
            )
            try:
                tf.config.set_visible_devices([], "GPU")
            except RuntimeError as exc:
                print(f"TensorFlow GPU visibility was already initialized: {exc}")
        elif _blackwell_gpu_present():
            print(
                "[*] Blackwell GPU detected. LEAF_TF_GPU_MODE=gpu; TensorFlow "
                "will use GPU and may JIT PTX on first inference."
            )

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
                    gpu_details.append(details.get("device_name", gpu.name))
                except Exception:
                    gpu_details.append(gpu.name)
            print(
                f"GPU devices detected ({len(gpus)}): {', '.join(gpu_details)}"
            )
        else:
            print("No GPU detected; using CPU backend.")
    except Exception as exc:
        print(f"Hardware configuration failed: {exc}")
    return tf.config.list_physical_devices()


def get_compute_info() -> dict:
    tf = _tf()

    try:
        gpus = tf.config.list_physical_devices("GPU")
        cpus = tf.config.list_physical_devices("CPU")
        gpu_names = []
        for gpu in gpus:
            try:
                details = tf.config.experimental.get_device_details(gpu)
                gpu_names.append(details.get("device_name", gpu.name))
            except Exception:
                gpu_names.append(gpu.name)

        return {
            "backend": "GPU" if gpus else "CPU",
            "gpu_count": len(gpus),
            "cpu_count": len(cpus),
            "gpu_names": gpu_names,
            "is_gpu_active": bool(gpus),
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
    tf = _tf()

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        print(f"Using MirroredStrategy across {len(gpus)} GPUs")
        return tf.distribute.MirroredStrategy()

    if len(gpus) == 1:
        print("Using single GPU with default strategy")
    else:
        print("Using CPU with default strategy")

    return tf.distribute.get_strategy()
