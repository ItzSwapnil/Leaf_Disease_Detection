from __future__ import annotations

import os
import subprocess

import torch


def _blackwell_gpu_present() -> bool:
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
    except Exception:
        return False
    gpu_info_lower = gpu_info.lower()
    return "rtx 50" in gpu_info_lower or "blackwell" in gpu_info_lower

def _setup_gpu_visibility():
    gpu_mode = os.getenv("LEAF_GPU_MODE", "auto").strip().lower()
    if gpu_mode in {"cpu", "off", "disable", "disabled"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

_setup_gpu_visibility()

def configure_hardware():
    """Configure hardware for PyTorch."""
    try:
        gpu_mode = os.getenv("LEAF_GPU_MODE", "auto").strip().lower()
        if gpu_mode in {"cpu", "off", "disable", "disabled"}:
            print("[*] LEAF_GPU_MODE=cpu; using CPU backend.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            print(f"GPU devices detected ({device_count}): {', '.join(gpu_names)}")

            # Enable TF32 for Ampere+ GPUs (speeds up training without much precision loss)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Use deterministic algorithms if needed
            # torch.backends.cudnn.benchmark = True
        else:
            print("No GPU detected; using CPU backend.")

    except Exception as exc:
        print(f"Hardware configuration failed: {exc}")

def get_compute_info() -> dict:
    """Get information about available compute resources."""
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            return {
                "backend": "GPU",
                "gpu_count": device_count,
                "cpu_count": os.cpu_count() or 1,
                "gpu_names": gpu_names,
                "is_gpu_active": True,
                "torch_version": torch.__version__,
            }
        else:
            return {
                "backend": "CPU",
                "gpu_count": 0,
                "cpu_count": os.cpu_count() or 1,
                "gpu_names": [],
                "is_gpu_active": False,
                "torch_version": torch.__version__,
            }
    except Exception as exc:
        return {
            "backend": "CPU",
            "gpu_count": 0,
            "cpu_count": os.cpu_count() or 1,
            "gpu_names": [],
            "is_gpu_active": False,
            "torch_version": getattr(torch, "__version__", "unknown"),
            "error": str(exc),
        }

def get_device() -> torch.device:
    """Get the primary PyTorch device."""
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        return torch.device("cuda")
    return torch.device("cpu")
