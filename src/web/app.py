"""Flask web application for plant leaf disease detection and classification.

Serves a web interface for uploading leaf images and receiving disease
predictions with confidence scores, disease descriptions, treatment
recommendations, and prevention guidelines. Also provides a control panel
for triggering training, fine-tuning, evaluation, and figure generation jobs.
"""

import base64
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

# Set TensorFlow logging before TensorFlow is imported. GPU/CPU selection is
# controlled by LEAF_TF_GPU_MODE in src.utils.hardware.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

from src.core.backbones import list_backbone_names
from src.core.inference_guard import (
    assess_leaf_likelihood,
    compute_prediction_diagnostics,
    evaluate_inference_safety,
)
from src.core.leaf_detector import detect_leaf_presence
from src.core.leaf_detector_model import create_leaf_detector
from src.core.preprocessing import preprocess_array_for_model
from src.training.training_utils import WarmupCosineSchedule
from src.utils.config import (
    CLASS_INDICES_PATH,
    CONFIDENCE_REJECT_THRESHOLD,
    ENTROPY_REJECT_THRESHOLD,
    FINAL_MODEL_PATH,
    IMG_SIZE,
    MODELS_DIR,
    OOD_MSP_THRESHOLD,
    USE_YOLO_LEAF_DETECTION,
)
from src.utils.hardware import configure_tensorflow, get_compute_info
from src.utils.model_paths import resolve_keras_model_path


def _log_tf_runtime_info():
    """Log TensorFlow build info and detected GPU devices at startup."""
    try:
        print(f"TensorFlow version: {tf.__version__}")
        print(
            f"CUDA visible devices: {tf.config.list_physical_devices('GPU')}"
        )
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"Built with ROCm: {tf.test.is_built_with_rocm()}")
    except Exception as exc:  # Best effort; do not block app startup
        print(f"TensorFlow runtime probe failed: {exc}")


def _extract_disease_predictions(predictions):
    """Return disease probabilities from single-output or multi-output models."""
    if isinstance(predictions, dict):
        if "disease_output" in predictions:
            predictions = predictions["disease_output"]
        else:
            predictions = next(iter(predictions.values()))
    elif isinstance(predictions, (list, tuple)):
        predictions = predictions[-1]
    return np.asarray(predictions)


configure_tensorflow()
_log_tf_runtime_info()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
app.config["UPLOAD_FOLDER"] = "uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Create uploads folder
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global model variable
model = None
class_indices = None
MODEL_LOAD_ERROR = None
ACTIVE_MODEL_PATH = None
# Backward compatibility alias for older helper scripts.
MODEL_PATH = None
MODEL_CACHE: dict[str, Any] = {}
ACTIVE_BACKBONE = None
LEAF_DETECTOR_MODEL = None
YOLO_FOCUS_DETECTOR = None

# Store per-job console history in memory.
# Set limit for truncation in low-memory envs.
JOB_LOG_LIMIT = None
JOBS: dict[str, Any] = {}
JOBS_LOCK = threading.Lock()
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
KERAS_BATCH_PROGRESS_RE = re.compile(r"^\d+/\d+\s+")

TRAIN_SCRIPT = "src/training/train_model.py"
TRAIN_DESC = "Run baseline EfficientNet training pipeline."
TRAIN_BACKBONES = list_backbone_names()
TRAIN_OPTIMIZER_OPTIONS = ["AdamW", "Adam", "SGD", "RMSprop"]
TRAIN_SAVE_MODES = [
    {"value": "with_optimizer", "label": "Save with optimizer state"},
    {
        "value": "without_optimizer",
        "label": "Save without optimizer state (inference-ready)",
    },
    {"value": "all", "label": "Save both variants"},
]

CONTROL_ACTIONS = {
    "train": {
        "label": "Train Model",
        "script": TRAIN_SCRIPT,
        "description": "Train from scratch and save checkpoint (models/leaf_disease_checkpoint.keras)",
    },
    "fine_tune": {
        "label": "Fine Tune Model",
        "script": "src/training/fine_tune_model.py",
        "description": "Fine-tune checkpoint into classifier (models/leaf_disease_classifier.keras)",
    },
    "refine": {
        "label": "Refine Model",
        "script": "src/training/refine_model.py",
        "description": "Refine classifier into deployment model (models/leaf_disease_refined.keras)",
    },
    "evaluate": {
        "label": "Evaluate Model",
        "script": "src/evaluation/evaluate_model.py",
        "description": "run validation and eval metrics",
    },
    "generate_figures": {
        "label": "Generate Figures",
        "script": "scripts/generate_figures.py",
        "description": "build plots and analysis artifacts",
    },
    "train_leaf_detector": {
        "label": "Train Leaf Detector",
        "script": "src/training/train_leaf_detector.py",
        "description": "train binary leaf/non-leaf detector used in stage-1 inference",
    },
    "gradcam_check": {
        "label": "Grad-CAM Check",
        "script": "scripts/gradcam_check.py",
        "description": "generate Grad-CAM heatmaps to verify model focuses on leaf, not background",
    },
}


def _resolve_model_path():
    return resolve_keras_model_path([FINAL_MODEL_PATH])


def _model_option_name(model_path):
    """Render a stable model option name relative to MODELS_DIR when possible."""
    models_root = os.path.abspath(str(MODELS_DIR))
    abs_path = os.path.abspath(str(model_path))
    rel = os.path.relpath(abs_path, models_root)
    if rel.startswith(".."):
        return os.path.basename(abs_path)
    return rel.replace(os.sep, "/")


def _list_available_model_paths():
    candidates = []
    models_root = os.path.abspath(str(MODELS_DIR))
    if not os.path.isdir(models_root):
        return candidates

    for root, _, files in os.walk(models_root):
        for filename in files:
            if filename.lower().endswith(".keras"):
                path = os.path.join(root, filename)
                if os.path.isfile(path):
                    candidates.append(os.path.abspath(path))

    return sorted(candidates)


def _resolve_requested_model_path(model_name=None):
    available_paths = _list_available_model_paths()
    option_to_path = {
        _model_option_name(path): path for path in available_paths
    }
    basename_to_paths = {}
    for path in available_paths:
        basename_to_paths.setdefault(os.path.basename(path), []).append(path)

    default_path = None
    try:
        default_path = os.path.abspath(_resolve_model_path())
    except Exception:
        default_path = None

    if not model_name:
        if default_path and os.path.exists(default_path):
            return default_path
        if available_paths:
            return available_paths[0]
        raise ValueError(
            "No model files were found under models/. Add at least one .keras model."
        )

    model_name = str(model_name).strip().replace("\\", "/")
    if not model_name:
        if default_path and os.path.exists(default_path):
            return default_path
        if available_paths:
            return available_paths[0]
        raise ValueError(
            "No model files were found under models/. Add at least one .keras model."
        )

    if model_name in option_to_path:
        return option_to_path[model_name]

    # Allow unique basename selection.
    if model_name in basename_to_paths:
        matches = basename_to_paths[model_name]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            "Model name is ambiguous across subfolders. "
            "Select using relative path under models/ (e.g. EfficientNetv2B0/model.keras)."
        )

    available = sorted(option_to_path.keys())
    shown = ", ".join(available[:10])
    if len(available) > 10:
        shown += f", ... (+{len(available) - 10} more)"

    raise ValueError(
        f"Unknown model '{model_name}'. Available options: {shown}"
    )


def _get_inference_model(model_name=None):
    global model, ACTIVE_MODEL_PATH, MODEL_PATH

    from src.utils.config import ENSEMBLE_MODEL_PATHS

    if not model_name and ENSEMBLE_MODEL_PATHS:
        models = []
        for path in ENSEMBLE_MODEL_PATHS:
            # Resolve absolute or relative path.
            try:
                target_path = _resolve_requested_model_path(path)
                if target_path in MODEL_CACHE:
                    models.append(MODEL_CACHE[target_path])
                else:
                    loaded = _load_model_robust(target_path)
                    MODEL_CACHE[target_path] = loaded
                    models.append(loaded)
            except ValueError as e:
                print(
                    f"[WARNING] Ensemble model path {path} failed to load: {e}"
                )
        if models:
            model = models
            ACTIVE_MODEL_PATH = "ensemble"
            MODEL_PATH = "ensemble"
            return model, ACTIVE_MODEL_PATH

    target_path = _resolve_requested_model_path(model_name)
    if target_path in MODEL_CACHE:
        model = MODEL_CACHE[target_path]
        ACTIVE_MODEL_PATH = target_path
        MODEL_PATH = target_path
        return model, ACTIVE_MODEL_PATH

    loaded = _load_model_robust(target_path)
    MODEL_CACHE[target_path] = loaded
    model = loaded
    ACTIVE_MODEL_PATH = target_path
    MODEL_PATH = target_path
    return model, ACTIVE_MODEL_PATH


def _load_model_robust(model_path: str):
    """Load model with compatibility fallback for older KerasHub ViT configs."""
    custom_objects = {"WarmupCosineSchedule": WarmupCosineSchedule}
    try:
        return load_model(
            model_path, custom_objects=custom_objects, compile=False
        )
    except TypeError as exc:
        error_text = str(exc)
        if "ViTPatchingAndEmbedding" not in error_text:
            raise

        if not _patch_vit_layer_init_for_compat():
            raise RuntimeError(
                "Failed to load ViT/DINO checkpoint due to keras-hub version mismatch. "
                "Install a compatible keras-hub version or retrain with current stack."
            ) from exc

        print(
            "Detected KerasHub ViT checkpoint compatibility mismatch; "
            "retrying load with compatibility shim."
        )
        return load_model(
            model_path, custom_objects=custom_objects, compile=False
        )


def _patch_vit_layer_init_for_compat() -> bool:
    """Patch keras-hub ViT layer init to ignore legacy serialized kwargs."""
    try:
        from keras_hub.src.models.vit import vit_layers

        layer_cls = vit_layers.ViTPatchingAndEmbedding
    except Exception:
        return False

    if getattr(layer_cls, "_leaf_compat_patched", False):
        return True

    original_init = layer_cls.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.pop("num_patches", None)
        kwargs.pop("num_positions", None)
        image_size = kwargs.get("image_size")
        if isinstance(image_size, int):
            kwargs["image_size"] = (image_size, image_size)
        patch_size = kwargs.get("patch_size")
        if isinstance(patch_size, int):
            kwargs["patch_size"] = (patch_size, patch_size)
        return original_init(self, *args, **kwargs)

    layer_cls.__init__ = _patched_init
    layer_cls._leaf_compat_patched = True
    return True


def _infer_backbone_name(active_model, model_path: str | None = None) -> str:
    """Best-effort backbone detection for selecting the correct preprocessing."""
    path_hint = (model_path or "").lower()
    if any(token in path_hint for token in ["dino", "vit", "refined"]):
        return "DINOv3"

    try:
        for layer in getattr(active_model, "layers", []):
            layer_name = (getattr(layer, "name", "") or "").lower()
            class_name = layer.__class__.__name__.lower()
            if any(
                token in layer_name
                for token in ["vit", "dino", "patching", "transformer"]
            ):
                return "DINOv3"
            if any(
                token in class_name
                for token in ["vit", "patch", "transformer"]
            ):
                return "DINOv3"
    except Exception:
        pass

    return "EfficientNetV2B0"


def _model_has_internal_preprocessing(active_model) -> bool:
    """Detect whether a model already performs input preprocessing internally."""
    try:
        for layer in getattr(active_model, "layers", []):
            lname = (getattr(layer, "name", "") or "").lower()
            lclass = layer.__class__.__name__.lower()
            if "dinov3_preprocess" in lname:
                return True
            if lclass in {"rescaling", "normalization"}:
                return True
    except Exception:
        return False
    return False


def _infer_model_num_classes(active_model):
    """Best-effort extraction of output class count from a loaded Keras model."""
    try:
        output_shape = active_model.output_shape
        output_names = list(getattr(active_model, "output_names", []) or [])
        if isinstance(output_shape, dict):
            output_shape = output_shape.get("disease_output") or next(
                iter(output_shape.values())
            )
        elif isinstance(output_shape, list):
            if "disease_output" in output_names:
                output_shape = output_shape[
                    output_names.index("disease_output")
                ]
            else:
                output_shape = output_shape[-1]
        if not output_shape:
            return None
        # Most classifiers expose shape like (None, num_classes).
        last_dim = output_shape[-1]
        if isinstance(last_dim, int) and last_dim > 0:
            return int(last_dim)
    except Exception:
        return None
    return None


def _load_class_indices_map(path):
    """Load and reverse a class-indices JSON map (label -> idx) into idx -> label."""
    with open(path, "r") as f:
        label_to_idx = json.load(f)
    return {int(v): k for k, v in label_to_idx.items()}


def _resolve_class_indices_for_model(model_path, active_model):
    """Select class indices JSON that matches the active model output size when possible."""
    expected_classes = _infer_model_num_classes(active_model)
    model_dir = os.path.dirname(model_path)
    model_stem = os.path.splitext(os.path.basename(model_path))[0]

    candidates = [
        os.path.join(model_dir, "class_indices.json"),
        os.path.join(model_dir, f"{model_stem}.class_indices.json"),
        os.path.join(model_dir, f"{model_stem}_class_indices.json"),
        str(CLASS_INDICES_PATH),
    ]

    checked = []
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            idx_to_label = _load_class_indices_map(candidate)
            checked.append((candidate, len(idx_to_label)))
            if (
                expected_classes is None
                or len(idx_to_label) == expected_classes
            ):
                print(
                    f"Using class indices: {candidate} "
                    f"({len(idx_to_label)} classes, model expects {expected_classes})"
                )
                return idx_to_label
        except Exception as exc:
            print(f"Skipping invalid class indices file {candidate}: {exc}")

    if checked:
        # Fallback to first mapping and warn on mismatch.
        fallback_path = checked[0][0]
        idx_to_label = _load_class_indices_map(fallback_path)
        print(
            "WARNING: No class index file matched model output dimension. "
            f"Using fallback {fallback_path} ({len(idx_to_label)} classes; "
            f"model expects {expected_classes}). Predictions may map to wrong labels."
        )
        return idx_to_label

    raise FileNotFoundError(
        "No readable class_indices JSON found. Tried: " + ", ".join(candidates)
    )


def _is_allowed_upload(filename):
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTENSIONS


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _normalize_train_optimizer(value):
    raw = str(value or "").strip().lower()
    for option in TRAIN_OPTIMIZER_OPTIONS:
        if option.lower() == raw:
            return option
    return None


def _normalize_save_mode(value):
    raw = str(value or "").strip().lower().replace("-", "_")
    allowed = {mode["value"] for mode in TRAIN_SAVE_MODES}
    if raw in allowed:
        return raw
    return None


def _normalize_leaf_detection_mode(value):
    raw = str(value or "auto").strip().lower()
    if raw in {"auto", "model", "heuristic", "off"}:
        return raw
    return "auto"


def _get_leaf_detector_model():
    global LEAF_DETECTOR_MODEL
    if LEAF_DETECTOR_MODEL is not None:
        return LEAF_DETECTOR_MODEL

    try:
        LEAF_DETECTOR_MODEL = create_leaf_detector()
        print("Leaf detector model loaded for UI-controlled inference.")
    except Exception as exc:
        print(f"Leaf detector model unavailable: {exc}")
        LEAF_DETECTOR_MODEL = None
    return LEAF_DETECTOR_MODEL


def _get_yolo_leaf_detector():
    global YOLO_FOCUS_DETECTOR
    if YOLO_FOCUS_DETECTOR is not None:
        return YOLO_FOCUS_DETECTOR

    try:
        from src.core.yolo_leaf import YOLOLeafDetector
        YOLO_FOCUS_DETECTOR = YOLOLeafDetector()
    except Exception as exc:
        print(f"YOLOLeafDetector unavailable: {exc}")
        YOLO_FOCUS_DETECTOR = None
    return YOLO_FOCUS_DETECTOR


def _parse_pipeline_options(payload):
    source = payload or {}
    options = {
        "leaf_detection_mode": _normalize_leaf_detection_mode(
            source.get("leaf_detection_mode")
        ),
        "use_background_removal": _to_bool(
            source.get("use_background_removal")
        ),
        "use_safety_gate": _to_bool(source.get("use_safety_gate")),
    }
    if "use_background_removal" not in source:
        options["use_background_removal"] = True
    if "use_safety_gate" not in source:
        options["use_safety_gate"] = True
    return options


def _create_job(
    action_key, archive_logs=False, base_model=None, train_options=None
):
    action = CONTROL_ACTIONS[action_key]
    script = action["script"]
    script_args = []
    train_options = train_options or {}

    train_fraction = train_options.get("train_fraction")
    train_fraction_pct = train_options.get("train_fraction_pct")
    optimizer = train_options.get("optimizer")
    save_mode = train_options.get("save_mode")
    class_equalizer = train_options.get("class_equalizer")
    must_review = train_options.get("must_review")

    if action_key == "train" and base_model:
        script_args = ["--base-model", str(base_model)]
    if action_key == "train" and train_fraction is not None:
        script_args.extend(
            ["--train-fraction", f"{float(train_fraction):.6f}"]
        )
    if action_key == "train" and optimizer:
        script_args.extend(["--optimizer", str(optimizer)])
    if action_key == "train" and save_mode:
        script_args.extend(["--save-mode", str(save_mode)])
    if action_key == "train" and class_equalizer is not None:
        script_args.extend(
            ["--class-equalizer", "on" if bool(class_equalizer) else "off"]
        )
    if action_key == "train" and must_review is not None:
        script_args.extend(
            ["--must-review", "on" if bool(must_review) else "off"]
        )

    command = [sys.executable, script, *script_args]
    env_overrides = {
        "LEAF_SAVE_LOG_ARCHIVE": "1" if archive_logs else "0",
        "LEAF_SAVE_RUN_MANIFESTS": "1" if archive_logs else "0",
    }
    if action_key == "train" and base_model:
        env_overrides["LEAF_BASE_MODEL"] = str(base_model)
    if action_key == "train" and train_fraction is not None:
        env_overrides["LEAF_TRAIN_DATA_FRACTION"] = (
            f"{float(train_fraction):.6f}"
        )
    if action_key == "train" and optimizer:
        env_overrides["LEAF_TRAIN_OPTIMIZER"] = str(optimizer)
    if action_key == "train" and save_mode:
        env_overrides["LEAF_SAVE_MODE"] = str(save_mode)
    if action_key == "train" and class_equalizer is not None:
        env_overrides["LEAF_CLASS_EQUALIZER"] = "1" if class_equalizer else "0"
    if action_key == "train" and must_review is not None:
        env_overrides["LEAF_MUST_REVIEW"] = "1" if must_review else "0"

    job_id = uuid.uuid4().hex
    now = time.time()
    job = {
        "id": job_id,
        "action": action_key,
        "label": action["label"],
        "description": action["description"],
        "script": script,
        "script_args": script_args,
        "command": " ".join(command),
        "archive_logs": bool(archive_logs),
        "base_model": base_model,
        "train_fraction": train_fraction,
        "train_fraction_pct": train_fraction_pct,
        "optimizer": optimizer,
        "save_mode": save_mode,
        "class_equalizer": class_equalizer,
        "must_review": must_review,
        "env_overrides": env_overrides,
        "status": "starting",
        "start_time": now,
        "end_time": None,
        "return_code": None,
        "progress_pct": 0.0,
        "eta_seconds": None,
        "progress_stage": "pending",
        "logs": [],
        "stop_requested": False,
        "process": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job


def _append_job_log(job, line):
    if not line:
        return

    cleaned = line.replace("\r", " ").replace("\b", "")
    cleaned = ANSI_ESCAPE_RE.sub("", cleaned)
    cleaned = "".join(
        ch for ch in cleaned if ch == "\t" or 32 <= ord(ch) <= 126
    )
    cleaned = cleaned.strip()

    if not cleaned:
        return

    if job["logs"] and job["logs"][-1] == cleaned:
        return

    # Keep a single live-updating batch progress line.
    is_batch_progress = (
        KERAS_BATCH_PROGRESS_RE.match(cleaned) is not None
        and "ms/step" in cleaned
    )
    if is_batch_progress and job["logs"]:
        prev = job["logs"][-1]
        prev_is_batch = (
            KERAS_BATCH_PROGRESS_RE.match(prev) is not None
            and "ms/step" in prev
        )
        if prev_is_batch:
            job["logs"][-1] = cleaned
            return

    job["logs"].append(cleaned)
    if (
        isinstance(JOB_LOG_LIMIT, int)
        and JOB_LOG_LIMIT > 0
        and len(job["logs"]) > JOB_LOG_LIMIT
    ):
        job["logs"] = job["logs"][-JOB_LOG_LIMIT:]


def _parse_progress_line(job, raw_line):
    prefix = "TRAINING_PROGRESS "
    line = (raw_line or "").strip()
    if not line.startswith(prefix):
        return False

    try:
        payload = json.loads(line[len(prefix) :])
        progress = float(
            payload.get("progress_pct", job.get("progress_pct", 0.0))
        )
        eta = payload.get("eta_seconds")

        job["progress_pct"] = max(0.0, min(100.0, progress))
        job["eta_seconds"] = None if eta is None else max(0.0, float(eta))
        job["progress_stage"] = payload.get(
            "stage", job.get("progress_stage", "running")
        )
        return True
    except Exception:
        return False


def _run_job(job):
    script_path = os.path.join(os.getcwd(), job["script"])
    command = [sys.executable, script_path, *(job.get("script_args") or [])]
    child_env = dict(os.environ)
    child_env.update(job.get("env_overrides") or {})
    try:
        process = subprocess.Popen(
            command,
            cwd=os.getcwd(),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job["process"] = process
        job["status"] = "running"
        _append_job_log(job, f"Started: {' '.join(command)}")

        assert process.stdout is not None
        for line in process.stdout:
            if _parse_progress_line(job, line):
                continue
            _append_job_log(job, line)

        return_code = process.wait()
        job["return_code"] = int(return_code)
        job["end_time"] = time.time()

        if job["stop_requested"]:
            job["status"] = "stopped"
        elif return_code == 0:
            job["status"] = "completed"
            job["progress_pct"] = 100.0
            job["eta_seconds"] = 0.0
            _append_job_log(job, "Completed successfully.")
        else:
            job["status"] = "failed"
            _append_job_log(job, f"Exited with code {return_code}.")

    except Exception as exc:
        job["status"] = "failed"
        job["end_time"] = time.time()
        _append_job_log(job, f"Error: {exc}")
    finally:
        job["process"] = None


def _job_response(job):
    runtime_seconds = None
    end_time = job["end_time"] if job["end_time"] is not None else time.time()
    if job["start_time"] is not None:
        runtime_seconds = round(end_time - job["start_time"], 1)

    return {
        "id": job["id"],
        "action": job["action"],
        "label": job["label"],
        "description": job["description"],
        "script": job["script"],
        "base_model": job.get("base_model"),
        "train_fraction": job.get("train_fraction"),
        "train_fraction_pct": job.get("train_fraction_pct"),
        "optimizer": job.get("optimizer"),
        "save_mode": job.get("save_mode"),
        "class_equalizer": job.get("class_equalizer"),
        "must_review": job.get("must_review"),
        "command": job["command"],
        "archive_logs": bool(job.get("archive_logs", False)),
        "status": job["status"],
        "start_time": job["start_time"],
        "end_time": job["end_time"],
        "runtime_seconds": runtime_seconds,
        "return_code": job["return_code"],
        "progress_pct": round(float(job.get("progress_pct", 0.0)), 2),
        "eta_seconds": job.get("eta_seconds"),
        "progress_stage": job.get("progress_stage", "pending"),
        "logs": list(job["logs"]),
    }


# Disease information database
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "description": "A fungal disease caused by Venturia inaequalis that affects apple trees.",
        "symptoms": "Dark, olive-green to brown spots on leaves and fruit. Leaves may curl and drop early.",
        "treatment": "Apply fungicides during spring. Remove infected leaves and fruit. Improve air circulation.",
        "prevention": "Plant resistant varieties. Rake and destroy fallen leaves. Prune trees for better airflow.",
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "description": "A fungal disease caused by Botryosphaeria obtusa affecting fruit, leaves, and bark.",
        "symptoms": "Brown spots with concentric rings on leaves. Rotting fruit with black, mummified appearance.",
        "treatment": "Remove infected plant parts. Apply fungicides during growing season.",
        "prevention": "Maintain tree health. Remove mummified fruits. Prune dead wood.",
    },
    "Apple___Brown spot": {
        "plant": "Apple",
        "disease": "Brown Spot",
        "description": "A fungal infection causing brown lesions on apple leaves.",
        "symptoms": "Brown circular spots on leaves, potentially leading to early defoliation.",
        "treatment": "Apply appropriate fungicides. Remove affected foliage.",
        "prevention": "Ensure good air circulation. Avoid overhead watering.",
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "description": "A fungal disease requiring both apple and cedar/juniper trees to complete its life cycle.",
        "symptoms": "Yellow-orange spots on leaves with small black dots. Tube-like structures on leaf undersides.",
        "treatment": "Apply fungicides in spring. Remove nearby cedar/juniper trees if possible.",
        "prevention": "Plant resistant varieties. Remove galls from cedars before spring.",
    },
    "Apple___Grey spot": {
        "plant": "Apple",
        "disease": "Grey Spot",
        "description": "A fungal disease affecting apple leaves causing grey lesions.",
        "symptoms": "Grey to brown spots on leaves, may cause premature leaf drop.",
        "treatment": "Apply fungicides. Improve orchard sanitation.",
        "prevention": "Remove fallen leaves. Ensure proper spacing between trees.",
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "description": "This apple leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present. Leaf appears healthy with normal coloration.",
        "treatment": "No treatment needed. Continue regular plant care.",
        "prevention": "Maintain good growing conditions and regular monitoring.",
    },
    "Apple___Mosaic": {
        "plant": "Apple",
        "disease": "Mosaic Virus",
        "description": "A viral disease causing mottled patterns on apple leaves.",
        "symptoms": "Yellow and green mosaic patterns on leaves. Stunted growth possible.",
        "treatment": "No cure for viral diseases. Remove infected plants to prevent spread.",
        "prevention": "Use virus-free planting material. Control insect vectors.",
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "description": "This blueberry leaf shows no signs of disease.",
        "symptoms": "No disease symptoms. Healthy green coloration.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper soil pH and nutrition.",
    },
    "Cherry___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "description": "This cherry leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular monitoring and proper care.",
    },
    "Cherry___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "description": "A fungal disease causing white powdery coating on leaves.",
        "symptoms": "White powdery spots on leaves and shoots. Leaves may curl and distort.",
        "treatment": "Apply sulfur-based or systemic fungicides. Remove heavily infected parts.",
        "prevention": "Ensure good air circulation. Avoid overhead watering. Plant resistant varieties.",
    },
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Gray Leaf Spot",
        "description": "A fungal disease caused by Cercospora zeae-maydis.",
        "symptoms": "Rectangular gray to tan lesions on leaves running parallel to veins.",
        "treatment": "Apply foliar fungicides. Use resistant hybrids.",
        "prevention": "Rotate crops. Till under crop residue. Plant resistant varieties.",
    },
    "Corn___Common_rust": {
        "plant": "Corn",
        "disease": "Common Rust",
        "description": "A fungal disease caused by Puccinia sorghi.",
        "symptoms": "Small, circular to elongated brown pustules on both leaf surfaces.",
        "treatment": "Apply fungicides if severe. Usually not economically damaging.",
        "prevention": "Plant resistant hybrids. Early planting can help avoid infection.",
    },
    "Corn___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "description": "This corn leaf shows no signs of disease.",
        "symptoms": "No disease symptoms. Normal green coloration.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper nutrition and irrigation.",
    },
    "Corn___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "description": "A fungal disease caused by Exserohilum turcicum.",
        "symptoms": "Long, elliptical gray-green to tan lesions on leaves.",
        "treatment": "Apply foliar fungicides. Remove crop debris.",
        "prevention": "Use resistant hybrids. Rotate crops. Till under residue.",
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "description": "A fungal disease caused by Guignardia bidwellii.",
        "symptoms": "Brown circular spots on leaves. Fruit shrivels and turns black (mummies).",
        "treatment": "Apply fungicides from bud break. Remove mummified fruit.",
        "prevention": "Prune for good air circulation. Remove infected plant material.",
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "description": "A complex fungal disease affecting grapevines.",
        "symptoms": "Tiger-stripe pattern on leaves. Dark spots on berries. Sudden vine collapse.",
        "treatment": "No effective treatment. Remove severely affected vines.",
        "prevention": "Avoid large pruning wounds. Paint pruning cuts with fungicide.",
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "description": "This grape leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular monitoring and proper vineyard management.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight (Isariopsis)",
        "description": "A fungal disease causing leaf spots on grapevines.",
        "symptoms": "Irregular brown spots on leaves with dark borders.",
        "treatment": "Apply fungicides. Remove infected leaves.",
        "prevention": "Improve air circulation. Avoid overhead irrigation.",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing (Citrus Greening)",
        "description": "A devastating bacterial disease spread by psyllid insects.",
        "symptoms": "Yellowing of leaves in blotchy pattern. Misshapen, bitter fruit. Tree decline.",
        "treatment": "No cure. Remove infected trees. Control psyllid vectors.",
        "prevention": "Use disease-free nursery stock. Control Asian citrus psyllid.",
    },
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease caused by Xanthomonas campestris.",
        "symptoms": "Small, dark spots on leaves that may fall out. Fruit has sunken spots.",
        "treatment": "Apply copper-based bactericides. Remove infected parts.",
        "prevention": "Plant resistant varieties. Avoid overhead irrigation.",
    },
    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "description": "This peach leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular care and monitoring.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease affecting pepper plants.",
        "symptoms": "Small, dark, water-soaked spots on leaves. Raised spots on fruit.",
        "treatment": "Apply copper-based sprays. Remove infected plants.",
        "prevention": "Use disease-free seeds. Rotate crops. Avoid overhead watering.",
    },
    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "Healthy",
        "description": "This pepper leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain good growing conditions.",
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "description": "A fungal disease caused by Alternaria solani.",
        "symptoms": "Dark brown spots with concentric rings (target-like) on older leaves.",
        "treatment": "Apply fungicides. Remove infected leaves.",
        "prevention": "Rotate crops. Use certified seed. Maintain plant vigor.",
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "description": "This potato leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular monitoring and proper irrigation.",
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "description": "A devastating disease caused by Phytophthora infestans (caused Irish Potato Famine).",
        "symptoms": "Water-soaked spots that turn brown. White mold on leaf undersides. Rapid plant death.",
        "treatment": "Apply fungicides immediately. Remove infected plants.",
        "prevention": "Use resistant varieties. Avoid overhead irrigation. Destroy infected tubers.",
    },
    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "description": "This raspberry leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Good pruning and air circulation.",
    },
    "Rice___Brown_Spot": {
        "plant": "Rice",
        "disease": "Brown Spot",
        "description": "A fungal disease caused by Bipolaris oryzae.",
        "symptoms": "Oval brown spots on leaves with gray centers.",
        "treatment": "Apply fungicides. Improve soil fertility.",
        "prevention": "Use resistant varieties. Balanced fertilization.",
    },
    "Rice___Healthy": {
        "plant": "Rice",
        "disease": "Healthy",
        "description": "This rice leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper water and nutrient management.",
    },
    "Rice___Leaf_Blast": {
        "plant": "Rice",
        "disease": "Leaf Blast",
        "description": "A serious fungal disease caused by Magnaporthe oryzae.",
        "symptoms": "Diamond-shaped spots with gray centers and brown borders.",
        "treatment": "Apply systemic fungicides. Drain fields if possible.",
        "prevention": "Use resistant varieties. Avoid excess nitrogen.",
    },
    "Rice___Neck_Blast": {
        "plant": "Rice",
        "disease": "Neck Blast",
        "description": "A severe form of rice blast affecting the panicle neck.",
        "symptoms": "Brown to black lesions on panicle neck. Panicle may break and fall.",
        "treatment": "Apply fungicides before heading. Remove infected panicles.",
        "prevention": "Plant resistant varieties. Balanced fertilization.",
    },
    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "description": "This soybean leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Crop rotation and proper spacing.",
    },
    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "description": "A common fungal disease affecting cucurbits.",
        "symptoms": "White powdery patches on leaves. Leaves may yellow and die.",
        "treatment": "Apply fungicides or baking soda solution. Remove infected leaves.",
        "prevention": "Plant resistant varieties. Ensure good air circulation.",
    },
    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "description": "This strawberry leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Proper spacing and mulching.",
    },
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "description": "A fungal disease caused by Diplocarpon earlianum.",
        "symptoms": "Irregular purple spots that merge. Leaf margins appear burned.",
        "treatment": "Apply fungicides. Remove infected leaves.",
        "prevention": "Use resistant varieties. Renovate beds after harvest.",
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease affecting tomato plants.",
        "symptoms": "Small, dark, water-soaked spots on leaves. Raised spots on fruit.",
        "treatment": "Apply copper-based bactericides. Remove infected parts.",
        "prevention": "Use disease-free seeds. Rotate crops. Avoid overhead watering.",
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "description": "A fungal disease caused by Alternaria solani.",
        "symptoms": "Dark brown spots with concentric rings on lower leaves first.",
        "treatment": "Apply fungicides. Remove infected leaves. Mulch around plants.",
        "prevention": "Rotate crops. Stake plants. Water at base of plants.",
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "description": "This tomato leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular monitoring and proper care.",
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "description": "A destructive disease caused by Phytophthora infestans.",
        "symptoms": "Large, irregular brown spots. White mold on undersides. Rapid spread.",
        "treatment": "Apply fungicides immediately. Remove infected plants.",
        "prevention": "Use resistant varieties. Improve air circulation. Avoid wet foliage.",
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "description": "A fungal disease caused by Passalora fulva.",
        "symptoms": "Pale green to yellow spots on upper leaf surface. Olive-brown mold below.",
        "treatment": "Improve ventilation. Apply fungicides. Remove infected leaves.",
        "prevention": "Reduce humidity. Space plants properly. Use resistant varieties.",
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "description": "A common fungal disease caused by Septoria lycopersici.",
        "symptoms": "Small, circular spots with dark borders and gray centers with black dots.",
        "treatment": "Apply fungicides. Remove infected lower leaves.",
        "prevention": "Rotate crops. Mulch around plants. Avoid overhead watering.",
    },
    "Tomato___Spider_mites_Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites",
        "description": "Tiny arachnid pests that feed on plant cells.",
        "symptoms": "Stippled, yellowing leaves. Fine webbing on undersides. Leaf drop.",
        "treatment": "Spray with water or insecticidal soap. Use miticides if severe.",
        "prevention": "Maintain plant health. Avoid dusty conditions. Introduce predatory mites.",
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "description": "A fungal disease caused by Corynespora cassiicola.",
        "symptoms": "Brown spots with concentric rings giving target-like appearance.",
        "treatment": "Apply fungicides. Remove infected leaves.",
        "prevention": "Improve air circulation. Avoid overhead irrigation.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "description": "A highly contagious viral disease.",
        "symptoms": "Mottled light and dark green pattern on leaves. Distorted growth.",
        "treatment": "No cure. Remove and destroy infected plants.",
        "prevention": "Use virus-free seeds. Disinfect tools. Wash hands before handling.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Yellow Leaf Curl Virus",
        "description": "A devastating viral disease spread by whiteflies.",
        "symptoms": "Upward curling of leaves. Yellowing. Stunted growth. Reduced fruit.",
        "treatment": "No cure. Remove infected plants. Control whiteflies.",
        "prevention": "Use resistant varieties. Control whitefly populations. Use reflective mulches.",
    },
    "Wheat brown spot disease": {
        "plant": "Wheat",
        "disease": "Brown Spot",
        "description": "A fungal disease affecting wheat leaves.",
        "symptoms": "Brown oval spots on leaves that may merge.",
        "treatment": "Apply fungicides. Remove crop residue.",
        "prevention": "Use resistant varieties. Crop rotation.",
    },
}


def load_model_and_classes():
    """Load the model and class indices"""
    global \
        model, \
        class_indices, \
        MODEL_LOAD_ERROR, \
        ACTIVE_MODEL_PATH, \
        MODEL_PATH, \
        ACTIVE_BACKBONE

    model = None
    class_indices = None
    MODEL_LOAD_ERROR = None
    ACTIVE_MODEL_PATH = None
    MODEL_PATH = None
    ACTIVE_BACKBONE = None

    print("Loading model...")
    model, model_path = _get_inference_model()
    MODEL_PATH = model_path

    # Use the first model in ensemble to infer backbone
    first_model = model[0] if isinstance(model, list) else model
    ACTIVE_BACKBONE = _infer_backbone_name(first_model, model_path)
    print(f"Loaded model file: {model_path}")
    print(f"Detected inference backbone: {ACTIVE_BACKBONE}")

    class_indices = _resolve_class_indices_for_model(model_path, model)
    print("Model loaded successfully.")
    return model, class_indices, ACTIVE_MODEL_PATH


def predict_disease(img_path, inference_model=None, pipeline_options=None):
    """Predict disease from image"""
    active_model = inference_model or model
    if active_model is None or class_indices is None:
        raise RuntimeError("Model or class indices are not loaded.")

    options = _parse_pipeline_options(pipeline_options)
    stage_details = []

    leaf_validation = assess_leaf_likelihood(img_path, IMG_SIZE)

    if options["leaf_detection_mode"] != "off":
        leaf_result = None
        leaf_source = "heuristic"

        if options["leaf_detection_mode"] in {"auto", "model"}:
            detector = _get_leaf_detector_model()
            if detector is not None:
                leaf_result = detector.predict(img_path)
                leaf_source = "trained_model"
            elif options["leaf_detection_mode"] == "model":
                raise RuntimeError(
                    "Leaf detector model is not available. "
                    "Train it from the Web UI using 'Train Leaf Detector'."
                )

        if leaf_result is None:
            leaf_result = detect_leaf_presence(img_path, img_size=IMG_SIZE)

        stage_details.append(
            {
                "stage": "leaf_detection",
                "source": leaf_source,
                "is_leaf": bool(leaf_result.get("is_leaf", False)),
                "leaf_score": float(leaf_result.get("leaf_score", 0.0)),
                "reason": leaf_result.get("reason", ""),
            }
        )

        if not leaf_result.get("is_leaf", False):
            return {
                "class_name": "Unknown",
                "confidence": 0.0,
                "plant": "Leaf Validation",
                "disease": "Not a leaf image",
                "description": "Stage-1 leaf detection rejected this image.",
                "symptoms": "No disease analysis was run because the image was not identified as a leaf.",
                "treatment": leaf_result.get("reason")
                or "Upload a clear close-up image of a single leaf.",
                "prevention": "Use a plain background and ensure the leaf fills most of the frame.",
                "is_healthy": False,
                "is_valid_leaf": False,
                "rejected": True,
                "validation": {
                    "leaf_score": float(leaf_result.get("leaf_score", 0.0)),
                    "vegetation_ratio": float(
                        leaf_validation.get("vegetation_ratio", 0.0)
                    ),
                    "confidence_margin": 0.0,
                    "entropy_bits": 0.0,
                    "rejection_reasons": [
                        leaf_result.get("reason") or "Leaf not detected"
                    ],
                },
                "pipeline": {
                    "options": options,
                    "stages": stage_details,
                },
            }

    model_input_array = None
    focus_overlay_b64 = None
    focus_overlay_enabled = bool(
        USE_YOLO_LEAF_DETECTION and options["use_background_removal"]
    )
    stage_details.append(
        {"stage": "leaf_focus_detection", "applied": focus_overlay_enabled}
    )

    if model_input_array is None:
        try:
            img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            model_input_array = np.expand_dims(image.img_to_array(img), axis=0)
        except Exception as exc:
            print(f"Failed to load image for classification: {exc}")

        if focus_overlay_enabled:
            detector = _get_yolo_leaf_detector()
            if detector is not None:
                try:
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is not None:
                        detection = detector.detect(img_bgr)
                        if detection["found"]:
                            x1, y1, x2, y2 = detection["bbox"]
                            img_boxed = img_bgr.copy()
                            cv2.rectangle(
                                img_boxed,
                                (x1, y1),
                                (x2, y2),
                                (34, 197, 94),
                                3,
                            )
                            cv2.putText(
                                img_boxed,
                                f"Leaf Focus ({int(detection['confidence'] * 100)}%)",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (34, 197, 94),
                                2,
                            )
                            img_boxed_rgb = cv2.cvtColor(
                                img_boxed, cv2.COLOR_BGR2RGB
                            )
                            img_pil = Image.fromarray(img_boxed_rgb)
                            buffered = io.BytesIO()
                            img_pil.save(buffered, format="JPEG")
                            focus_overlay_b64 = base64.b64encode(
                                buffered.getvalue()
                            ).decode("utf-8")
                except Exception as exc:
                    print(f"YOLOv26 focus visualization failed: {exc}")

    # Skip preprocessing if model has internal block.
    if not _model_has_internal_preprocessing(active_model):
        model_input_array = preprocess_array_for_model(
            model_input_array, backbone_name=ACTIVE_BACKBONE
        )

    # Make prediction
    if isinstance(active_model, list):
        all_preds = [
            _extract_disease_predictions(m.predict(model_input_array, verbose=0))[0]
            for m in active_model
        ]
        prediction_probs = np.mean(all_preds, axis=0)
        diagnostics = compute_prediction_diagnostics(prediction_probs)
    else:
        predictions = _extract_disease_predictions(
            active_model.predict(model_input_array, verbose=0)
        )
        diagnostics = compute_prediction_diagnostics(predictions[0])
    predicted_class_idx = int(diagnostics["top1_index"])
    confidence = float(diagnostics["top1_prob"]) * 100.0
    confidence_margin = float(diagnostics["confidence_margin"]) * 100.0
    entropy_bits = float(diagnostics["entropy_bits"])

    # Get class name
    class_name = class_indices.get(predicted_class_idx, "Unknown")

    safety = {"reject": False, "reasons": [], "uncertainty_score": 0}
    if options["use_safety_gate"]:
        safety = evaluate_inference_safety(
            diagnostics=diagnostics,
            leaf_validation=leaf_validation,
            confidence_threshold=CONFIDENCE_REJECT_THRESHOLD,
            entropy_threshold_bits=ENTROPY_REJECT_THRESHOLD,
            msp_threshold=OOD_MSP_THRESHOLD,
            min_margin=0.08,
        )

    stage_details.append(
        {
            "stage": "classification",
            "class_name": class_name,
            "confidence": round(confidence, 2),
            "confidence_margin": round(confidence_margin, 2),
            "entropy_bits": round(entropy_bits, 4),
        }
    )
    stage_details.append(
        {
            "stage": "safety_gate",
            "enabled": options["use_safety_gate"],
            "rejected": bool(safety.get("reject", False)),
            "reasons": safety.get("reasons", []),
        }
    )

    if safety["reject"]:
        guidance = "Upload a clear, close-up image of a single leaf under good lighting."
        if leaf_validation["reason"]:
            guidance = f"{leaf_validation['reason']} {guidance}"
        else:
            safety_reason = (
                ", ".join(safety["reasons"])
                if safety["reasons"]
                else "low trust score"
            )
            guidance = (
                f"Inference was rejected due to {safety_reason}. {guidance}"
            )

        parts = class_name.split("___")
        plant = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
        disease = parts[1].replace("_", " ") if len(parts) > 1 else class_name

        ret = {
            "class_name": class_name,
            "confidence": round(confidence, 2),
            "plant": plant,
            "disease": disease,
            "description": "The model produced a prediction, but the runtime safety gate flagged the image as low-trust or out-of-domain.",
            "symptoms": "A best-guess prediction is available, but the image quality checks failed.",
            "treatment": guidance,
            "prevention": "Use a plain background and make sure the leaf fills most of the frame.",
            "is_healthy": "healthy" in class_name.lower(),
            "is_valid_leaf": False,
            "rejected": True,
            "validation": {
                "leaf_score": leaf_validation["leaf_score"],
                "vegetation_ratio": leaf_validation["vegetation_ratio"],
                "confidence_margin": round(confidence_margin, 2),
                "entropy_bits": round(entropy_bits, 4),
                "uncertainty_score": int(safety["uncertainty_score"]),
                "rejection_reasons": safety["reasons"],
            },
            "pipeline": {
                "options": options,
                "stages": stage_details,
            },
        }
        if focus_overlay_b64:
            ret["cropped_image"] = f"data:image/jpeg;base64,{focus_overlay_b64}"
        return ret

    # Get disease info
    info = DISEASE_INFO.get(
        class_name,
        {
            "plant": class_name.split("___")[0]
            if "___" in class_name
            else "Unknown",
            "disease": class_name.split("___")[1]
            if "___" in class_name
            else class_name,
            "description": "Information not available for this disease.",
            "symptoms": "Please consult a plant pathologist for detailed symptoms.",
            "treatment": "Consult with local agricultural extension for treatment options.",
            "prevention": "Maintain good plant hygiene and regular monitoring.",
        },
    )

    ret = {
        "class_name": class_name,
        "confidence": round(confidence, 2),
        "plant": info["plant"],
        "disease": info["disease"],
        "description": info["description"],
        "symptoms": info["symptoms"],
        "treatment": info["treatment"],
        "prevention": info["prevention"],
        "is_healthy": "healthy" in class_name.lower(),
        "is_valid_leaf": True,
        "validation": {
            "leaf_score": leaf_validation["leaf_score"],
            "vegetation_ratio": leaf_validation["vegetation_ratio"],
            "confidence_margin": round(confidence_margin, 2),
            "entropy_bits": round(entropy_bits, 4),
            "rejection_reasons": [],
        },
        "pipeline": {
            "options": options,
            "stages": stage_details,
        },
    }
    if focus_overlay_b64:
        ret["cropped_image"] = f"data:image/jpeg;base64,{focus_overlay_b64}"
    return ret


def init_review_samples():
    """Scan the dataset/train folder, select the first image from each class,
    and copy it to annotations/samples/<class_name>.jpg.
    """
    import shutil

    from src.utils.config import TRAIN_DIR

    samples_dir = os.path.join("annotations", "samples")
    masks_dir = os.path.join("annotations", "masks")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    if not os.path.exists(TRAIN_DIR):
        print(f"[Warning] Training directory {TRAIN_DIR} does not exist.")
        return

    class_names = sorted(
        [entry.name for entry in os.scandir(TRAIN_DIR) if entry.is_dir()]
    )

    for class_name in class_names:
        sample_path = os.path.join(samples_dir, f"{class_name}.jpg")
        if not os.path.exists(sample_path):
            class_dir = os.path.join(TRAIN_DIR, class_name)
            images = sorted(
                [
                    entry.name
                    for entry in os.scandir(class_dir)
                    if entry.is_file()
                    and entry.name.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            if images:
                src_path = os.path.join(class_dir, images[0])
                shutil.copy(src_path, sample_path)
                print(f"Copied sample for {class_name}: {images[0]}")


@app.route("/review/classes", methods=["GET"])
def get_review_classes():
    init_review_samples()
    samples_dir = os.path.join("annotations", "samples")
    masks_dir = os.path.join("annotations", "masks")

    classes_status = []
    if os.path.exists(samples_dir):
        files = sorted(os.listdir(samples_dir))
        for filename in files:
            if filename.lower().endswith(".jpg"):
                class_name = filename[:-4]
                leaf_exists = os.path.exists(
                    os.path.join(masks_dir, f"{class_name}_leaf.png")
                )
                focus_exists = os.path.exists(
                    os.path.join(masks_dir, f"{class_name}_focus.png")
                )
                classes_status.append(
                    {
                        "class_name": class_name,
                        "annotated": leaf_exists or focus_exists,
                        "leaf_annotated": leaf_exists,
                        "focus_annotated": focus_exists,
                    }
                )
    return jsonify({"classes": classes_status})


@app.route("/review/sample/<class_name>")
def get_review_sample(class_name):
    from flask import send_from_directory

    samples_dir = os.path.abspath(os.path.join("annotations", "samples"))
    filename = f"{class_name}.jpg"
    if not os.path.exists(os.path.join(samples_dir, filename)):
        return jsonify({"error": "Sample image not found"}), 404
    return send_from_directory(samples_dir, filename)


def generate_default_leaf_mask(image_path):
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    leaf_mask = np.ones((h, w), dtype=bool)

    # If YOLOv26 leaf detection is active, run it to get leaf bounding box.
    if USE_YOLO_LEAF_DETECTION:
        detector = _get_yolo_leaf_detector()
        if detector is not None:
            try:
                detection = detector.detect(img)
                if detection["found"]:
                    x1, y1, x2, y2 = detection["bbox"]
                    yolo_mask = np.zeros((h, w), dtype=bool)
                    yolo_mask[y1:y2, x1:x2] = True
                    leaf_mask = yolo_mask
            except Exception as exc:
                print(f"YOLOv26 detect failed in default mask gen: {exc}")

    # Refine the mask using HSV color thresholding
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    bg_mask = (s <= 38) | (v <= 20) | (v >= 240)

    refined_mask = leaf_mask & (~bg_mask)
    if np.sum(refined_mask) > 100:
        leaf_mask = refined_mask

    # Create 4-channel transparent PNG
    mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Paint leaf green: BGR = (94, 197, 34), alpha = 255
    mask_rgba[leaf_mask, 0] = 94
    mask_rgba[leaf_mask, 1] = 197
    mask_rgba[leaf_mask, 2] = 34
    mask_rgba[leaf_mask, 3] = 255

    # Resize to 448x448 to match front-end canvas
    mask_rgba = cv2.resize(
        mask_rgba, (448, 448), interpolation=cv2.INTER_NEAREST
    )
    return mask_rgba


def generate_default_focus_mask(class_name, image_path):
    import cv2
    import numpy as np

    global model, class_indices, ACTIVE_BACKBONE

    img = cv2.imread(image_path)
    if img is None:
        return None

    # Attempt to use model Grad-CAM if available
    try:
        if model is None or class_indices is None:
            load_model_and_classes()

        if model is not None and class_indices is not None:
            from tensorflow.keras.preprocessing import image as keras_image

            from scripts.gradcam_check import (
                _find_target_layer,
                _make_gradcam_heatmap,
            )
            from src.core.preprocessing import preprocess_array_for_model

            # Load and preprocess image
            img_loaded = keras_image.load_img(
                image_path, target_size=(224, 224)
            )
            img_array = keras_image.img_to_array(img_loaded)
            img_array_exp = np.expand_dims(img_array, axis=0)

            if not _model_has_internal_preprocessing(model):
                img_array_exp = preprocess_array_for_model(
                    img_array_exp, backbone_name=ACTIVE_BACKBONE
                )

            # Find target class index
            pred_index = None
            for idx, name in class_indices.items():
                if name == class_name:
                    pred_index = idx
                    break

            # Resolve healthy partner for deviation calculation
            healthy_partner_idx = None
            from src.training.training_utils import parse_class_structure

            class_names_list = [
                class_indices[i] for i in sorted(class_indices.keys())
            ]
            partners = parse_class_structure(class_names_list)
            if pred_index is not None and pred_index < len(partners):
                healthy_partner_idx = partners[pred_index]

            target_layer_name = None
            if ACTIVE_BACKBONE != "DINOv3":
                target_layer_name = _find_target_layer(model)

            # Generate Grad-CAM heatmaps
            _, disease_heatmap = _make_gradcam_heatmap(
                model=model,
                img_array=img_array_exp,
                target_layer_name=target_layer_name,
                pred_index=pred_index,
                backbone_name=ACTIVE_BACKBONE or "DINOv3",
                vit_block_idx=6,
                healthy_partner_idx=healthy_partner_idx,
            )

            if disease_heatmap is not None and np.max(disease_heatmap) > 0:
                h_orig, w_orig = img.shape[:2]
                disease_heatmap_resized = cv2.resize(
                    disease_heatmap, (w_orig, h_orig)
                )

                # Threshold Grad-CAM heatmap at > 0.3 as candidate focus
                focus_mask = disease_heatmap_resized > 0.3

                mask_rgba = np.zeros((h_orig, w_orig, 4), dtype=np.uint8)
                # Paint focus red: BGR = (68, 68, 239), alpha = 255
                mask_rgba[focus_mask, 0] = 68
                mask_rgba[focus_mask, 1] = 68
                mask_rgba[focus_mask, 2] = 239
                mask_rgba[focus_mask, 3] = 255

                mask_rgba = cv2.resize(
                    mask_rgba, (448, 448), interpolation=cv2.INTER_NEAREST
                )
                return mask_rgba
    except Exception as e:
        print(f"Error generating model focus mask: {e}")

    # Fallback: segment center region of leaf mask
    try:
        leaf_mask_rgba = generate_default_leaf_mask(image_path)
        if leaf_mask_rgba is not None:
            mask_rgba = np.zeros((448, 448, 4), dtype=np.uint8)
            h_c, w_c = 224, 224
            for r in range(448):
                for c in range(448):
                    if leaf_mask_rgba[r, c, 3] == 255:
                        dist = np.sqrt((r - h_c) ** 2 + (c - w_c) ** 2)
                        if dist < 100:  # center circle of radius 100
                            mask_rgba[r, c, 0] = 68
                            mask_rgba[r, c, 1] = 68
                            mask_rgba[r, c, 2] = 239
                            mask_rgba[r, c, 3] = 255
            return mask_rgba
    except Exception:
        pass

    # Ultimate fallback: center circle on blank background
    mask_rgba = np.zeros((448, 448, 4), dtype=np.uint8)
    cv2.circle(mask_rgba, (224, 224), 80, (68, 68, 239, 255), -1)
    return mask_rgba


@app.route("/review/mask/<class_name>/<mask_type>")
def get_review_mask(class_name, mask_type):
    from flask import send_from_directory

    masks_dir = os.path.abspath(os.path.join("annotations", "masks"))
    os.makedirs(masks_dir, exist_ok=True)
    filename = f"{class_name}_{mask_type}.png"
    filepath = os.path.join(masks_dir, filename)

    if not os.path.exists(filepath):
        samples_dir = os.path.abspath(os.path.join("annotations", "samples"))
        sample_img_path = os.path.join(samples_dir, f"{class_name}.jpg")
        if os.path.exists(sample_img_path):
            import cv2

            if mask_type == "leaf":
                mask_data = generate_default_leaf_mask(sample_img_path)
            else:
                mask_data = generate_default_focus_mask(
                    class_name, sample_img_path
                )

            if mask_data is not None:
                cv2.imwrite(filepath, mask_data)

    if not os.path.exists(filepath):
        return jsonify({"error": "Mask not found"}), 404

    return send_from_directory(masks_dir, filename)


@app.route("/review/save", methods=["POST"])
def save_review_annotation():
    payload = request.get_json(silent=True) or {}
    class_name = payload.get("class_name")
    leaf_mask_b64 = payload.get("leaf_mask")
    focus_mask_b64 = payload.get("focus_mask")

    if not class_name:
        return jsonify({"error": "Missing class_name"}), 400

    masks_dir = os.path.join("annotations", "masks")
    os.makedirs(masks_dir, exist_ok=True)

    def decode_and_save(b64_str, filename):
        if not b64_str:
            return
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_data = base64.b64decode(b64_str)
        filepath = os.path.join(masks_dir, filename)
        with open(filepath, "wb") as f:
            f.write(img_data)

    try:
        if leaf_mask_b64:
            decode_and_save(leaf_mask_b64, f"{class_name}_leaf.png")
        if focus_mask_b64:
            decode_and_save(focus_mask_b64, f"{class_name}_focus.png")
        return jsonify({"message": f"Annotations saved for {class_name}."})
    except Exception as e:
        return jsonify({"error": f"Failed to save annotations: {str(e)}"}), 500


@app.route("/")
def index():
    """Render the main page"""
    available_model_names = sorted(
        [_model_option_name(path) for path in _list_available_model_paths()],
        key=lambda s: s.lower(),
    )
    active_name = (
        _model_option_name(ACTIVE_MODEL_PATH) if ACTIVE_MODEL_PATH else None
    )

    default_fraction_pct = 100.0
    env_fraction = _to_float(os.getenv("LEAF_TRAIN_DATA_FRACTION"), None)
    if env_fraction is not None:
        default_fraction_pct = max(0.1, min(100.0, env_fraction * 100.0))

    default_optimizer = (
        _normalize_train_optimizer(
            os.getenv("LEAF_TRAIN_OPTIMIZER") or TRAIN_OPTIMIZER_OPTIONS[0]
        )
        or TRAIN_OPTIMIZER_OPTIONS[0]
    )

    default_save_mode = _normalize_save_mode(os.getenv("LEAF_SAVE_MODE"))
    if default_save_mode is None:
        default_save_mode = TRAIN_SAVE_MODES[0]["value"]

    class_equalizer_env = os.getenv("LEAF_CLASS_EQUALIZER")
    default_class_equalizer = True
    if class_equalizer_env is not None:
        default_class_equalizer = _to_bool(class_equalizer_env)

    return render_template(
        "index.html",
        control_actions=CONTROL_ACTIONS,
        compute_info=get_compute_info(),
        available_models=available_model_names,
        active_model_name=active_name,
        training_backbones=sorted(TRAIN_BACKBONES, key=lambda s: s.lower()),
        training_optimizer_options=TRAIN_OPTIMIZER_OPTIONS,
        training_save_modes=TRAIN_SAVE_MODES,
        default_train_fraction_pct=round(default_fraction_pct, 2),
        default_training_optimizer=default_optimizer,
        default_training_save_mode=default_save_mode,
        default_class_equalizer=default_class_equalizer,
        default_leaf_detection_mode="auto",
        default_use_background_removal=True,
        default_use_safety_gate=True,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and prediction"""
    if model is None or class_indices is None:
        details = MODEL_LOAD_ERROR or (
            "Model artifacts are not loaded. Run training first or place a valid model in the models/ directory."
        )
        return jsonify({"error": details}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename_input = file.filename

    if not filename_input:
        return jsonify({"error": "No file selected"}), 400

    if not _is_allowed_upload(filename_input):
        return jsonify(
            {"error": "Unsupported file type. Use JPG, PNG, or WEBP."}
        ), 400

    selected_model_name = (request.form.get("model_name") or "").strip()
    pipeline_options = {
        "leaf_detection_mode": request.form.get("leaf_detection_mode", "auto"),
        "use_background_removal": request.form.get(
            "use_background_removal", "on"
        ),
        "use_safety_gate": request.form.get("use_safety_gate", "on"),
    }

    if file:
        # Save file temporarily
        original_name = secure_filename(filename_input)
        _, ext = os.path.splitext(original_name)
        filename = f"{uuid.uuid4().hex}{ext.lower()}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            inference_model, active_model_path = _get_inference_model(
                selected_model_name
            )

            # Make prediction
            result = predict_disease(
                filepath,
                inference_model=inference_model,
                pipeline_options=pipeline_options,
            )
            result["model_name"] = _model_option_name(active_model_path)

            if "cropped_image" in result:
                result["image"] = result["cropped_image"]
            else:
                # Read image for preview
                with open(filepath, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                result["image"] = f"data:image/jpeg;base64,{img_data}"

            return jsonify(result)

        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        except Exception as e:
            return jsonify({"error": str(e)}), 500

        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({"error": "Invalid file"}), 400


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None and class_indices is not None,
            "model_error": MODEL_LOAD_ERROR,
            "active_model": _model_option_name(ACTIVE_MODEL_PATH)
            if ACTIVE_MODEL_PATH
            else None,
            "available_models": sorted(
                [_model_option_name(path) for path in _list_available_model_paths()],
                key=lambda s: s.lower()
            ),
        }
    )


@app.route("/control/actions", methods=["GET"])
def control_actions():
    """Return available control panel actions."""
    actions = []
    for action_key, meta in CONTROL_ACTIONS.items():
        actions.append(
            {
                "action": action_key,
                "label": meta["label"],
                "description": meta["description"],
                "script": meta["script"],
            }
        )
    return jsonify({"actions": actions})


@app.route("/control/run/<action_key>", methods=["POST"])
def control_run(action_key):
    """Start a background workflow action."""
    if action_key not in CONTROL_ACTIONS:
        return jsonify({"error": "Unknown control action."}), 404

    with JOBS_LOCK:
        for job in JOBS.values():
            if job["action"] == action_key and job["status"] in {
                "starting",
                "running",
            }:
                return jsonify(
                    {
                        "error": "This action is already running.",
                        "job": _job_response(job),
                    }
                ), 409

    payload = request.get_json(silent=True) or {}
    archive_logs = _to_bool(payload.get("archive_logs"))
    base_model = (payload.get("base_model") or "").strip() if payload else ""
    train_fraction_percent = (
        payload.get("train_fraction_percent") if payload else None
    )
    optimizer = (payload.get("optimizer") or "").strip() if payload else ""
    save_mode = (payload.get("save_mode") or "").strip() if payload else ""
    class_equalizer = payload.get("class_equalizer") if payload else None
    must_review = payload.get("must_review") if payload else None

    if not payload and request.form:
        archive_logs = _to_bool(request.form.get("archive_logs"))
        base_model = (request.form.get("base_model") or "").strip()
        train_fraction_percent = request.form.get("train_fraction_percent")
        optimizer = (request.form.get("optimizer") or "").strip()
        save_mode = (request.form.get("save_mode") or "").strip()
        class_equalizer = request.form.get("class_equalizer")
        must_review = request.form.get("must_review")

    if base_model and action_key not in {"train"}:
        return jsonify(
            {
                "error": "Base model selection is only available for training actions."
            }
        ), 400

    if action_key == "train":
        if not base_model:
            base_model = TRAIN_BACKBONES[0] if TRAIN_BACKBONES else None
        if base_model and base_model not in TRAIN_BACKBONES:
            return jsonify(
                {
                    "error": "Unknown backbone selected for training.",
                    "available_backbones": TRAIN_BACKBONES,
                }
            ), 400

    train_options = None
    if action_key == "train":
        train_options = {}

        fraction_value = _to_float(train_fraction_percent, 100.0)
        if fraction_value is None or not (0.1 <= fraction_value <= 100.0):
            return jsonify(
                {
                    "error": "Training data percentage must be between 0.1 and 100."
                }
            ), 400
        train_options["train_fraction_pct"] = round(float(fraction_value), 2)
        train_options["train_fraction"] = float(fraction_value) / 100.0

        resolved_optimizer = _normalize_train_optimizer(
            optimizer or TRAIN_OPTIMIZER_OPTIONS[0]
        )
        if not resolved_optimizer:
            return jsonify(
                {
                    "error": "Unknown optimizer selected for training.",
                    "available_optimizers": TRAIN_OPTIMIZER_OPTIONS,
                }
            ), 400
        train_options["optimizer"] = resolved_optimizer

        resolved_save_mode = _normalize_save_mode(
            save_mode or TRAIN_SAVE_MODES[0]["value"]
        )
        if not resolved_save_mode:
            return jsonify(
                {
                    "error": "Unknown save mode selected for training.",
                    "available_save_modes": [
                        m["value"] for m in TRAIN_SAVE_MODES
                    ],
                }
            ), 400
        train_options["save_mode"] = resolved_save_mode

        if class_equalizer is None or class_equalizer == "":
            train_options["class_equalizer"] = True
        else:
            train_options["class_equalizer"] = _to_bool(class_equalizer)
        train_options["must_review"] = _to_bool(must_review)

    job = _create_job(
        action_key,
        archive_logs=archive_logs,
        base_model=base_model or None,
        train_options=train_options,
    )
    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()

    return jsonify(
        {"message": f"{job['label']} started.", "job": _job_response(job)}
    )


@app.route("/control/jobs", methods=["GET"])
def control_jobs():
    """List all workflow jobs."""
    with JOBS_LOCK:
        jobs = [_job_response(job) for job in JOBS.values()]
    jobs.sort(key=lambda x: x["start_time"] or 0, reverse=True)
    return jsonify({"jobs": jobs})


@app.route("/control/resume", methods=["POST"])
def control_resume():
    """Resume a paused training job."""
    try:
        os.makedirs("logs", exist_ok=True)
        flag_path = os.path.join("logs", "resume_epoch.flag")
        with open(flag_path, "w", encoding="utf-8") as f:
            f.write("resume")

        # Set job stage to resuming for UI feedback.
        with JOBS_LOCK:
            for job in JOBS.values():
                if (
                    job["status"] == "running"
                    and job.get("progress_stage") == "paused_for_review"
                ):
                    job["progress_stage"] = "resuming"

        return jsonify({"message": "Resume signal sent."})
    except Exception as e:
        return jsonify({"error": f"Failed to send resume signal: {e}"}), 500


@app.route("/control/system", methods=["GET"])
def control_system():
    """Return compute backend information for control panel status."""
    return jsonify({"compute": get_compute_info()})


@app.route("/control/stop/<job_id>", methods=["POST"])
def control_stop(job_id):
    """Stop a running workflow job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    process = job.get("process")
    if job["status"] not in {"starting", "running"} or process is None:
        return jsonify(
            {"error": "Job is not running.", "job": _job_response(job)}
        ), 409

    job["stop_requested"] = True
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    except Exception as exc:
        _append_job_log(job, f"Stop request failed: {exc}")

    return jsonify({"message": "Stop signal sent.", "job": _job_response(job)})


if __name__ == "__main__":
    try:
        init_review_samples()
        load_model_and_classes()
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        print(f"Model initialization skipped: {MODEL_LOAD_ERROR}")
        print(
            "The web app will still start, but prediction requests will return 503 until a model is available."
        )
    print("\nLeaf Disease Detection Web App")
    print("=" * 40)
    print("Open http://localhost:5000 in your browser")
    print("=" * 40)
    app.run(host="0.0.0.0", port=5000, debug=False)
