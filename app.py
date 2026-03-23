"""Flask web application for plant leaf disease detection and classification.

Serves a web interface for uploading leaf images and receiving disease
predictions with confidence scores, disease descriptions, treatment
recommendations, and prevention guidelines. Also provides a control panel
for triggering training, fine-tuning, evaluation, and figure generation jobs.
"""

import os
import json
import uuid
import sys
import time
import glob
import threading
import subprocess
import re
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import base64
from PIL import Image
from config import IMG_SIZE, FINAL_MODEL_PATH, CLASS_INDICES_PATH, MODELS_DIR, BASE_MODEL
from model_paths import resolve_keras_model_path
from hardware import configure_tensorflow, get_compute_info
from preprocessing import preprocess_array_for_model
from training_utils import WarmupCosineSchedule

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _log_tf_runtime_info():
    """Log TensorFlow build info and detected GPU devices at startup."""
    try:
        print(f"TensorFlow version: {tf.__version__}")
        print(f"CUDA visible devices: {tf.config.list_physical_devices('GPU')}")
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"Built with ROCm: {tf.test.is_built_with_rocm()}")
    except Exception as exc:  # Best effort; do not block app startup
        print(f"TensorFlow runtime probe failed: {exc}")


configure_tensorflow()
_log_tf_runtime_info()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variable
model = None
class_indices = None
MODEL_LOAD_ERROR = None
ACTIVE_MODEL_PATH = None
MODEL_CACHE = {}

# Keep complete per-job console history in memory for UI display.
# Set to an integer to enable truncation if needed in low-memory environments.
JOB_LOG_LIMIT = None
JOBS = {}
JOBS_LOCK = threading.Lock()
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
KERAS_BATCH_PROGRESS_RE = re.compile(r"^\d+/\d+\s+")

TRAIN_SCRIPT = 'model_training.py'
TRAIN_DESC = "Run baseline EfficientNet training pipeline."

CONTROL_ACTIONS = {
    "train": {
        "label": "Train Model",
        "script": TRAIN_SCRIPT,
        "description": TRAIN_DESC,
    },
    "fine_tune": {
        "label": "Fine Tune Model",
        "script": "model_fine_tuning.py",
        "description": "continue training from a saved checkpoint"
    },
    "evaluate": {
        "label": "Evaluate Model",
        "script": "model_evaluation.py",
        "description": "run validation and eval metrics"
    },
    "generate_figures": {
        "label": "Generate Figures",
        "script": "visualization_pipeline.py",
        "description": "build plots and analysis artifacts"
    }
}


def _resolve_model_path():
    return resolve_keras_model_path([FINAL_MODEL_PATH])


def _list_available_model_paths():
    candidates = []
    for path in glob.glob(os.path.join(MODELS_DIR, '*.keras')):
        if os.path.isfile(path):
            candidates.append(os.path.abspath(path))
    return sorted(candidates)


def _resolve_requested_model_path(model_name=None):
    default_path = _resolve_model_path()
    if not model_name:
        return os.path.abspath(default_path)

    model_name = str(model_name).strip()
    if not model_name:
        return os.path.abspath(default_path)

    available_paths = _list_available_model_paths()
    by_name = {os.path.basename(path): path for path in available_paths}
    if model_name in by_name:
        return by_name[model_name]

    raise ValueError(
        f"Unknown model '{model_name}'. Available: {', '.join(sorted(by_name.keys()))}"
    )


def _get_inference_model(model_name=None):
    global model, ACTIVE_MODEL_PATH

    target_path = _resolve_requested_model_path(model_name)
    if target_path in MODEL_CACHE:
        model = MODEL_CACHE[target_path]
        ACTIVE_MODEL_PATH = target_path
        return model, ACTIVE_MODEL_PATH

    loaded = load_model(
        target_path,
        custom_objects={"WarmupCosineSchedule": WarmupCosineSchedule}
    )
    MODEL_CACHE[target_path] = loaded
    model = loaded
    ACTIVE_MODEL_PATH = target_path
    return model, ACTIVE_MODEL_PATH


def _is_allowed_upload(filename):
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTENSIONS


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _create_job(action_key, archive_logs=False):
    action = CONTROL_ACTIONS[action_key]
    script = action["script"]
    command = [sys.executable, script]
    env_overrides = {
        'LEAF_SAVE_LOG_ARCHIVE': '1' if archive_logs else '0',
        'LEAF_SAVE_RUN_MANIFESTS': '1' if archive_logs else '0',
    }
    job_id = uuid.uuid4().hex
    now = time.time()
    job = {
        "id": job_id,
        "action": action_key,
        "label": action["label"],
        "description": action["description"],
        "script": script,
        "command": " ".join(command),
        "archive_logs": bool(archive_logs),
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
    cleaned = "".join(ch for ch in cleaned if ch == "\t" or 32 <= ord(ch) <= 126)
    cleaned = cleaned.strip()

    if not cleaned:
        return

    if job["logs"] and job["logs"][-1] == cleaned:
        return

    # Keras/TensorFlow batch progress emits carriage-return updates.
    # Keep a single live-updating progress line instead of appending a new line for each step.
    is_batch_progress = KERAS_BATCH_PROGRESS_RE.match(cleaned) is not None and "ms/step" in cleaned
    if is_batch_progress and job["logs"]:
        prev = job["logs"][-1]
        prev_is_batch = KERAS_BATCH_PROGRESS_RE.match(prev) is not None and "ms/step" in prev
        if prev_is_batch:
            job["logs"][-1] = cleaned
            return

    job["logs"].append(cleaned)
    if isinstance(JOB_LOG_LIMIT, int) and JOB_LOG_LIMIT > 0 and len(job["logs"]) > JOB_LOG_LIMIT:
        job["logs"] = job["logs"][-JOB_LOG_LIMIT:]


def _parse_progress_line(job, raw_line):
    prefix = "TRAINING_PROGRESS "
    line = (raw_line or "").strip()
    if not line.startswith(prefix):
        return False

    try:
        payload = json.loads(line[len(prefix):])
        progress = float(payload.get("progress_pct", job.get("progress_pct", 0.0)))
        eta = payload.get("eta_seconds")

        job["progress_pct"] = max(0.0, min(100.0, progress))
        job["eta_seconds"] = None if eta is None else max(0.0, float(eta))
        job["progress_stage"] = payload.get("stage", job.get("progress_stage", "running"))
        return True
    except Exception:
        return False


def _run_job(job):
    script_path = os.path.join(os.getcwd(), job["script"])
    command = [sys.executable, script_path]
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
        "prevention": "Plant resistant varieties. Rake and destroy fallen leaves. Prune trees for better airflow."
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "description": "A fungal disease caused by Botryosphaeria obtusa affecting fruit, leaves, and bark.",
        "symptoms": "Brown spots with concentric rings on leaves. Rotting fruit with black, mummified appearance.",
        "treatment": "Remove infected plant parts. Apply fungicides during growing season.",
        "prevention": "Maintain tree health. Remove mummified fruits. Prune dead wood."
    },
    "Apple___Brown spot": {
        "plant": "Apple",
        "disease": "Brown Spot",
        "description": "A fungal infection causing brown lesions on apple leaves.",
        "symptoms": "Brown circular spots on leaves, potentially leading to early defoliation.",
        "treatment": "Apply appropriate fungicides. Remove affected foliage.",
        "prevention": "Ensure good air circulation. Avoid overhead watering."
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "description": "A fungal disease requiring both apple and cedar/juniper trees to complete its life cycle.",
        "symptoms": "Yellow-orange spots on leaves with small black dots. Tube-like structures on leaf undersides.",
        "treatment": "Apply fungicides in spring. Remove nearby cedar/juniper trees if possible.",
        "prevention": "Plant resistant varieties. Remove galls from cedars before spring."
    },
    "Apple___Grey spot": {
        "plant": "Apple",
        "disease": "Grey Spot",
        "description": "A fungal disease affecting apple leaves causing grey lesions.",
        "symptoms": "Grey to brown spots on leaves, may cause premature leaf drop.",
        "treatment": "Apply fungicides. Improve orchard sanitation.",
        "prevention": "Remove fallen leaves. Ensure proper spacing between trees."
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "description": "This apple leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present. Leaf appears healthy with normal coloration.",
        "treatment": "No treatment needed. Continue regular plant care.",
        "prevention": "Maintain good growing conditions and regular monitoring."
    },
    "Apple___Mosaic": {
        "plant": "Apple",
        "disease": "Mosaic Virus",
        "description": "A viral disease causing mottled patterns on apple leaves.",
        "symptoms": "Yellow and green mosaic patterns on leaves. Stunted growth possible.",
        "treatment": "No cure for viral diseases. Remove infected plants to prevent spread.",
        "prevention": "Use virus-free planting material. Control insect vectors."
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "description": "This blueberry leaf shows no signs of disease.",
        "symptoms": "No disease symptoms. Healthy green coloration.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper soil pH and nutrition."
    },
    "Cherry___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "description": "This cherry leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular monitoring and proper care."
    },
    "Cherry___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "description": "A fungal disease causing white powdery coating on leaves.",
        "symptoms": "White powdery spots on leaves and shoots. Leaves may curl and distort.",
        "treatment": "Apply sulfur-based or systemic fungicides. Remove heavily infected parts.",
        "prevention": "Ensure good air circulation. Avoid overhead watering. Plant resistant varieties."
    },
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Gray Leaf Spot",
        "description": "A fungal disease caused by Cercospora zeae-maydis.",
        "symptoms": "Rectangular gray to tan lesions on leaves running parallel to veins.",
        "treatment": "Apply foliar fungicides. Use resistant hybrids.",
        "prevention": "Rotate crops. Till under crop residue. Plant resistant varieties."
    },
    "Corn___Common_rust": {
        "plant": "Corn",
        "disease": "Common Rust",
        "description": "A fungal disease caused by Puccinia sorghi.",
        "symptoms": "Small, circular to elongated brown pustules on both leaf surfaces.",
        "treatment": "Apply fungicides if severe. Usually not economically damaging.",
        "prevention": "Plant resistant hybrids. Early planting can help avoid infection."
    },
    "Corn___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "description": "This corn leaf shows no signs of disease.",
        "symptoms": "No disease symptoms. Normal green coloration.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper nutrition and irrigation."
    },
    "Corn___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "description": "A fungal disease caused by Exserohilum turcicum.",
        "symptoms": "Long, elliptical gray-green to tan lesions on leaves.",
        "treatment": "Apply foliar fungicides. Remove crop debris.",
        "prevention": "Use resistant hybrids. Rotate crops. Till under residue."
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "description": "A fungal disease caused by Guignardia bidwellii.",
        "symptoms": "Brown circular spots on leaves. Fruit shrivels and turns black (mummies).",
        "treatment": "Apply fungicides from bud break. Remove mummified fruit.",
        "prevention": "Prune for good air circulation. Remove infected plant material."
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "description": "A complex fungal disease affecting grapevines.",
        "symptoms": "Tiger-stripe pattern on leaves. Dark spots on berries. Sudden vine collapse.",
        "treatment": "No effective treatment. Remove severely affected vines.",
        "prevention": "Avoid large pruning wounds. Paint pruning cuts with fungicide."
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "description": "This grape leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular monitoring and proper vineyard management."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight (Isariopsis)",
        "description": "A fungal disease causing leaf spots on grapevines.",
        "symptoms": "Irregular brown spots on leaves with dark borders.",
        "treatment": "Apply fungicides. Remove infected leaves.",
        "prevention": "Improve air circulation. Avoid overhead irrigation."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing (Citrus Greening)",
        "description": "A devastating bacterial disease spread by psyllid insects.",
        "symptoms": "Yellowing of leaves in blotchy pattern. Misshapen, bitter fruit. Tree decline.",
        "treatment": "No cure. Remove infected trees. Control psyllid vectors.",
        "prevention": "Use disease-free nursery stock. Control Asian citrus psyllid."
    },
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease caused by Xanthomonas campestris.",
        "symptoms": "Small, dark spots on leaves that may fall out. Fruit has sunken spots.",
        "treatment": "Apply copper-based bactericides. Remove infected parts.",
        "prevention": "Plant resistant varieties. Avoid overhead irrigation."
    },
    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "description": "This peach leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular care and monitoring."
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease affecting pepper plants.",
        "symptoms": "Small, dark, water-soaked spots on leaves. Raised spots on fruit.",
        "treatment": "Apply copper-based sprays. Remove infected plants.",
        "prevention": "Use disease-free seeds. Rotate crops. Avoid overhead watering."
    },
    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "Healthy",
        "description": "This pepper leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain good growing conditions."
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "description": "A fungal disease caused by Alternaria solani.",
        "symptoms": "Dark brown spots with concentric rings (target-like) on older leaves.",
        "treatment": "Apply fungicides. Remove infected leaves.",
        "prevention": "Rotate crops. Use certified seed. Maintain plant vigor."
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "description": "This potato leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular monitoring and proper irrigation."
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "description": "A devastating disease caused by Phytophthora infestans (caused Irish Potato Famine).",
        "symptoms": "Water-soaked spots that turn brown. White mold on leaf undersides. Rapid plant death.",
        "treatment": "Apply fungicides immediately. Remove infected plants.",
        "prevention": "Use resistant varieties. Avoid overhead irrigation. Destroy infected tubers."
    },
    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "description": "This raspberry leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Good pruning and air circulation."
    },
    "Rice___Brown_Spot": {
        "plant": "Rice",
        "disease": "Brown Spot",
        "description": "A fungal disease caused by Bipolaris oryzae.",
        "symptoms": "Oval brown spots on leaves with gray centers.",
        "treatment": "Apply fungicides. Improve soil fertility.",
        "prevention": "Use resistant varieties. Balanced fertilization."
    },
    "Rice___Healthy": {
        "plant": "Rice",
        "disease": "Healthy",
        "description": "This rice leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper water and nutrient management."
    },
    "Rice___Leaf_Blast": {
        "plant": "Rice",
        "disease": "Leaf Blast",
        "description": "A serious fungal disease caused by Magnaporthe oryzae.",
        "symptoms": "Diamond-shaped spots with gray centers and brown borders.",
        "treatment": "Apply systemic fungicides. Drain fields if possible.",
        "prevention": "Use resistant varieties. Avoid excess nitrogen."
    },
    "Rice___Neck_Blast": {
        "plant": "Rice",
        "disease": "Neck Blast",
        "description": "A severe form of rice blast affecting the panicle neck.",
        "symptoms": "Brown to black lesions on panicle neck. Panicle may break and fall.",
        "treatment": "Apply fungicides before heading. Remove infected panicles.",
        "prevention": "Plant resistant varieties. Balanced fertilization."
    },
    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "description": "This soybean leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Crop rotation and proper spacing."
    },
    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "description": "A common fungal disease affecting cucurbits.",
        "symptoms": "White powdery patches on leaves. Leaves may yellow and die.",
        "treatment": "Apply fungicides or baking soda solution. Remove infected leaves.",
        "prevention": "Plant resistant varieties. Ensure good air circulation."
    },
    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "description": "This strawberry leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Proper spacing and mulching."
    },
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "description": "A fungal disease caused by Diplocarpon earlianum.",
        "symptoms": "Irregular purple spots that merge. Leaf margins appear burned.",
        "treatment": "Apply fungicides. Remove infected leaves.",
        "prevention": "Use resistant varieties. Renovate beds after harvest."
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease affecting tomato plants.",
        "symptoms": "Small, dark, water-soaked spots on leaves. Raised spots on fruit.",
        "treatment": "Apply copper-based bactericides. Remove infected parts.",
        "prevention": "Use disease-free seeds. Rotate crops. Avoid overhead watering."
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "description": "A fungal disease caused by Alternaria solani.",
        "symptoms": "Dark brown spots with concentric rings on lower leaves first.",
        "treatment": "Apply fungicides. Remove infected leaves. Mulch around plants.",
        "prevention": "Rotate crops. Stake plants. Water at base of plants."
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "description": "This tomato leaf shows no signs of disease.",
        "symptoms": "No disease symptoms present.",
        "treatment": "No treatment needed.",
        "prevention": "Regular monitoring and proper care."
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "description": "A destructive disease caused by Phytophthora infestans.",
        "symptoms": "Large, irregular brown spots. White mold on undersides. Rapid spread.",
        "treatment": "Apply fungicides immediately. Remove infected plants.",
        "prevention": "Use resistant varieties. Improve air circulation. Avoid wet foliage."
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "description": "A fungal disease caused by Passalora fulva.",
        "symptoms": "Pale green to yellow spots on upper leaf surface. Olive-brown mold below.",
        "treatment": "Improve ventilation. Apply fungicides. Remove infected leaves.",
        "prevention": "Reduce humidity. Space plants properly. Use resistant varieties."
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "description": "A common fungal disease caused by Septoria lycopersici.",
        "symptoms": "Small, circular spots with dark borders and gray centers with black dots.",
        "treatment": "Apply fungicides. Remove infected lower leaves.",
        "prevention": "Rotate crops. Mulch around plants. Avoid overhead watering."
    },
    "Tomato___Spider_mites_Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites",
        "description": "Tiny arachnid pests that feed on plant cells.",
        "symptoms": "Stippled, yellowing leaves. Fine webbing on undersides. Leaf drop.",
        "treatment": "Spray with water or insecticidal soap. Use miticides if severe.",
        "prevention": "Maintain plant health. Avoid dusty conditions. Introduce predatory mites."
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "description": "A fungal disease caused by Corynespora cassiicola.",
        "symptoms": "Brown spots with concentric rings giving target-like appearance.",
        "treatment": "Apply fungicides. Remove infected leaves.",
        "prevention": "Improve air circulation. Avoid overhead irrigation."
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "description": "A highly contagious viral disease.",
        "symptoms": "Mottled light and dark green pattern on leaves. Distorted growth.",
        "treatment": "No cure. Remove and destroy infected plants.",
        "prevention": "Use virus-free seeds. Disinfect tools. Wash hands before handling."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Yellow Leaf Curl Virus",
        "description": "A devastating viral disease spread by whiteflies.",
        "symptoms": "Upward curling of leaves. Yellowing. Stunted growth. Reduced fruit.",
        "treatment": "No cure. Remove infected plants. Control whiteflies.",
        "prevention": "Use resistant varieties. Control whitefly populations. Use reflective mulches."
    },
    "Wheat brown spot disease": {
        "plant": "Wheat",
        "disease": "Brown Spot",
        "description": "A fungal disease affecting wheat leaves.",
        "symptoms": "Brown oval spots on leaves that may merge.",
        "treatment": "Apply fungicides. Remove crop residue.",
        "prevention": "Use resistant varieties. Crop rotation."
    }
}


def load_model_and_classes():
    """Load the model and class indices"""
    global model, class_indices, MODEL_LOAD_ERROR, ACTIVE_MODEL_PATH

    model = None
    class_indices = None
    MODEL_LOAD_ERROR = None
    ACTIVE_MODEL_PATH = None

    print("Loading model...")
    model, model_path = _get_inference_model()
    print(f"Loaded model file: {model_path}")

    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)

    # Reverse the indices to get class names from predictions
    class_indices = {v: k for k, v in class_indices.items()}
    print("Model loaded successfully.")


def predict_disease(img_path, inference_model=None):
    """Predict disease from image"""
    active_model = inference_model or model
    if active_model is None or class_indices is None:
        raise RuntimeError("Model or class indices are not loaded.")

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_array_for_model(img_array)
    
    # Make prediction
    predictions = active_model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx]) * 100

    # Top-2 margin helps detect ambiguous predictions.
    top_two = np.argsort(predictions[0])[-2:]
    top1_prob = float(predictions[0][top_two[-1]])
    top2_prob = float(predictions[0][top_two[-2]])
    confidence_margin = (top1_prob - top2_prob) * 100
    
    # Get class name
    class_name = class_indices.get(predicted_class_idx, "Unknown")

    leaf_validation = assess_leaf_likelihood(img_path)
    low_confidence = confidence < 45.0
    low_margin = confidence_margin < 12.0

    appears_non_leaf = (
        leaf_validation["leaf_score"] < 0.23
        and leaf_validation["vegetation_ratio"] < 0.08
    )

    weak_leaf_signal = leaf_validation["leaf_score"] < 0.35
    uncertain_prediction = low_confidence or low_margin

    if appears_non_leaf or (uncertain_prediction and weak_leaf_signal):
        guidance = "Upload a clear, close-up image of a single leaf under good lighting."
        if leaf_validation["reason"]:
            guidance = f"{leaf_validation['reason']} {guidance}"

        return {
            "class_name": "Unknown",
            "confidence": round(confidence, 2),
            "plant": "Unknown",
            "disease": "Not a valid leaf image",
            "description": "The uploaded photo does not appear to be a plant leaf, or the model is not confident enough to classify it safely.",
            "symptoms": "No disease analysis was performed because leaf validation failed.",
            "treatment": guidance,
            "prevention": "Use a plain background and make sure the leaf fills most of the frame.",
            "is_healthy": False,
            "is_valid_leaf": False,
            "validation": {
                "leaf_score": leaf_validation["leaf_score"],
                "vegetation_ratio": leaf_validation["vegetation_ratio"],
                "confidence_margin": round(confidence_margin, 2),
            },
        }
    
    # Get disease info
    info = DISEASE_INFO.get(class_name, {
        "plant": class_name.split("___")[0] if "___" in class_name else "Unknown",
        "disease": class_name.split("___")[1] if "___" in class_name else class_name,
        "description": "Information not available for this disease.",
        "symptoms": "Please consult a plant pathologist for detailed symptoms.",
        "treatment": "Consult with local agricultural extension for treatment options.",
        "prevention": "Maintain good plant hygiene and regular monitoring."
    })
    
    return {
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
        },
    }


def assess_leaf_likelihood(img_path):
    """Heuristic leaf plausibility check to reject obvious non-leaf uploads."""
    try:
        with Image.open(img_path) as img:
            arr = np.asarray(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)

        if arr.size == 0:
            return {
                "leaf_score": 0.0,
                "vegetation_ratio": 0.0,
                "reason": "Image content could not be analyzed.",
            }

        arr_norm = arr / 255.0
        maxc = np.max(arr_norm, axis=2)
        minc = np.min(arr_norm, axis=2)
        delta = np.maximum(maxc - minc, 1e-8)

        hue = np.zeros_like(maxc)
        r = arr_norm[:, :, 0]
        g = arr_norm[:, :, 1]
        b = arr_norm[:, :, 2]

        r_mask = maxc == r
        g_mask = maxc == g
        b_mask = maxc == b

        hue[r_mask] = (60.0 * ((g[r_mask] - b[r_mask]) / delta[r_mask]) + 360.0) % 360.0
        hue[g_mask] = (60.0 * ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 120.0) % 360.0
        hue[b_mask] = (60.0 * ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 240.0) % 360.0

        sat = np.where(maxc <= 0.0, 0.0, delta / maxc)
        val = maxc

        vegetation_mask = (hue >= 20.0) & (hue <= 140.0) & (sat >= 0.15) & (val >= 0.15)
        vegetation_ratio = float(np.mean(vegetation_mask))
        contrast = float(np.std(arr_norm))

        leaf_score = min(1.0, vegetation_ratio * 1.7 + contrast * 0.6)

        reason = ""
        if vegetation_ratio < 0.08:
            reason = "Very little leaf-like color/texture was detected."
        elif leaf_score < 0.35:
            reason = "Leaf signal is weak in this image."

        return {
            "leaf_score": round(leaf_score, 3),
            "vegetation_ratio": round(vegetation_ratio, 3),
            "reason": reason,
        }
    except Exception as exc:
        return {
            "leaf_score": 0.0,
            "vegetation_ratio": 0.0,
            "reason": f"Image validation failed: {exc}",
        }


@app.route('/')
def index():
    """Render the main page"""
    available_model_names = [os.path.basename(path) for path in _list_available_model_paths()]
    active_name = os.path.basename(ACTIVE_MODEL_PATH) if ACTIVE_MODEL_PATH else None
    return render_template(
        'index.html',
        control_actions=CONTROL_ACTIONS,
        compute_info=get_compute_info(),
        available_models=available_model_names,
        active_model_name=active_name,
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if model is None or class_indices is None:
        details = MODEL_LOAD_ERROR or (
            "Model artifacts are not loaded. Run training first or place a valid model in the models/ directory."
        )
        return jsonify({'error': details}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    filename_input = file.filename

    if not filename_input:
        return jsonify({'error': 'No file selected'}), 400

    if not _is_allowed_upload(filename_input):
        return jsonify({'error': 'Unsupported file type. Use JPG, PNG, or WEBP.'}), 400

    selected_model_name = (request.form.get('model_name') or '').strip()
    
    if file:
        # Save file temporarily
        original_name = secure_filename(filename_input)
        _, ext = os.path.splitext(original_name)
        filename = f"{uuid.uuid4().hex}{ext.lower()}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            inference_model, active_model_path = _get_inference_model(selected_model_name)

            # Make prediction
            result = predict_disease(filepath, inference_model=inference_model)
            result['model_name'] = os.path.basename(active_model_path)
            
            # Read image for preview
            with open(filepath, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            result['image'] = f"data:image/jpeg;base64,{img_data}"
            
            return jsonify(result)
        
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Invalid file'}), 400


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and class_indices is not None,
        'model_error': MODEL_LOAD_ERROR,
        'active_model': os.path.basename(ACTIVE_MODEL_PATH) if ACTIVE_MODEL_PATH else None,
        'available_models': [os.path.basename(path) for path in _list_available_model_paths()],
    })


@app.route('/control/actions', methods=['GET'])
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


@app.route('/control/run/<action_key>', methods=['POST'])
def control_run(action_key):
    """Start a background workflow action."""
    if action_key not in CONTROL_ACTIONS:
        return jsonify({"error": "Unknown control action."}), 404

    with JOBS_LOCK:
        for job in JOBS.values():
            if job["action"] == action_key and job["status"] in {"starting", "running"}:
                return jsonify({"error": "This action is already running.", "job": _job_response(job)}), 409

    payload = request.get_json(silent=True) or {}
    archive_logs = _to_bool(payload.get('archive_logs'))
    if not payload and request.form:
        archive_logs = _to_bool(request.form.get('archive_logs'))

    job = _create_job(action_key, archive_logs=archive_logs)
    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()

    return jsonify({"message": f"{job['label']} started.", "job": _job_response(job)})


@app.route('/control/jobs', methods=['GET'])
def control_jobs():
    """List all workflow jobs."""
    with JOBS_LOCK:
        jobs = [_job_response(job) for job in JOBS.values()]
    jobs.sort(key=lambda x: x["start_time"] or 0, reverse=True)
    return jsonify({"jobs": jobs})


@app.route('/control/system', methods=['GET'])
def control_system():
    """Return compute backend information for control panel status."""
    return jsonify({"compute": get_compute_info()})


@app.route('/control/stop/<job_id>', methods=['POST'])
def control_stop(job_id):
    """Stop a running workflow job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    process = job.get("process")
    if job["status"] not in {"starting", "running"} or process is None:
        return jsonify({"error": "Job is not running.", "job": _job_response(job)}), 409

    job["stop_requested"] = True
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    except Exception as exc:
        _append_job_log(job, f"Stop request failed: {exc}")

    return jsonify({"message": "Stop signal sent.", "job": _job_response(job)})


if __name__ == '__main__':
    try:
        load_model_and_classes()
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        print(f"Model initialization skipped: {MODEL_LOAD_ERROR}")
        print("The web app will still start, but prediction requests will return 503 until a model is available.")
    print("\nLeaf Disease Detection Web App")
    print("=" * 40)
    print("Open http://localhost:5000 in your browser")
    print("=" * 40)
    app.run(host='0.0.0.0', port=5000, debug=False)
