# Leaf Disease Detection Workflow

This document describes the full project workflow from local setup through
training, YOLO-guided focus alignment, evaluation, inference, and web serving.
The classifier always receives the original RGB image pixels. YOLO detections
are used only for focus masks, saliency regularization, and visual overlays.

## End-to-End System Diagram

```mermaid
flowchart TD
    %% =========================
    %% Inputs and environment
    %% =========================
    subgraph ENV["Environment and Configuration"]
        OS["Windows host + WSL Bash runtime"]
        UV["uv dependency and command runner"]
        PY["Python 3.13 virtual environment"]
        CFG["src/utils/config.py<br/>paths, flags, thresholds, image size"]
        HW["src/utils/hardware.py<br/>TensorFlow GPU setup, memory growth, mixed precision"]
        CLI["src/main.py<br/>serve | train | fine_tune | refine | evaluate | visualize"]
        OS --> UV --> PY
        PY --> CFG
        CFG --> HW
        CLI --> CFG
    end

    %% =========================
    %% Dataset and labels
    %% =========================
    subgraph DATA["Dataset and Label Sources"]
        RAW["PlantVillage-style image corpus<br/>Mendeley 32vfdrj76m/1"]
        SPLITS["dataset/train<br/>dataset/val<br/>dataset/test"]
        CLASSDIRS["Class folders<br/>Plant___disease_or_healthy"]
        CLASSIDX["models/class_indices.json<br/>index to class-name mapping"]
        MANIFESTS["Optional reports/manifests<br/>dedupe, leakage checks, split summaries"]
        RAW --> SPLITS --> CLASSDIRS
        CLASSDIRS --> CLASSIDX
        SPLITS --> MANIFESTS
    end

    %% =========================
    %% Optional YOLO detector training
    %% =========================
    subgraph YOLOTRAIN["Optional YOLO Leaf Focus Detector Training"]
        SEG["src/core/leaf_segmentation.py<br/>color/contour leaf localization"]
        AUTOBOX["Auto-generated YOLO bbox labels<br/>largest valid leaf contour"]
        YOLODATA["dataset/yolo_dataset<br/>images/train, images/val<br/>labels/train, labels/val<br/>leaf_data.yaml"]
        YOLOFIT["src/training/train_yolo_leaf_detector.py<br/>fine-tune YOLOv26n"]
        YOLOWEIGHTS["models/yolo26_leaf_detector.pt"]
        CLASSDIRS --> SEG --> AUTOBOX --> YOLODATA --> YOLOFIT --> YOLOWEIGHTS
    end

    %% =========================
    %% Model construction
    %% =========================
    subgraph MODEL["Classifier Model Construction"]
        BACKBONES["src/core/backbones.py<br/>EfficientNetV2 variants, DINOv3/ViT"]
        PREPROC["src/core/preprocessing.py<br/>backbone-specific preprocessing"]
        HEAD["Multi-output classifier head<br/>crop/family auxiliary output + disease output"]
        LOSS["src/training/training_utils.py<br/>categorical/hierarchical/focal loss<br/>class weighting and schedules"]
        BASEMODEL["Functional Keras classifier"]
        CFG --> BACKBONES
        BACKBONES --> PREPROC --> HEAD --> BASEMODEL
        LOSS --> BASEMODEL
    end

    %% =========================
    %% Training input pipeline
    %% =========================
    subgraph TRAINPIPE["Training Data Pipeline"]
        FILESCAN["Scan class image paths<br/>collect_dataset_files / directory loaders"]
        DECODE["Decode original RGB image<br/>resize to IMG_SIZE"]
        YOLOFOCUS["src/core/yolo_leaf.py<br/>YOLOLeafDetector.detect()<br/>YOLOLeafDetector.get_focus_mask()"]
        FOCUSMASK["Binary focus mask<br/>shape H x W x 1<br/>1 inside bbox, 0 outside<br/>all-ones fallback if no detection"]
        TFDATA["tf.data.Dataset sample<br/>((image_tensor, yolo_mask), label_targets)"]
        AUG["Training augmentations<br/>MixUp, CutMix, RandAugment, color jitter,<br/>blur/noise/erasing/random resized crop"]
        LABELS["One-hot disease labels<br/>optional crop/family auxiliary labels"]
        CLASSDIRS --> FILESCAN --> DECODE
        YOLOWEIGHTS -.->|if enabled| YOLOFOCUS
        DECODE --> YOLOFOCUS --> FOCUSMASK
        DECODE --> TFDATA
        FOCUSMASK --> TFDATA
        LABELS --> TFDATA --> AUG
    end

    %% =========================
    %% Saliency alignment
    %% =========================
    subgraph ALIGN["YOLO-Guided Saliency Alignment"]
        WRAP["src/core/saliency_alignment.py<br/>SaliencyAlignedModel"]
        FORWARD["call((image, yolo_mask))<br/>passes image only into classifier"]
        MASKROUTE["train_step unpacks yolo_mask<br/>leaf_mask = yolo_mask<br/>bg_mask = 1 - yolo_mask<br/>HSV fallback only if mask missing/invalid"]
        GRADCAM["Grad-CAM / ViT attention extraction<br/>target convolution or ViT blocks"]
        PENALTY["Alignment losses<br/>penalize attention outside focus region<br/>sparsity penalty<br/>disease-focus reward"]
        TOTALLOSS["Total training objective<br/>classification + weighted saliency penalties"]
        AUG --> WRAP --> FORWARD --> BASEMODEL
        WRAP --> MASKROUTE --> GRADCAM --> PENALTY --> TOTALLOSS
        BASEMODEL --> TOTALLOSS
    end

    %% =========================
    %% Training phases and artifacts
    %% =========================
    subgraph TRAINING["Training, Fine-Tuning, and Refinement"]
        TRAIN["src/training/train_model.py<br/>Phase 1 frozen-backbone warm-up<br/>Phase 2 unfrozen fine-tuning"]
        CKPT["models/leaf_disease_checkpoint.keras<br/>best checkpoint callbacks"]
        FINETUNE["src/training/fine_tune_model.py<br/>continue training from checkpoint"]
        CLASSIFIER["models/leaf_disease_classifier.keras"]
        REFINE["src/training/refine_model.py<br/>rolling pre-overfit restoration<br/>strict validation monitoring"]
        REFINED["models/leaf_disease_refined.keras<br/>deployment-preferred model"]
        LOGS["logs/ and CSV histories<br/>progress events, review flags, TensorBoard"]
        TOTALLOSS --> TRAIN --> CKPT
        TRAIN --> LOGS
        CKPT --> FINETUNE --> CLASSIFIER
        CLASSIFIER --> REFINE --> REFINED
        REFINE --> LOGS
    end

    %% =========================
    %% Evaluation
    %% =========================
    subgraph EVAL["Evaluation and Publication Artifacts"]
        EVALSCRIPT["src/evaluation/evaluate_model.py"]
        METRICS["Accuracy, macro precision/recall/F1<br/>confusion matrix, per-class metrics"]
        CALIB["evaluation/calibration.py<br/>ECE, reliability, temperature scaling"]
        ROBUST["evaluation/robustness.py<br/>blur, noise, brightness/contrast stress tests"]
        OOD["Safety diagnostics<br/>confidence, entropy, MSP/OOD-style checks"]
        REPORTS["reports/*.json and reports/*.md"]
        PLOTS["plots/ and zzplots/<backbone>/<artifact>.png"]
        REFINED --> EVALSCRIPT
        CLASSIFIER --> EVALSCRIPT
        EVALSCRIPT --> METRICS --> REPORTS
        EVALSCRIPT --> CALIB --> REPORTS
        EVALSCRIPT --> ROBUST --> PLOTS
        EVALSCRIPT --> OOD --> REPORTS
    end

    %% =========================
    %% Inference
    %% =========================
    subgraph INFER["Inference Pipeline"]
        USERIMG["User image<br/>CLI path or web upload"]
        LEAFGATE["Stage 1 leaf validation<br/>trained leaf detector if available<br/>heuristic fallback"]
        FOCUSONLY["Stage 2 focus detection<br/>original RGB image is preserved<br/>YOLO bbox is metadata only"]
        LOADIMG["Resize original RGB to IMG_SIZE<br/>no cropping, no masking, no background removal"]
        MODELLOAD["resolve_keras_model_path()<br/>load Keras model or ensemble"]
        INFPRE["Backbone-aware preprocessing<br/>skip if model includes internal preprocessing"]
        PROBS["model.predict()<br/>class probability vector"]
        DIAG["src/core/inference_guard.py<br/>confidence, margin, entropy, MSP diagnostics"]
        SAFETY{"Safety gate enabled<br/>and thresholds passed?"}
        ACCEPT["Accepted prediction<br/>class, confidence, plant/disease info"]
        REJECT["Rejected or best-guess response<br/>reason list and diagnostics"]
        USERIMG --> LEAFGATE
        LEAFGATE --> FOCUSONLY
        FOCUSONLY --> LOADIMG
        MODELLOAD --> INFPRE
        LOADIMG --> INFPRE --> PROBS --> DIAG --> SAFETY
        SAFETY -- yes --> ACCEPT
        SAFETY -- no --> REJECT
    end

    %% =========================
    %% Web serving and review
    %% =========================
    subgraph WEB["Flask Web UI and Control Plane"]
        APP["src/web/app.py"]
        UPLOAD["/predict upload endpoint<br/>secure filename + temporary upload"]
        OVERLAY["Green YOLO focus overlay<br/>returned as cropped_image compatibility field<br/>not used as model input"]
        HEALTH["/health endpoint<br/>model status and options"]
        CONTROL["Training/evaluation job endpoints<br/>launch subprocesses through uv/python scripts"]
        REVIEW["Manual annotation review UI<br/>leaf mask and focus mask overlays<br/>annotations/masks/*.png"]
        JOBS["In-memory job log store<br/>progress polling and optional archive logs"]
        APP --> UPLOAD --> USERIMG
        FOCUSONLY --> OVERLAY --> UPLOAD
        APP --> HEALTH
        APP --> CONTROL --> CLI
        CONTROL --> JOBS
        APP --> REVIEW
    end

    %% =========================
    %% Figure generation
    %% =========================
    subgraph FIGURES["Figure and Report Generation"]
        FIGSCRIPT["scripts/generate_figures.py<br/>scripts/generate_publication_figures.py<br/>specialized figure scripts"]
        TABLES["tools/reporting/generate_report_tables.py"]
        VISOUT["Publication-ready plots and tables"]
        REPORTS --> FIGSCRIPT
        PLOTS --> FIGSCRIPT
        REPORTS --> TABLES
        FIGSCRIPT --> VISOUT
        TABLES --> VISOUT
    end

    %% =========================
    %% Feedback loops
    %% =========================
    REPORTS --> DECISION{"Metrics, calibration,<br/>robustness, and safety acceptable?"}
    PLOTS --> DECISION
    DECISION -->|yes| APP
    DECISION -->|no: tune config/backbone/augmentation| CFG
    DECISION -->|no: improve focus detector| YOLOTRAIN
```

## Training Sequence

```mermaid
sequenceDiagram
    autonumber
    participant U as Engineer / UI Control Panel
    participant CLI as src/main.py or training script
    participant CFG as config.py
    participant DS as Dataset Loader
    participant YOLO as YOLOLeafDetector
    participant SAL as SaliencyAlignedModel
    participant M as Keras Classifier
    participant CB as Callbacks
    participant ART as Model Artifacts

    U->>CLI: Start train / fine_tune / refine
    CLI->>CFG: Read paths, backbone, flags, thresholds
    CLI->>DS: Scan dataset/train and dataset/val
    loop each image
        DS->>DS: Decode original RGB and resize
        alt YOLO focus enabled
            DS->>YOLO: detect original image
            YOLO-->>DS: bbox focus mask or all-ones fallback
        else YOLO disabled
            DS-->>DS: all-ones focus mask
        end
        DS-->>CLI: ((image_tensor, yolo_mask), labels)
    end
    CLI->>M: Build backbone + classification head
    CLI->>SAL: Wrap model when attention guidance is enabled
    loop training batches
        SAL->>M: Forward image tensor only
        SAL->>SAL: Compute Grad-CAM / attention maps
        SAL->>SAL: Penalize saliency outside YOLO focus mask
        SAL-->>M: Apply gradients from total loss
        CB->>CB: Track metrics, early stop, checkpoint
    end
    CB->>ART: Save checkpoint/classifier/refined model
```

## Inference Sequence

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant W as Flask UI or CLI
    participant L as Leaf Validation
    participant Y as YOLO Focus Detector
    participant P as Preprocessing
    participant M as Keras Model
    participant G as Inference Guard
    participant R as Response

    U->>W: Submit image
    W->>L: Validate leaf presence
    alt invalid leaf
        L-->>R: Reject with reason
        R-->>U: Not a valid leaf response
    else valid leaf
        L-->>W: leaf_score and vegetation diagnostics
        opt focus overlay enabled
            W->>Y: Detect bbox on original upload
            Y-->>W: bbox and confidence
            W->>W: Draw green overlay for review only
        end
        W->>P: Load original RGB and resize
        P->>M: Send preprocessed original image
        M-->>P: Class probabilities
        P->>G: Confidence, margin, entropy, MSP
        alt safety gate rejects
            G-->>R: Best guess + rejection reasons
        else accepted
            G-->>R: Accepted class and confidence
        end
        R-->>U: Prediction, diagnostics, optional focus overlay
    end
```

## Artifact Map

| Area | Main Inputs | Main Code | Outputs |
|---|---|---|---|
| Dataset | `dataset/train`, `dataset/val`, `dataset/test` | `training_utils.py`, `train_model.py` | `tf.data.Dataset` batches, class mappings |
| YOLO focus | original RGB images, optional YOLO weights | `yolo_leaf.py`, `train_yolo_leaf_detector.py` | focus masks, bbox metadata, `models/yolo26_leaf_detector.pt` |
| Training | dataset batches, config flags, backbone choice | `train_model.py`, `saliency_alignment.py`, `backbones.py` | checkpoints, classifier models, logs |
| Refinement | trained classifier/checkpoint | `fine_tune_model.py`, `refine_model.py` | deployment-ready `.keras` models |
| Evaluation | model + test split | `evaluate_model.py`, `evaluation/*.py` | reports, calibration artifacts, robustness plots |
| Inference | user image + model | `predict.py`, `disease_detection_pipeline.py`, `inference_guard.py` | prediction JSON, safety diagnostics |
| Web UI | uploads and control actions | `web/app.py`, `web/templates/index.html` | dashboard, job logs, focus overlay preview |
| Reporting | reports and metrics | `scripts/generate_*.py`, `tools/reporting/*.py` | publication figures and tables |

## Key Invariants

- Classification inputs are original resized RGB pixels.
- YOLO does not crop, mask, or remove image backgrounds for the classifier.
- YOLO focus masks are training targets for saliency alignment and review
  overlays only.
- The inference safety gate can reject low-confidence, high-entropy, or
  out-of-distribution-style predictions.
- Model preprocessing must match the loaded backbone; internal preprocessing
  blocks are detected and skipped where appropriate.
