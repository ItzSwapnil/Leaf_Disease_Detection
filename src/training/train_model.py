import argparse
import json
import math
import os
import random
import re
import time

# Some notebook environments export an inline backend string that may be
# unsupported in script mode. Normalize before TensorFlow imports Keras.
if (os.getenv("MPLBACKEND") or "").startswith("module://matplotlib_inline"):
    os.environ["MPLBACKEND"] = "Agg"

from datetime import datetime

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, TensorBoard
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model

from src.core.backbones import (
    list_backbone_names,
    resolve_backbone_factory,
    resolve_backbone_name,
)
from src.core.preprocessing import preprocess_batch_for_model_tf
from src.training.training_progress import (
    EpochReviewCallback,
    IntervalMetricsLogger,
    ProgressEmitter,
)
from src.training.training_utils import (
    BestModelSaver,
    FamilyDeviationClassifier,
    GradCamEpochCollageCallback,
    PreOverfitRestorer,
    WarmupCosineSchedule,
    _build_heavy_augmentation_layer,
    build_loss,
    build_optimizer,
    cutmix_batch_tf,
    mixup_batch_tf,
    parse_class_structure,
    resolve_augmentation_probabilities,
    resolve_step_count,
    tensorboard_available,
)
from src.utils.config import (
    BASE_MODEL,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLASS_INDICES_PATH,
    CUTMIX_PROB,
    DENSE_UNITS,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    IMG_SIZE,
    INTER_OP_THREADS,
    INTRA_OP_THREADS,
    LABEL_SMOOTHING,
    LEARNING_RATE_PHASE1,
    LEARNING_RATE_PHASE2,
    MIXUP_ALPHA,
    MIXUP_PROB,
    NORMAL_PROB,
    NUM_CLASSES,
    NUM_CROPS,
    OPTIMIZER,
    OVERFITTING_STOP_ENABLED,
    OVERFITTING_STOP_MIN_GAP,
    OVERFITTING_STOP_PATIENCE,
    SAVE_LOG_ARCHIVE,
    SAVE_RUN_MANIFESTS,
    STEPS_PER_EPOCH,
    TRAIN_DATA_FRACTION,
    TRAIN_DIR,
    UNFREEZE_LAYERS,
    USE_ATTENTION_GUIDANCE,
    USE_MIXUP,
    USE_RANDAUGMENT,
    VAL_DIR,
    VALIDATION_STEPS,
    WARMUP_EPOCHS,
)
from src.utils.hardware import configure_tensorflow, get_training_strategy

# Import optional sota augmentation flags with safe fallbacks
try:
    from src.utils.config import CUTMIX_ALPHA, USE_CUTMIX
except ImportError:
    USE_CUTMIX = False
    CUTMIX_ALPHA = 1.0

VALID_OPTIMIZERS = {
    "adamw": "AdamW",
    "adam": "Adam",
    "sgd": "SGD",
    "rmsprop": "RMSprop",
}
VALID_SAVE_MODES = {"with_optimizer", "without_optimizer", "all"}


def _parse_fraction(raw_value, default_value):
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = float(default_value)
    return max(1e-6, min(1.0, value))


def _parse_class_equalizer(arg_value):
    if arg_value is not None:
        return str(arg_value).strip().lower() == "on"
    env_value = os.getenv("LEAF_CLASS_EQUALIZER")
    if env_value is None:
        return True
    return str(env_value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_optimizer_name(raw_value):
    key = str(raw_value or OPTIMIZER or "AdamW").strip().lower()
    if key not in VALID_OPTIMIZERS:
        raise ValueError(
            "Unsupported optimizer "
            f"'{raw_value}'. Expected one of: {', '.join(VALID_OPTIMIZERS.values())}."
        )
    return VALID_OPTIMIZERS[key]


def _normalize_save_mode(raw_value):
    mode = str(raw_value or "with_optimizer").strip().lower().replace("-", "_")
    if mode not in VALID_SAVE_MODES:
        raise ValueError(
            "Unsupported save mode "
            f"'{raw_value}'. Expected one of: with_optimizer, without_optimizer, all."
        )
    return mode


def _canonical_class_name(name):
    """Normalize class names so small punctuation/whitespace differences still match."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _resolve_validation_class_aliases(val_class_names, train_class_names):
    """Map validation folder names to training class names using canonical aliases."""
    train_set = set(train_class_names)
    train_by_canonical = {}
    for train_name in train_class_names:
        key = _canonical_class_name(train_name)
        if key in train_by_canonical and train_by_canonical[key] != train_name:
            raise ValueError(
                "Ambiguous canonical training class names detected: "
                f"'{train_by_canonical[key]}' and '{train_name}'."
            )
        train_by_canonical[key] = train_name

    val_to_train = {}
    unmatched = []
    for val_name in val_class_names:
        if val_name in train_set:
            val_to_train[val_name] = val_name
            continue

        key = _canonical_class_name(val_name)
        mapped_train = train_by_canonical.get(key)
        if mapped_train is None:
            unmatched.append(val_name)
            continue
        val_to_train[val_name] = mapped_train

    if unmatched:
        raise ValueError(
            "Validation classes could not be mapped to training classes: "
            + ", ".join(sorted(unmatched))
        )

    return val_to_train


def _collect_sampled_training_files(train_dir, class_names, fraction, seed):
    rng = random.Random(seed)
    filepaths = []
    labels = []
    full_counts = {}
    sampled_counts = {}

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(train_dir, class_name)
        class_files = [
            entry.path for entry in os.scandir(class_dir) if entry.is_file()
        ]
        class_files.sort()

        full_counts[class_index] = len(class_files)
        if not class_files:
            sampled_counts[class_index] = 0
            continue

        if fraction >= 1.0:
            selected = list(class_files)
        else:
            keep_count = max(1, int(math.ceil(len(class_files) * fraction)))
            shuffled = list(class_files)
            rng.shuffle(shuffled)
            selected = shuffled[:keep_count]

        sampled_counts[class_index] = len(selected)
        filepaths.extend(selected)
        labels.extend([class_index] * len(selected))

    if not filepaths:
        raise ValueError(
            "No training files were found after applying the data fraction."
        )

    combined = list(zip(filepaths, labels))
    rng.shuffle(combined)
    sampled_paths, sampled_labels = zip(*combined)
    return (
        list(sampled_paths),
        list(sampled_labels),
        full_counts,
        sampled_counts,
    )


def _collect_validation_files(val_dir, val_to_train_map, class_indices):
    filepaths = []
    labels = []
    counts_by_val_class = {}

    for val_class_name in sorted(val_to_train_map.keys()):
        train_class_name = val_to_train_map[val_class_name]
        train_class_index = class_indices[train_class_name]
        class_dir = os.path.join(val_dir, val_class_name)
        class_files = [
            entry.path for entry in os.scandir(class_dir) if entry.is_file()
        ]
        class_files.sort()

        counts_by_val_class[val_class_name] = len(class_files)
        filepaths.extend(class_files)
        labels.extend([train_class_index] * len(class_files))

    if not filepaths:
        raise ValueError(
            "No validation files were found in the validation directory."
        )

    return filepaths, labels, counts_by_val_class


def _build_labeled_image_dataset(
    filepaths, labels, batch_size, shuffle, seed, crop_mapping=None
):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=max(1, min(len(filepaths), 20000)),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    if crop_mapping is not None:
        crop_mapping_tensor = tf.constant(crop_mapping, dtype=tf.int32)
    else:
        crop_mapping_tensor = None

    def _decode(path, label):
        image_bytes = tf.io.read_file(path)
        image_tensor = tf.io.decode_jpeg(image_bytes, channels=3)
        image_tensor = tf.image.resize(image_tensor, (IMG_SIZE, IMG_SIZE))
        image_tensor = tf.cast(image_tensor, tf.float32)

        disease_one_hot = tf.one_hot(
            tf.cast(label, tf.int32), depth=NUM_CLASSES
        )
        disease_one_hot = tf.cast(disease_one_hot, tf.float32)

        if crop_mapping_tensor is not None:
            crop_label = crop_mapping_tensor[label]
            crop_one_hot = tf.one_hot(crop_label, depth=NUM_CROPS)
            crop_one_hot = tf.cast(crop_one_hot, tf.float32)
            targets = {
                "crop_output": crop_one_hot,
                "disease_output": disease_one_hot,
            }
        else:
            targets = disease_one_hot

        return image_tensor, targets

    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds


def _compute_equalizer_weights_from_counts(counts_by_class):
    counts = [
        int(counts_by_class[idx]) for idx in sorted(counts_by_class.keys())
    ]
    if not counts or any(count <= 0 for count in counts):
        return None

    mean_count = float(sum(counts)) / float(len(counts))
    if mean_count <= 0.0:
        return None

    weights = {
        idx: mean_count / float(count) for idx, count in enumerate(counts)
    }

    avg_weight = float(sum(weights.values())) / float(len(weights))
    if avg_weight <= 0.0:
        return None

    return {idx: float(weight / avg_weight) for idx, weight in weights.items()}


# Main training entrypoint


def main():
    parser = argparse.ArgumentParser(
        description="Leaf Disease Detection training pipeline"
    )
    parser.add_argument(
        "--base-model",
        choices=list_backbone_names(),
        default=None,
        help="Backbone to use for training (defaults to LEAF_BASE_MODEL or EfficientNetV2B0).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help=(
            "Per-class random sampling fraction for train data (0..1]. "
            "Defaults to LEAF_TRAIN_DATA_FRACTION."
        ),
    )
    parser.add_argument(
        "--optimizer",
        default=None,
        help="Optimizer for training (AdamW, Adam, SGD, RMSprop).",
    )
    parser.add_argument(
        "--save-mode",
        default=None,
        help="Model save mode: with_optimizer, without_optimizer, or all.",
    )
    parser.add_argument(
        "--class-equalizer",
        choices=["on", "off"],
        default=None,
        help="Enable or disable strict per-class equalizer weighting.",
    )
    parser.add_argument(
        "--must-review",
        choices=["on", "off"],
        default=None,
        help="Wait for user review at the end of each epoch.",
    )
    args = parser.parse_args()

    must_review_env = os.getenv("LEAF_MUST_REVIEW") == "1"
    must_review_arg = (
        args.must_review == "on" if args.must_review is not None else None
    )
    must_review_enabled = (
        must_review_arg if must_review_arg is not None else must_review_env
    )

    backbone_name = resolve_backbone_name(
        args.base_model or os.getenv("LEAF_BASE_MODEL"),
        default=BASE_MODEL,
    )
    train_data_fraction = _parse_fraction(
        args.train_fraction,
        os.getenv("LEAF_TRAIN_DATA_FRACTION", TRAIN_DATA_FRACTION),
    )
    optimizer_name = _normalize_optimizer_name(
        args.optimizer or os.getenv("LEAF_TRAIN_OPTIMIZER") or OPTIMIZER
    )
    save_mode = _normalize_save_mode(
        args.save_mode or os.getenv("LEAF_SAVE_MODE")
    )
    class_equalizer_enabled = _parse_class_equalizer(args.class_equalizer)

    env_batch_size = os.getenv("LEAF_BATCH_SIZE")
    batch_size = int(BATCH_SIZE)
    gpu_count = len(tf.config.list_physical_devices("GPU"))
    if env_batch_size is not None:
        try:
            batch_size = max(1, int(env_batch_size))
        except Exception:
            batch_size = int(BATCH_SIZE)
    elif backbone_name == "DINOv3" and batch_size > 8:
        target_batch = 16 if gpu_count > 1 else 8
        if batch_size > target_batch:
            batch_size = target_batch
            print(
                "Auto-adjusted batch size to "
                f"{batch_size} for DINOv3 on {gpu_count} GPU(s). "
                "Override via LEAF_BATCH_SIZE if needed."
            )

    print(f"Training pipeline  |  Backbone: {backbone_name}")
    print("Target: 99%+ top-1 accuracy on PlantVillage-46")
    print(
        "Train options: "
        f"fraction={train_data_fraction:.3f}, "
        f"optimizer={optimizer_name}, "
        f"save_mode={save_mode}, "
        f"class_equalizer={'on' if class_equalizer_enabled else 'off'}"
    )

    # Reproducibility
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # Allow overriding the run seed via RUN_SEED env var for multi-seed experiments
    seed_env = os.environ.get("RUN_SEED")
    try:
        seed = int(seed_env) if seed_env is not None else 42
    except Exception:
        seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    keras.utils.set_random_seed(seed)
    print(f"Using run seed: {seed}")

    try:
        tf.config.threading.set_intra_op_parallelism_threads(INTRA_OP_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)
    except RuntimeError as exc:
        print(f"TensorFlow threading config skipped: {exc}")

    configure_tensorflow()

    # Mixed precision: halves gpu memory footprint with negligible accuracy loss
    if tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: mixed_float16 (2x memory savings)")
    else:
        keras.mixed_precision.set_global_policy("float32")

    models_dir = os.path.dirname(CHECKPOINT_PATH)
    os.makedirs(models_dir, exist_ok=True)
    logs_dir = os.path.join(models_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # ── data loading ──────────────────────────────────────────────────────
    autotune = tf.data.AUTOTUNE
    dataset_parallelism = autotune

    print(f"\nLoading training data from: {TRAIN_DIR}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}  |  Batch size: {batch_size}")

    train_class_names = sorted(
        entry.name for entry in os.scandir(TRAIN_DIR) if entry.is_dir()
    )
    if not train_class_names:
        raise ValueError(
            f"No class folders found under training directory: {TRAIN_DIR}"
        )

    class_indices = {name: idx for idx, name in enumerate(train_class_names)}

    crop_names = sorted(
        list(set(name.split("___")[0] for name in train_class_names))
    )
    if len(crop_names) != NUM_CROPS:
        print(
            f"Warning: Extracted {len(crop_names)} crops, but NUM_CROPS={NUM_CROPS}"
        )
    crop_mapping = [
        crop_names.index(name.split("___")[0]) for name in train_class_names
    ]

    sampled_paths, sampled_labels, full_counts, sampled_counts = (
        _collect_sampled_training_files(
            str(TRAIN_DIR),
            train_class_names,
            fraction=float(train_data_fraction),
            seed=seed,
        )
    )
    train_ds = _build_labeled_image_dataset(
        sampled_paths,
        sampled_labels,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        crop_mapping=crop_mapping,
    )

    val_class_names = sorted(
        entry.name for entry in os.scandir(VAL_DIR) if entry.is_dir()
    )
    if not val_class_names:
        raise ValueError(
            f"No class folders found under validation directory: {VAL_DIR}"
        )

    val_to_train_map = _resolve_validation_class_aliases(
        val_class_names=val_class_names,
        train_class_names=train_class_names,
    )
    alias_changes = {
        val_name: train_name
        for val_name, train_name in val_to_train_map.items()
        if val_name != train_name
    }
    if alias_changes:
        print("Validation class alias mapping applied:")
        for val_name, train_name in sorted(alias_changes.items()):
            print(f"  - {val_name} -> {train_name}")

    val_paths, val_labels, val_counts = _collect_validation_files(
        str(VAL_DIR),
        val_to_train_map=val_to_train_map,
        class_indices=class_indices,
    )
    val_ds = _build_labeled_image_dataset(
        val_paths,
        val_labels,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        crop_mapping=crop_mapping,
    )

    train_samples = len(sampled_paths)
    val_samples = len(val_paths)

    print(
        "Training subset selection: "
        f"{train_samples} samples from {sum(full_counts.values())} total "
        f"(fraction={train_data_fraction:.3f} per class)"
    )
    print(
        "Per-class sampled counts: "
        f"min={min(sampled_counts.values())}, "
        f"max={max(sampled_counts.values())}, "
        f"classes={len(sampled_counts)}"
    )
    print(
        "Validation class counts: "
        f"min={min(val_counts.values())}, "
        f"max={max(val_counts.values())}, "
        f"classes={len(val_counts)}"
    )
    print(f"Validation samples: {val_samples}")
    print(f"Number of classes: {NUM_CLASSES}")

    train_options = tf.data.Options()
    train_options.experimental_deterministic = False
    train_ds = train_ds.with_options(train_options)

    if USE_RANDAUGMENT:
        print(
            "Augmentation: Heavy pipeline "
            "(RandomResizedCrop+Flip+Rotation+ColorJitter+GaussianBlur+"
            "GaussianNoise+RandomErasing)"
        )
        heavy_augment_fn = _build_heavy_augmentation_layer(
            value_range=(0.0, 255.0),
        )
        train_ds = train_ds.map(
            lambda images, labels: (
                heavy_augment_fn(images, training=True),
                labels,
            ),
            num_parallel_calls=dataset_parallelism,
        )

    train_ds = train_ds.map(
        lambda images, labels: (
            preprocess_batch_for_model_tf(images, backbone_name=backbone_name),
            labels,
        ),
        num_parallel_calls=dataset_parallelism,
    )
    val_ds = val_ds.map(
        lambda images, labels: (
            preprocess_batch_for_model_tf(images, backbone_name=backbone_name),
            labels,
        ),
        num_parallel_calls=dataset_parallelism,
    )

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    # Parse healthy baseline mapping for Family-Based Disease Learning
    healthy_partners = parse_class_structure(train_class_names)

    # ── model construction ────────────────────────────────────────────────
    strategy = get_training_strategy()
    with strategy.scope():
        backbone_factory = resolve_backbone_factory(backbone_name)
        base_model = backbone_factory(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False

        x = base_model.output
        output_shape = getattr(base_model, "output_shape", None)
        output_rank = (
            len(output_shape) if isinstance(output_shape, tuple) else None
        )
        if output_rank == 4:
            x = GlobalAveragePooling2D()(x)
        elif output_rank == 3:
            x = GlobalAveragePooling1D()(x)
        elif output_rank == 2:
            pass
        else:
            raise ValueError(
                f"Unsupported backbone output shape for {backbone_name}: {output_shape}"
            )
        x = BatchNormalization(dtype="float32", name="head_bn")(x)
        x = Dense(DENSE_UNITS, activation="swish", name="head_dense_1")(x)
        x = Dropout(DROPOUT_RATE, name="head_dropout_1")(x)
        x = Dense(DENSE_UNITS // 2, activation="swish", name="head_dense_2")(x)
        x = Dropout(DROPOUT_RATE * 0.5, name="head_dropout_2")(x)
        crop_logits = Dense(NUM_CROPS, name="crop_logits")(x)
        crop_outputs = keras.layers.Activation(
            "softmax", dtype="float32", name="crop_output"
        )(crop_logits)

        disease_logits = FamilyDeviationClassifier(
            num_classes=NUM_CLASSES,
            healthy_partners=healthy_partners,
            name="disease_logits",
        )(x)
        disease_outputs = keras.layers.Activation(
            "softmax", dtype="float32", name="disease_output"
        )(disease_logits)

        model = Model(
            inputs=base_model.input,
            outputs={
                "crop_output": crop_outputs,
                "disease_output": disease_outputs,
            },
        )

    print(f"\nBackbone: {backbone_name}")
    print(f"Total parameters: {model.count_params():,}")
    trainable_params = sum(p.numpy().size for p in model.trainable_weights)
    print(f"Trainable parameters (phase 1): {trainable_params:,}")
    full_steps_per_epoch = resolve_step_count(
        STEPS_PER_EPOCH, train_samples, batch_size
    )
    steps_per_epoch = full_steps_per_epoch
    validation_steps = resolve_step_count(
        VALIDATION_STEPS, val_samples, batch_size
    )

    phase1_epochs = max(0, int(EPOCHS_PHASE1))
    phase2_epochs = max(0, int(EPOCHS_PHASE2))
    total_epochs = phase1_epochs + phase2_epochs

    # Persist class-to-index mapping for inference
    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as class_file:
        json.dump(class_indices, class_file, indent=2)

    # ── class weighting and loss ──────────────────────────────────────────
    class_weight = None
    if class_equalizer_enabled:
        class_weight = _compute_equalizer_weights_from_counts(sampled_counts)
        if class_weight:
            print(
                "Class equalizer enabled (strict inverse-frequency weighting)."
            )
            class_names_by_idx = {
                idx: name for name, idx in class_indices.items()
            }
            top = sorted(
                class_weight.items(), key=lambda item: item[1], reverse=True
            )[:6]
            print(
                "Highest class weights: "
                + ", ".join(
                    f"{class_names_by_idx.get(idx, idx)}={weight:.2f}"
                    for idx, weight in top
                )
            )
        else:
            print(
                "Class equalizer requested, but weights could not be computed."
            )
    else:
        print("Class equalizer disabled; no class weighting will be used.")

    selected_loss, fit_class_weight = build_loss(
        class_weight, class_names=train_class_names
    )

    # ── mixup / cutmix ───────────────────────────────────────────────────
    if USE_MIXUP or USE_CUTMIX:
        augmentation_desc = []
        if USE_MIXUP:
            augmentation_desc.append(f"MixUp(alpha={MIXUP_ALPHA})")
        if USE_CUTMIX:
            augmentation_desc.append(f"CutMix(alpha={CUTMIX_ALPHA})")
        print(f"Augmentation: {' + '.join(augmentation_desc)}")
        mixup_prob, cutmix_prob, normal_prob = (
            resolve_augmentation_probabilities(
                use_mixup=USE_MIXUP,
                use_cutmix=USE_CUTMIX,
                mixup_prob=float(MIXUP_PROB),
                cutmix_prob=float(CUTMIX_PROB),
                normal_prob=float(NORMAL_PROB),
            )
        )
        print(
            "Batch routing probabilities: "
            f"MixUp={mixup_prob:.2f}, CutMix={cutmix_prob:.2f}, Normal={normal_prob:.2f}"
        )
        if fit_class_weight:
            print(
                "MixUp/CutMix active: disabling class_weight to avoid label-mix conflicts."
            )
            fit_class_weight = None

        def _apply_batch_augmentation(images, labels):
            route_sample = tf.random.uniform([])
            if USE_MIXUP and USE_CUTMIX:
                return tf.cond(
                    route_sample < mixup_prob,
                    lambda: mixup_batch_tf(
                        images, labels, alpha=float(MIXUP_ALPHA)
                    ),
                    lambda: tf.cond(
                        route_sample < (mixup_prob + cutmix_prob),
                        lambda: cutmix_batch_tf(
                            images, labels, alpha=float(CUTMIX_ALPHA)
                        ),
                        lambda: (images, labels),
                    ),
                )
            if USE_MIXUP:
                return tf.cond(
                    route_sample < mixup_prob,
                    lambda: mixup_batch_tf(
                        images, labels, alpha=float(MIXUP_ALPHA)
                    ),
                    lambda: (images, labels),
                )
            return tf.cond(
                route_sample < cutmix_prob,
                lambda: cutmix_batch_tf(
                    images, labels, alpha=float(CUTMIX_ALPHA)
                ),
                lambda: (images, labels),
            )

        train_ds = train_ds.map(
            _apply_batch_augmentation, num_parallel_calls=dataset_parallelism
        )

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    train_source = train_ds
    val_source = val_ds

    # ── callbacks ─────────────────────────────────────────────────────────
    checkpoint = BestModelSaver(
        CHECKPOINT_PATH,
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_mode=save_mode,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )
    if OVERFITTING_STOP_ENABLED:
        print(
            "Overfitting stop enabled: "
            f"min_gap={OVERFITTING_STOP_MIN_GAP:.3f}, "
            f"patience={int(OVERFITTING_STOP_PATIENCE)}"
        )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"train_{run_stamp}"
    train_history_latest_path = os.path.join(logs_dir, "train_history.csv")
    train_history_archive_path = os.path.join(
        logs_dir, f"train_history_{run_stamp}.csv"
    )
    train_interval_latest_path = os.path.join(
        logs_dir, "train_interval_history.csv"
    )
    train_interval_archive_path = os.path.join(
        logs_dir, f"train_interval_history_{run_stamp}.csv"
    )
    latest_runs_path = os.path.join(logs_dir, "latest_runs.json")

    csv_loggers_phase1 = [CSVLogger(train_history_latest_path, append=False)]
    if SAVE_LOG_ARCHIVE:
        csv_loggers_phase1.append(
            CSVLogger(train_history_archive_path, append=False)
        )
    csv_loggers_phase2 = [CSVLogger(train_history_latest_path, append=True)]
    if SAVE_LOG_ARCHIVE:
        csv_loggers_phase2.append(
            CSVLogger(train_history_archive_path, append=True)
        )

    interval_loggers_phase1 = [
        IntervalMetricsLogger(
            train_interval_latest_path,
            points_per_epoch=12,
            stage="train_full",
            append=False,
            run_id=run_id,
        ),
    ]
    if SAVE_LOG_ARCHIVE:
        interval_loggers_phase1.append(
            IntervalMetricsLogger(
                train_interval_archive_path,
                points_per_epoch=12,
                stage="train_full",
                append=False,
                run_id=run_id,
            )
        )
    interval_loggers_phase2 = [
        IntervalMetricsLogger(
            train_interval_latest_path,
            points_per_epoch=12,
            stage="train_full",
            append=True,
            run_id=run_id,
        ),
    ]
    if SAVE_LOG_ARCHIVE:
        interval_loggers_phase2.append(
            IntervalMetricsLogger(
                train_interval_archive_path,
                points_per_epoch=12,
                stage="train_full",
                append=True,
                run_id=run_id,
            )
        )

    print(f"Training history log: {train_history_latest_path}")
    print(f"Interval history log: {train_interval_latest_path}")
    if SAVE_LOG_ARCHIVE:
        print(f"Archive training log: {train_history_archive_path}")
        print(f"Archive interval log: {train_interval_archive_path}")

    tensorboard = None
    if tensorboard_available():
        tensorboard = TensorBoard(
            log_dir=os.path.join(logs_dir, "tensorboard"),
            histogram_freq=1,
            update_freq="epoch",
            write_graph=False,
        )
    else:
        print("TensorBoard not installed; skipping TensorBoard callback.")

    print(f"\n{'=' * 70}")
    print(f"Phase 1 (head-only): {phase1_epochs} epochs")
    print(f"Phase 2 (full fine-tune): {phase2_epochs} epochs")
    print(f"Total epochs: {total_epochs}  |  Steps/epoch: {steps_per_epoch}")
    print(
        f"Optimiser: {optimizer_name}  |  Label smoothing: {LABEL_SMOOTHING}"
    )
    print(f"{'=' * 70}\n")

    run_start_time = time.time()

    collage_callback = GradCamEpochCollageCallback(
        val_dir=VAL_DIR,
        class_names=train_class_names,
        backbone_name=backbone_name,
    )
    # Everlearning: Load existing weights if they exist
    from src.utils.config import FINAL_MODEL_PATH

    if os.path.exists(CHECKPOINT_PATH):
        try:
            print(
                f"\n[Everlearning] Resuming training from existing checkpoint: {CHECKPOINT_PATH}"
            )
            if str(CHECKPOINT_PATH).endswith(".keras"):
                import tempfile
                import zipfile

                with zipfile.ZipFile(CHECKPOINT_PATH, "r") as zip_ref:
                    with tempfile.TemporaryDirectory() as td:
                        zip_ref.extract("model.weights.h5", path=td)
                        model.load_weights(
                            os.path.join(td, "model.weights.h5"),
                            by_name=True,
                            skip_mismatch=True,
                        )
            else:
                model.load_weights(
                    CHECKPOINT_PATH, by_name=True, skip_mismatch=True
                )
            print("[Everlearning] Successfully loaded matching weights.")
        except Exception as e:
            print(f"Failed to load weights: {e}")
    elif os.path.exists(FINAL_MODEL_PATH):
        print(
            f"\n[Everlearning] Resuming training from final model: {FINAL_MODEL_PATH}"
        )
        try:
            model.load_weights(
                FINAL_MODEL_PATH, skip_mismatch=True, by_name=True
            )
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Failed to load weights: {e}")
    else:
        print(
            "\\n[Everlearning] No existing weights found. Starting from scratch (pretrained backbone)."
        )

    # ── phase 1: train classification head only ───────────────────────────
    if phase1_epochs > 0:
        phase1_total_steps = max(1, steps_per_epoch * phase1_epochs)
        phase1_warmup_steps = max(
            0, steps_per_epoch * min(int(WARMUP_EPOCHS), phase1_epochs)
        )
        phase1_lr = WarmupCosineSchedule(
            peak_lr=LEARNING_RATE_PHASE1,
            min_lr=max(LEARNING_RATE_PHASE1 * 0.01, 1e-7),
            warmup_steps=phase1_warmup_steps,
            total_steps=phase1_total_steps,
        )

        training_model = model
        if USE_ATTENTION_GUIDANCE:
            from src.core.saliency_alignment import SaliencyAlignedModel

            disease_class_indices = [
                idx
                for idx, name in enumerate(train_class_names)
                if "healthy" not in name.lower()
                and "background" not in name.lower()
            ]

            with strategy.scope():
                training_model = SaliencyAlignedModel(
                    model,
                    backbone_name=backbone_name,
                    disease_class_indices=disease_class_indices,
                    enable_penalties=False,  # Phase 1: backbone frozen
                    class_names=train_class_names,
                )

        with strategy.scope():
            selected_loss_crop = keras.losses.CategoricalCrossentropy(
                label_smoothing=LABEL_SMOOTHING, name="crop_loss"
            )
            selected_loss_disease = selected_loss

            training_model.compile(
                optimizer=build_optimizer(
                    phase1_lr, optimizer_name=optimizer_name
                ),
                loss={
                    "crop_output": selected_loss_crop,
                    "disease_output": selected_loss_disease,
                },
                loss_weights={
                    "crop_output": 0.3,
                    "disease_output": 1.0,
                },
                metrics={
                    "crop_output": ["accuracy"],
                    "disease_output": ["accuracy"],
                },
            )

        progress_phase1 = ProgressEmitter(
            stage="phase1_warmup",
            total_epochs=total_epochs,
            completed_epochs_before=0,
            run_start_time=run_start_time,
        )

        print(f"\n--- Phase 1: warm-up on {train_samples} training images ---")
        phase1_callbacks: list[keras.callbacks.Callback] = [
            checkpoint,
            early_stopping,
            *csv_loggers_phase1,
            *interval_loggers_phase1,
            progress_phase1,
            collage_callback,
            EpochReviewCallback(
                enabled=must_review_enabled,
                total_epochs=total_epochs,
                stage="phase1_warmup",
            ),
        ]
        if OVERFITTING_STOP_ENABLED:
            phase1_callbacks.append(
                PreOverfitRestorer(
                    min_gap=float(OVERFITTING_STOP_MIN_GAP),
                    patience=int(OVERFITTING_STOP_PATIENCE),
                    verbose=1,
                    save_path=CHECKPOINT_PATH,
                    save_mode=save_mode,
                )
            )
        if tensorboard is not None:
            phase1_callbacks.append(tensorboard)

        phase1_history = training_model.fit(
            train_source,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_source,
            validation_steps=validation_steps,
            epochs=phase1_epochs,
            callbacks=phase1_callbacks,
            class_weight=fit_class_weight,
            verbose=1,
        )
        completed_phase1_epochs = len(phase1_history.history.get("loss", []))
    else:
        completed_phase1_epochs = 0

    # ── phase 2: unfreeze backbone and fine-tune ──────────────────────────
    if phase2_epochs > 0:
        base_model.trainable = True
        if int(UNFREEZE_LAYERS) > 0:
            for layer in base_model.layers[: -int(UNFREEZE_LAYERS)]:
                layer.trainable = False
            print(f"Unfroze top {UNFREEZE_LAYERS} backbone layers.")
        else:
            print("Unfroze entire backbone for fine-tuning.")

        # Keep batchnormalization layers frozen for training stability
        bn_frozen = 0
        for layer in base_model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
                bn_frozen += 1
        if bn_frozen > 0:
            print(
                f"Froze {bn_frozen} BatchNormalization layers for stability."
            )

        phase2_total_steps = max(1, steps_per_epoch * phase2_epochs)
        phase2_warmup_steps = max(
            0, steps_per_epoch * min(int(WARMUP_EPOCHS), phase2_epochs)
        )
        phase2_lr = WarmupCosineSchedule(
            peak_lr=LEARNING_RATE_PHASE2,
            min_lr=max(LEARNING_RATE_PHASE2 * 0.01, 1e-7),
            warmup_steps=phase2_warmup_steps,
            total_steps=phase2_total_steps,
        )

        training_model = model
        if USE_ATTENTION_GUIDANCE:
            from src.core.saliency_alignment import SaliencyAlignedModel

            disease_class_indices = [
                idx
                for idx, name in enumerate(train_class_names)
                if "healthy" not in name.lower()
                and "background" not in name.lower()
            ]

            with strategy.scope():
                training_model = SaliencyAlignedModel(
                    model,
                    backbone_name=backbone_name,
                    disease_class_indices=disease_class_indices,
                    enable_penalties=True,  # Phase 2: backbone unfrozen
                    class_names=train_class_names,
                )

        with strategy.scope():
            selected_loss_crop = keras.losses.CategoricalCrossentropy(
                label_smoothing=LABEL_SMOOTHING, name="crop_loss"
            )
            selected_loss_disease = selected_loss

            training_model.compile(
                optimizer=build_optimizer(
                    phase2_lr, optimizer_name=optimizer_name
                ),
                loss={
                    "crop_output": selected_loss_crop,
                    "disease_output": selected_loss_disease,
                },
                loss_weights={
                    "crop_output": 0.3,
                    "disease_output": 1.0,
                },
                metrics={
                    "crop_output": ["accuracy"],
                    "disease_output": ["accuracy"],
                },
            )

        total_epochs_phase2_view = max(
            1, completed_phase1_epochs + phase2_epochs
        )
        progress_phase2 = ProgressEmitter(
            stage="phase2_finetune",
            total_epochs=total_epochs_phase2_view,
            completed_epochs_before=completed_phase1_epochs,
            run_start_time=run_start_time,
        )

        print("\n--- Phase 2: fine-tuning entire network ---")
        phase2_callbacks: list[keras.callbacks.Callback] = [
            checkpoint,
            early_stopping,
            *csv_loggers_phase2,
            *interval_loggers_phase2,
            progress_phase2,
            collage_callback,
            EpochReviewCallback(
                enabled=must_review_enabled,
                total_epochs=total_epochs,
                stage="phase2_finetune",
            ),
        ]
        if OVERFITTING_STOP_ENABLED:
            phase2_callbacks.append(
                PreOverfitRestorer(
                    min_gap=float(OVERFITTING_STOP_MIN_GAP),
                    patience=int(OVERFITTING_STOP_PATIENCE),
                    verbose=1,
                    save_path=CHECKPOINT_PATH,
                    save_mode=save_mode,
                )
            )
        if tensorboard is not None:
            phase2_callbacks.append(tensorboard)

        training_model.fit(
            train_source,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_source,
            validation_steps=validation_steps,
            initial_epoch=completed_phase1_epochs,
            epochs=completed_phase1_epochs + phase2_epochs,
            callbacks=phase2_callbacks,
            class_weight=fit_class_weight,
            verbose=1,
        )

    elapsed = time.time() - run_start_time
    print(f"\nTraining completed in {elapsed / 3600:.2f} hours")

    # ── run manifest ──────────────────────────────────────────────────────
    latest_runs = {}
    if os.path.exists(latest_runs_path):
        try:
            with open(latest_runs_path, "r", encoding="utf-8") as in_file:
                latest_runs = json.load(in_file)
        except Exception:
            latest_runs = {}

    train_manifest = {
        "run_id": run_id,
        "run_stamp": run_stamp,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_history_latest": train_history_latest_path,
        "train_history_archive": train_history_archive_path
        if SAVE_LOG_ARCHIVE
        else None,
        "train_interval_latest": train_interval_latest_path,
        "train_interval_archive": train_interval_archive_path
        if SAVE_LOG_ARCHIVE
        else None,
        "base_model": backbone_name,
        "optimizer": optimizer_name,
        "save_mode": save_mode,
        "class_equalizer": bool(class_equalizer_enabled),
        "batch_size": int(batch_size),
        "train_data_fraction": float(train_data_fraction),
        "train_samples": int(train_samples),
        "train_samples_per_class": {
            train_class_names[idx]: int(sampled_counts[idx])
            for idx in sorted(sampled_counts.keys())
        },
        "use_mixup": USE_MIXUP,
        "use_cutmix": USE_CUTMIX,
        "epochs_phase1": phase1_epochs,
        "epochs_phase2": phase2_epochs,
    }
    latest_runs["train"] = train_manifest

    if SAVE_RUN_MANIFESTS or SAVE_LOG_ARCHIVE:
        with open(
            os.path.join(logs_dir, f"train_run_manifest_{run_stamp}.json"),
            "w",
            encoding="utf-8",
        ) as out_file:
            json.dump(train_manifest, out_file, indent=2)
    with open(latest_runs_path, "w", encoding="utf-8") as out_file:
        json.dump(latest_runs, out_file, indent=2)

    return model


if __name__ == "__main__":
    main()
