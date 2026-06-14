import tensorflow as tf
import tensorflow.keras as keras

from src.utils.config import (
    ATTENTION_BG_PENALTY_WEIGHT,
    ATTENTION_SPARSITY_WEIGHT,
    ATTENTION_VIT_BLOCK_IDX,
    ATTENTION_VIT_BLOCK_INDICES,
)


class SaliencyAlignedModel(keras.Model):
    """Keras Model wrapper to apply Saliency/Attention Alignment training.

    Supports multi-block ViT attention regularization: penalties are
    computed independently for each target encoder block and then
    averaged, giving the optimizer consistent pressure across the
    full depth of the network.
    """

    def __init__(
        self,
        functional_model: keras.Model,
        backbone_name: str = "DINOv3",
        bg_weight: float = ATTENTION_BG_PENALTY_WEIGHT,
        sparsity_weight: float = ATTENTION_SPARSITY_WEIGHT,
        disease_reward_weight: float = 0.05,
        disease_class_indices: list[int] | None = None,
        vit_block_indices: tuple[int, ...] | list[int] | None = None,
        vit_block_idx: int = ATTENTION_VIT_BLOCK_IDX,
        enable_penalties: bool = True,
        class_names: list[str] | None = None,
    ):
        super().__init__()
        self.class_names = class_names
        self.has_human_ann = False

        if class_names:
            ann_data = self._load_human_annotations(class_names)
            if ann_data is not None:
                from src.core.backbones import resolve_preprocess_function

                preprocess_fn = resolve_preprocess_function(backbone_name)
                proc_imgs = preprocess_fn(ann_data["images"])

                self.ann_images = tf.constant(proc_imgs, dtype=tf.float32)
                self.ann_leaf_masks = tf.constant(
                    ann_data["leaf_masks"], dtype=tf.float32
                )
                self.ann_focus_masks = tf.constant(
                    ann_data["focus_masks"], dtype=tf.float32
                )
                self.ann_labels = tf.constant(
                    ann_data["labels"], dtype=tf.int32
                )
                self.has_human_ann = True
                print(
                    f"Loaded {len(proc_imgs)} human annotations for saliency alignment!"
                )
        self.functional_model = functional_model
        self.backbone_name = backbone_name
        self.bg_weight = bg_weight
        self.sparsity_weight = sparsity_weight
        self.disease_reward_weight = disease_reward_weight
        self.disease_class_indices = disease_class_indices
        # Penalties are disabled in Phase 1 (frozen backbone).
        self.enable_penalties = enable_penalties

        # Resolve multi-block indices with backward compat
        if vit_block_indices is not None:
            self.vit_block_indices = tuple(vit_block_indices)
        elif ATTENTION_VIT_BLOCK_INDICES:
            self.vit_block_indices = tuple(ATTENTION_VIT_BLOCK_INDICES)
        else:
            self.vit_block_indices = (vit_block_idx,)

        # Initialize metrics for tracking
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.cls_loss_tracker = keras.metrics.Mean(name="cls_loss")
        self.bg_penalty_tracker = keras.metrics.Mean(name="bg_penalty")
        self.sparsity_penalty_tracker = keras.metrics.Mean(
            name="sparsity_penalty"
        )
        self.disease_reward_tracker = keras.metrics.Mean(name="disease_reward")
        self.crop_bg_penalty_tracker = keras.metrics.Mean(
            name="crop_bg_penalty"
        )
        self.crop_coverage_reward_tracker = keras.metrics.Mean(
            name="crop_coverage_reward"
        )
        self.human_bg_penalty_tracker = keras.metrics.Mean(
            name="human_bg_penalty"
        )
        self.human_disease_reward_tracker = keras.metrics.Mean(
            name="human_disease_reward"
        )

        self.grad_model = None
        if self.backbone_name != "DINOv3":
            target_layer = self._find_conv_layer(self.functional_model)
            if target_layer is not None:
                outputs_list = [
                    target_layer.output,
                    self.functional_model.output,
                ]
                self.grad_model = keras.Model(
                    inputs=self.functional_model.input,
                    outputs=outputs_list,
                )

    def _load_human_annotations(self, class_names):
        import os

        import cv2
        import numpy as np

        annotated_images = []
        annotated_leaf_masks = []
        annotated_focus_masks = []
        annotated_labels = []

        samples_dir = os.path.join("annotations", "samples")
        masks_dir = os.path.join("annotations", "masks")

        if not os.path.exists(masks_dir):
            return None

        for idx, class_name in enumerate(class_names):
            img_path = os.path.join(samples_dir, f"{class_name}.jpg")
            leaf_mask_path = os.path.join(masks_dir, f"{class_name}_leaf.png")
            focus_mask_path = os.path.join(
                masks_dir, f"{class_name}_focus.png"
            )

            if (
                os.path.exists(img_path)
                and os.path.exists(leaf_mask_path)
                and os.path.exists(focus_mask_path)
            ):
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))

                # Load masks with alpha if possible, or fall back to gray
                leaf_mask = cv2.imread(leaf_mask_path, cv2.IMREAD_UNCHANGED)
                if leaf_mask is None:
                    continue
                if len(leaf_mask.shape) == 3 and leaf_mask.shape[2] == 4:
                    leaf_mask = leaf_mask[:, :, 3]
                else:
                    leaf_mask = cv2.cvtColor(leaf_mask, cv2.COLOR_BGR2GRAY)
                leaf_mask = cv2.resize(leaf_mask, (224, 224))
                _, leaf_mask = cv2.threshold(
                    leaf_mask, 127, 255, cv2.THRESH_BINARY
                )
                leaf_mask = (leaf_mask / 255.0).astype(np.float32)

                focus_mask = cv2.imread(focus_mask_path, cv2.IMREAD_UNCHANGED)
                if focus_mask is None:
                    continue
                if len(focus_mask.shape) == 3 and focus_mask.shape[2] == 4:
                    focus_mask = focus_mask[:, :, 3]
                else:
                    focus_mask = cv2.cvtColor(focus_mask, cv2.COLOR_BGR2GRAY)
                focus_mask = cv2.resize(focus_mask, (224, 224))
                _, focus_mask = cv2.threshold(
                    focus_mask, 127, 255, cv2.THRESH_BINARY
                )
                focus_mask = (focus_mask / 255.0).astype(np.float32)

                annotated_images.append(img.astype(np.float32))
                annotated_leaf_masks.append(np.expand_dims(leaf_mask, -1))
                annotated_focus_masks.append(np.expand_dims(focus_mask, -1))
                annotated_labels.append(idx)

        if not annotated_images:
            return None

        return {
            "images": np.array(annotated_images),
            "leaf_masks": np.array(annotated_leaf_masks),
            "focus_masks": np.array(annotated_focus_masks),
            "labels": np.array(annotated_labels),
        }

    def _find_conv_layer(self, model):
        for layer in reversed(model.layers):
            if hasattr(layer, "layers"):
                for sublayer in reversed(layer.layers):
                    if (
                        len(getattr(sublayer, "output_shape", ())) == 4
                        and "conv" in sublayer.name.lower()
                    ):
                        return sublayer
            if (
                len(getattr(layer, "output_shape", ())) == 4
                and "conv" in layer.name.lower()
            ):
                return layer
        return None

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.cls_loss_tracker,
            self.bg_penalty_tracker,
            self.sparsity_penalty_tracker,
            self.disease_reward_tracker,
            self.crop_bg_penalty_tracker,
            self.crop_coverage_reward_tracker,
            self.human_bg_penalty_tracker,
            self.human_disease_reward_tracker,
        ] + super().metrics

    def call(self, inputs, training=None):
        return self.functional_model(inputs, training=training)

    def get_layer(self, name=None, index=None):
        return self.functional_model.get_layer(name=name, index=index)

    @property
    def layers(self):
        return self.functional_model.layers

    def save_weights(self, filepath, overwrite=True, **kwargs):
        return self.functional_model.save_weights(
            filepath, overwrite=overwrite, **kwargs
        )

    def load_weights(self, filepath, **kwargs):
        return self.functional_model.load_weights(filepath, **kwargs)

    def save(self, filepath, overwrite=True, **kwargs):
        return self.functional_model.save(
            filepath, overwrite=overwrite, **kwargs
        )

    def _compute_crop_heatmap_penalties(self, heatmap_grid, bg_mask_resized):
        bg_mask_resized = tf.cast(bg_mask_resized, dtype=heatmap_grid.dtype)
        gc_bg = tf.reduce_sum(heatmap_grid * bg_mask_resized, axis=[1, 2, 3])
        gc_tot = tf.reduce_sum(heatmap_grid, axis=[1, 2, 3]) + 1e-5
        crop_bg_penalty = gc_bg / gc_tot

        eps = 1e-5
        gc_tot_keepdims = (
            tf.reduce_sum(heatmap_grid, axis=[1, 2, 3], keepdims=True) + 1e-5
        )
        hm_prob = heatmap_grid / gc_tot_keepdims
        entropy = -tf.reduce_sum(
            hm_prob * tf.math.log(hm_prob + eps), axis=[1, 2, 3]
        )
        crop_coverage_reward = entropy

        return tf.reduce_mean(crop_bg_penalty), tf.reduce_mean(
            crop_coverage_reward
        )

    def _compute_disease_heatmap_penalties(
        self,
        heatmap_grid,
        bg_mask_resized,
        anomaly_mask_resized,
        is_diseased,
        attn_scores=None,
    ):
        """Compute bg_penalty, sparsity, and disease_reward for one block.

        Returns (bg_penalty, sparsity_penalty, disease_reward) as scalars.
        """
        # Calculate per-sample penalties conditioned on disease status.

        bg_mask_resized = tf.cast(bg_mask_resized, dtype=heatmap_grid.dtype)
        anomaly_mask_resized = tf.cast(
            anomaly_mask_resized, dtype=heatmap_grid.dtype
        )

        # 1. Background Penalty (energy outside leaf)
        gc_bg = tf.reduce_sum(heatmap_grid * bg_mask_resized, axis=[1, 2, 3])
        gc_tot = tf.reduce_sum(heatmap_grid, axis=[1, 2, 3]) + 1e-5
        gradcam_bg_penalty = gc_bg / gc_tot

        # 2. Sparsity Penalty (spatial entropy of normalized heatmap)
        eps = 1e-5
        # Normalize heatmap per image.
        gc_tot_keepdims = (
            tf.reduce_sum(heatmap_grid, axis=[1, 2, 3], keepdims=True) + 1e-5
        )
        hm_prob = heatmap_grid / gc_tot_keepdims
        # Compute entropy per image
        entropy = -tf.reduce_sum(
            hm_prob * tf.math.log(hm_prob + eps), axis=[1, 2, 3]
        )

        # Sparsity for diseased samples only.
        gradcam_sparsity = entropy * tf.cast(
            tf.squeeze(is_diseased, axis=[-1, -2, -3]), heatmap_grid.dtype
        )

        # 3. Disease Reward (energy inside anomaly mask)
        gc_anomaly = tf.reduce_sum(
            heatmap_grid * anomaly_mask_resized, axis=[1, 2, 3]
        )
        gradcam_disease_reward = (gc_anomaly / gc_tot) * tf.cast(
            tf.squeeze(is_diseased, axis=[-1, -2, -3]), heatmap_grid.dtype
        )

        # Mix with ViT self-attention if available.
        if attn_scores is not None:
            # CLS token attention.
            cls_attn = attn_scores[:, :, 0, 1:]
            mean_cls_attn = tf.reduce_mean(cls_attn, axis=1)
            attn_grid = tf.reshape(mean_cls_attn, (-1, 14, 14, 1))
            # Normalize attention map.
            attn_f32 = tf.cast(attn_grid, tf.float32)
            max_attn = (
                tf.reduce_max(
                    attn_f32,
                    axis=[1, 2, 3],
                    keepdims=True,
                )
                + 1e-5
            )
            attn_norm = attn_f32 / max_attn
            attn_grid = tf.cast(
                attn_norm,
                dtype=bg_mask_resized.dtype,
            )

            attn_bg = tf.reduce_sum(
                attn_grid * bg_mask_resized, axis=[1, 2, 3]
            )
            attn_tot = tf.reduce_sum(attn_grid, axis=[1, 2, 3]) + 1e-5
            vit_bg_penalty = attn_bg / attn_tot

            attn_sparsity = tf.reduce_mean(
                attn_grid, axis=[1, 2, 3]
            ) * tf.cast(
                tf.squeeze(is_diseased, axis=[-1, -2, -3]), attn_grid.dtype
            )

            attn_anomaly = tf.reduce_sum(
                attn_grid * anomaly_mask_resized, axis=[1, 2, 3]
            )
            vit_disease_reward = (attn_anomaly / attn_tot) * tf.cast(
                tf.squeeze(is_diseased, axis=[-1, -2, -3]), attn_grid.dtype
            )

            bg_penalty = tf.reduce_mean(
                0.5 * gradcam_bg_penalty + 0.5 * vit_bg_penalty
            )
            sparsity_penalty = tf.reduce_mean(
                0.5 * gradcam_sparsity + 0.5 * attn_sparsity
            )
            disease_reward = tf.reduce_mean(
                0.5 * gradcam_disease_reward + 0.5 * vit_disease_reward
            )
        else:
            bg_penalty = tf.reduce_mean(gradcam_bg_penalty)
            sparsity_penalty = tf.reduce_mean(gradcam_sparsity)
            disease_reward = tf.reduce_mean(gradcam_disease_reward)

        return bg_penalty, sparsity_penalty, disease_reward

    @tf.function
    def train_step(self, data):
        x, y = data
        if isinstance(y, dict):
            _y_crop = y["crop_output"]
            y_disease = y["disease_output"]
        else:
            _y_crop = y
            y_disease = y

        # 1. Reconstruct original images for bg mask (in [0, 255])
        if self.backbone_name == "DINOv3":
            mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
            std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
            x_orig = (x * std + mean) * 255.0
        else:
            x_orig = x

        # HSV-based background/shadow mask
        x_01 = tf.clip_by_value(x_orig / 255.0, 0.0, 1.0)
        hsv = tf.image.rgb_to_hsv(x_01)
        saturation = hsv[..., 1:2]
        value = hsv[..., 2:3]
        bg_mask = tf.cast(
            (saturation <= 0.15) | (value <= 0.08) | (value >= 0.94),
            dtype=tf.float32,
        )

        # Leaf mask for anomaly detection
        leaf_mask = 1.0 - bg_mask
        leaf_pixels_count = (
            tf.reduce_sum(leaf_mask, axis=[1, 2], keepdims=True) + 1e-5
        )
        mean_leaf_color = (
            tf.reduce_sum(x_orig * leaf_mask, axis=[1, 2], keepdims=True)
            / leaf_pixels_count
        )
        color_diff = tf.reduce_mean(
            tf.abs(x_orig - mean_leaf_color), axis=-1, keepdims=True
        )
        anomaly_score = color_diff * leaf_mask
        max_anomaly = (
            tf.reduce_max(anomaly_score, axis=[1, 2], keepdims=True) + 1e-5
        )
        anomaly_mask = anomaly_score / max_anomaly

        # Determine if sample is diseased
        batch_size = tf.shape(y_disease)[0]
        if self.disease_class_indices:
            disease_tensor = tf.constant(
                self.disease_class_indices, dtype=tf.int32
            )
            is_diseased_1d = tf.reduce_sum(
                tf.gather(y_disease, disease_tensor, axis=1), axis=1
            )
        else:
            is_diseased_1d = tf.ones((batch_size,), dtype=y_disease.dtype)
        is_diseased = tf.reshape(is_diseased_1d, (-1, 1, 1, 1))

        # 2. Forward/backward with double gradient tape
        trainable_vars = self.functional_model.trainable_variables

        with tf.GradientTape() as tape:
            with tf.GradientTape(persistent=True) as inner:
                if self.backbone_name == "DINOv3":
                    (
                        logits,
                        target_activations,
                        target_attn_list,
                    ) = self._forward_vit_multiblock(x, inner)
                else:
                    if self.grad_model is not None:
                        target_act, logits = self.grad_model(x, training=True)
                        inner.watch(target_act)
                        target_activations = [target_act]
                        target_attn_list = [None]
                    else:
                        target_activations = []
                        target_attn_list = []
                        logits = self.functional_model(x, training=True)

                # Classification loss
                cls_loss = self.compiled_loss(y, logits)

                # Saliency alignment on human annotated images
                if self.has_human_ann and self.enable_penalties:
                    ann_size = tf.shape(self.ann_images)[0]
                    num_samples = tf.minimum(4, ann_size)
                    indices = tf.random.uniform(
                        [num_samples],
                        minval=0,
                        maxval=ann_size,
                        dtype=tf.int32,
                    )

                    batch_ann_images = tf.gather(self.ann_images, indices)
                    batch_ann_leaf_masks = tf.gather(
                        self.ann_leaf_masks, indices
                    )
                    batch_ann_focus_masks = tf.gather(
                        self.ann_focus_masks, indices
                    )
                    batch_ann_labels = tf.gather(self.ann_labels, indices)

                    # Determine is_diseased for annotations
                    if self.disease_class_indices:
                        disease_tensor = tf.constant(
                            self.disease_class_indices, dtype=tf.int32
                        )
                        ann_is_diseased_1d = tf.map_fn(
                            lambda lbl: tf.cast(
                                tf.reduce_any(tf.equal(disease_tensor, lbl)),
                                tf.float32,
                            ),
                            batch_ann_labels,
                            fn_output_signature=tf.float32,
                        )
                    else:
                        ann_is_diseased_1d = tf.ones(
                            (num_samples,), dtype=tf.float32
                        )
                    ann_is_diseased = tf.reshape(
                        ann_is_diseased_1d, (-1, 1, 1, 1)
                    )

                    # Forward pass
                    if self.backbone_name == "DINOv3":
                        (
                            ann_logits,
                            ann_activations,
                            ann_attn_list,
                        ) = self._forward_vit_multiblock(
                            batch_ann_images, inner
                        )
                    else:
                        if self.grad_model is not None:
                            ann_act, ann_log = self.grad_model(
                                batch_ann_images, training=True
                            )
                            inner.watch(ann_act)
                            ann_activations = [ann_act]
                            ann_attn_list = [None]
                            ann_logits = ann_log
                        else:
                            ann_activations = []
                            ann_attn_list = []
                            ann_logits = self.functional_model(
                                batch_ann_images, training=True
                            )

                    if isinstance(ann_logits, dict):
                        ann_disease_logits = ann_logits["disease_output"]
                    else:
                        ann_disease_logits = ann_logits
                    # Gather the scores for the ground-truth labels
                    ann_scores = tf.gather(
                        ann_disease_logits, batch_ann_labels, batch_dims=1
                    )

                # Predicted class score for Grad-CAM
                if isinstance(logits, dict):
                    crop_logits = logits["crop_output"]
                    disease_logits = logits["disease_output"]
                else:
                    crop_logits = logits
                    disease_logits = logits

                crop_idx = tf.argmax(crop_logits, axis=-1)
                crop_scores = tf.reduce_sum(
                    crop_logits
                    * tf.one_hot(crop_idx, tf.shape(crop_logits)[-1]),
                    axis=-1,
                )

                disease_idx = tf.argmax(disease_logits, axis=-1)
                disease_scores = tf.reduce_sum(
                    disease_logits
                    * tf.one_hot(disease_idx, tf.shape(disease_logits)[-1]),
                    axis=-1,
                )

            # Compute per-block penalties, then average
            bg_penalties = []
            sparsity_penalties = []
            disease_rewards = []
            crop_bg_penalties = []
            crop_coverage_rewards = []

            for blk_i in range(len(target_activations)):
                act = target_activations[blk_i]
                attn = target_attn_list[blk_i]

                disease_grads = inner.gradient(disease_scores, act)
                crop_grads = inner.gradient(crop_scores, act)
                if disease_grads is None or crop_grads is None:
                    continue

                if self.backbone_name == "DINOv3":
                    _, disease_hm_grid = self._gradcam_vit(disease_grads, act)
                    _, crop_hm_grid = self._gradcam_vit(crop_grads, act)

                    bg_resized = tf.image.resize(
                        bg_mask,
                        (14, 14),
                        method="nearest",
                    )
                    bg_resized = tf.cast(
                        bg_resized,
                        dtype=disease_hm_grid.dtype,
                    )
                    anomaly_resized = tf.image.resize(
                        anomaly_mask,
                        (14, 14),
                        method="nearest",
                    )
                    anomaly_resized = tf.cast(
                        anomaly_resized,
                        dtype=disease_hm_grid.dtype,
                    )
                else:
                    disease_hm_grid = self._gradcam_cnn(disease_grads, act)
                    crop_hm_grid = self._gradcam_cnn(crop_grads, act)
                    act_shape = tf.shape(act)
                    h_a, w_a = act_shape[1], act_shape[2]
                    bg_resized = tf.image.resize(
                        bg_mask,
                        (h_a, w_a),
                        method="nearest",
                    )
                    bg_resized = tf.cast(
                        bg_resized,
                        dtype=disease_hm_grid.dtype,
                    )
                    anomaly_resized = tf.image.resize(
                        anomaly_mask,
                        (h_a, w_a),
                        method="nearest",
                    )
                    anomaly_resized = tf.cast(
                        anomaly_resized,
                        dtype=disease_hm_grid.dtype,
                    )
                    attn = None  # CNN has no attn scores

                bp, sp, dr = self._compute_disease_heatmap_penalties(
                    disease_hm_grid,
                    bg_resized,
                    anomaly_resized,
                    is_diseased,
                    attn,
                )
                cbp, ccr = self._compute_crop_heatmap_penalties(
                    crop_hm_grid, bg_resized
                )

                bg_penalties.append(bp)
                sparsity_penalties.append(sp)
                disease_rewards.append(dr)
                crop_bg_penalties.append(cbp)
                crop_coverage_rewards.append(ccr)

            # Compute human annotation alignment penalties
            human_bg_penalty = tf.constant(0.0, dtype=tf.float32)
            human_disease_reward = tf.constant(0.0, dtype=tf.float32)

            if self.has_human_ann and self.enable_penalties:
                ann_bg_penalties = []
                ann_sparsity_penalties = []
                ann_disease_rewards = []

                for blk_i in range(len(ann_activations)):
                    act = ann_activations[blk_i]
                    attn = ann_attn_list[blk_i]

                    ann_disease_grads = inner.gradient(ann_scores, act)
                    if ann_disease_grads is None:
                        continue

                    if self.backbone_name == "DINOv3":
                        _, ann_hm_grid = self._gradcam_vit(
                            ann_disease_grads, act
                        )
                        ann_bg_resized = tf.image.resize(
                            1.0 - batch_ann_leaf_masks,
                            (14, 14),
                            method="nearest",
                        )
                        ann_bg_resized = tf.cast(
                            ann_bg_resized, dtype=ann_hm_grid.dtype
                        )
                        ann_anomaly_resized = tf.image.resize(
                            batch_ann_focus_masks, (14, 14), method="nearest"
                        )
                        ann_anomaly_resized = tf.cast(
                            ann_anomaly_resized, dtype=ann_hm_grid.dtype
                        )
                    else:
                        ann_hm_grid = self._gradcam_cnn(ann_disease_grads, act)
                        act_shape = tf.shape(act)
                        h_a, w_a = act_shape[1], act_shape[2]
                        ann_bg_resized = tf.image.resize(
                            1.0 - batch_ann_leaf_masks,
                            (h_a, w_a),
                            method="nearest",
                        )
                        ann_bg_resized = tf.cast(
                            ann_bg_resized, dtype=ann_hm_grid.dtype
                        )
                        ann_anomaly_resized = tf.image.resize(
                            batch_ann_focus_masks, (h_a, w_a), method="nearest"
                        )
                        ann_anomaly_resized = tf.cast(
                            ann_anomaly_resized, dtype=ann_hm_grid.dtype
                        )
                        attn = None

                    bp, sp, dr = self._compute_disease_heatmap_penalties(
                        ann_hm_grid,
                        ann_bg_resized,
                        ann_anomaly_resized,
                        ann_is_diseased,
                        attn,
                    )
                    ann_bg_penalties.append(bp)
                    ann_sparsity_penalties.append(sp)
                    ann_disease_rewards.append(dr)

                if ann_bg_penalties:
                    human_bg_penalty = tf.add_n(ann_bg_penalties) / len(
                        ann_bg_penalties
                    )
                    human_disease_reward = tf.add_n(ann_disease_rewards) / len(
                        ann_disease_rewards
                    )

            del inner  # Release persistent tape

            # Average across blocks (or zero if penalties disabled)
            if bg_penalties and self.enable_penalties:
                bg_penalty = tf.add_n(bg_penalties) / len(bg_penalties)
                sparsity_penalty = tf.add_n(sparsity_penalties) / len(
                    sparsity_penalties
                )
                disease_reward = tf.add_n(disease_rewards) / len(
                    disease_rewards
                )
                crop_bg_penalty = tf.add_n(crop_bg_penalties) / len(
                    crop_bg_penalties
                )
                crop_coverage_reward = tf.add_n(crop_coverage_rewards) / len(
                    crop_coverage_rewards
                )
            else:
                bg_penalty = tf.constant(0.0, dtype=tf.float32)
                sparsity_penalty = tf.constant(0.0, dtype=tf.float32)
                disease_reward = tf.constant(0.0, dtype=tf.float32)
                crop_bg_penalty = tf.constant(0.0, dtype=tf.float32)
                crop_coverage_reward = tf.constant(0.0, dtype=tf.float32)

            # Combined total loss
            total_loss = (
                cls_loss
                + self.bg_weight * tf.cast(bg_penalty, tf.float32)
                + self.sparsity_weight * tf.cast(sparsity_penalty, tf.float32)
                - self.disease_reward_weight
                * tf.cast(disease_reward, tf.float32)
                + 0.5 * self.bg_weight * tf.cast(crop_bg_penalty, tf.float32)
                - 0.1
                * self.sparsity_weight
                * tf.cast(crop_coverage_reward, tf.float32)
            )

            if self.has_human_ann and self.enable_penalties:
                # Add human alignment loss with a strong weight
                human_weight = 5.0
                total_loss = total_loss + human_weight * (
                    self.bg_weight * tf.cast(human_bg_penalty, tf.float32)
                    - self.disease_reward_weight
                    * tf.cast(human_disease_reward, tf.float32)
                )

        # Outer gradients
        grads_outer = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads_outer, trainable_vars))

        # Update metrics
        self.loss_tracker.update_state(total_loss)
        self.cls_loss_tracker.update_state(cls_loss)
        self.bg_penalty_tracker.update_state(bg_penalty)
        self.sparsity_penalty_tracker.update_state(sparsity_penalty)
        self.disease_reward_tracker.update_state(disease_reward)
        self.crop_bg_penalty_tracker.update_state(crop_bg_penalty)
        self.crop_coverage_reward_tracker.update_state(crop_coverage_reward)
        self.human_bg_penalty_tracker.update_state(human_bg_penalty)
        self.human_disease_reward_tracker.update_state(human_disease_reward)

        self.compute_metrics(x, y, logits, sample_weight=None)

        results = {}
        for m in self.metrics:
            res = m.result()
            if isinstance(res, dict):
                results.update(res)
            else:
                results[m.name] = res
        return results

    def _forward_vit_multiblock(self, x, tape):
        """Forward through the ViT, intercepting multiple
        encoder blocks for Grad-CAM and attention scores.

        Returns (logits, activations_list, attn_list).
        """
        vit_encoder = self.functional_model.get_layer("vit_encoder")
        patch_embed = self.functional_model.get_layer(
            "vit_patching_and_embedding"
        )
        pool = self.functional_model.get_layer("global_average_pooling1d")

        pe_out = patch_embed(x)
        val_x = vit_encoder.dropout(pe_out)

        target_indices = set(self.vit_block_indices)
        activations = []
        attn_list = []

        # Robustly extract encoder blocks
        encoder_blocks = []
        if hasattr(vit_encoder, "encoder_layers"):
            encoder_blocks = vit_encoder.encoder_layers
        elif hasattr(vit_encoder, "transformer_layers"):
            encoder_blocks = vit_encoder.transformer_layers
        elif hasattr(vit_encoder, "layers"):
            for layer in vit_encoder.layers:
                if "encoder_block" in layer.name:
                    encoder_blocks.append(layer)

        for i, blk in enumerate(encoder_blocks):
            if i in target_indices:
                # Try to extract attention scores
                if hasattr(blk, "layer_norm_1") and hasattr(blk, "mha"):
                    x_norm = blk.layer_norm_1(val_x)
                    mha_out, attn_scores = blk.mha(
                        x_norm,
                        x_norm,
                        return_attention_scores=True,
                        training=True,
                    )
                    x_attn = blk.dropout(mha_out, training=True)
                    x_attn = x_attn + val_x
                    y_norm = blk.layer_norm_2(x_attn)
                    y_mlp = blk.mlp(y_norm, training=True)
                    val_x = x_attn + y_mlp
                elif hasattr(blk, "attention"):
                    # For some other architectures
                    val_x = blk(val_x, training=True)
                    attn_scores = (
                        None  # We cannot easily extract without custom forward
                    )
                else:
                    # Fallback standard call
                    val_x = blk(val_x, training=True)
                    attn_scores = None

                tape.watch(val_x)
                activations.append(val_x)
                attn_list.append(attn_scores)
            else:
                val_x = blk(val_x, training=True)

        final_vit_out = vit_encoder.layer_norm(val_x)
        pooled = pool(final_vit_out)

        x_head = pooled
        try:
            x_head = self.functional_model.get_layer("head_bn")(x_head)
            x_head = self.functional_model.get_layer("head_dense_1")(x_head)
            x_head = self.functional_model.get_layer("head_dropout_1")(x_head)
            x_head = self.functional_model.get_layer("head_dense_2")(x_head)
            x_head = self.functional_model.get_layer("head_dropout_2")(x_head)

            crop_logits = self.functional_model.get_layer("crop_logits")(
                x_head
            )
            crop_output = self.functional_model.get_layer("crop_output")(
                crop_logits
            )

            disease_logits = self.functional_model.get_layer("disease_logits")(
                x_head
            )
            disease_output = self.functional_model.get_layer("disease_output")(
                disease_logits
            )
            logits = {
                "crop_output": crop_output,
                "disease_output": disease_output,
            }
        except ValueError:
            raise ValueError(
                "SaliencyAlignedModel requires standard head names (head_bn, etc.)"
            )

        return logits, activations, attn_list

    def _gradcam_vit(self, grads, activation):
        """Compute Grad-CAM heatmap for a ViT block.

        Returns (heatmap_1d, heatmap_grid_14x14).
        """
        pooled_grads = tf.reduce_mean(grads, axis=1, keepdims=True)
        heatmap = tf.reduce_sum(activation * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0.0)
        heatmap = heatmap[:, 1:]  # Remove CLS token
        # Normalize in float32
        hm_f32 = tf.cast(heatmap, tf.float32)
        max_val = tf.reduce_max(hm_f32, axis=-1, keepdims=True) + 1e-5
        hm_norm = hm_f32 / max_val
        hm_grid = tf.reshape(
            tf.cast(hm_norm, heatmap.dtype),
            (-1, 14, 14, 1),
        )
        return heatmap, hm_grid

    def _gradcam_cnn(self, grads, activation):
        """Compute Grad-CAM heatmap for a CNN block.

        Returns heatmap_grid (B, H, W, 1).
        """
        pooled_grads = tf.reduce_mean(grads, axis=[1, 2], keepdims=True)
        heatmap = tf.reduce_sum(activation * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0.0)
        heatmap = tf.expand_dims(heatmap, axis=-1)
        # Normalize in float32
        hm_f32 = tf.cast(heatmap, tf.float32)
        max_val = (
            tf.reduce_max(
                hm_f32,
                axis=[1, 2, 3],
                keepdims=True,
            )
            + 1e-5
        )
        hm_norm = hm_f32 / max_val
        return tf.cast(hm_norm, dtype=heatmap.dtype)

    @tf.function
    def test_step(self, data):
        x, y = data
        logits = self.functional_model(x, training=False)
        loss = self.compute_loss(x, y, logits)

        self.loss_tracker.update_state(loss)
        self.cls_loss_tracker.update_state(loss)
        self.bg_penalty_tracker.update_state(0.0)
        self.sparsity_penalty_tracker.update_state(0.0)
        self.disease_reward_tracker.update_state(0.0)
        self.crop_bg_penalty_tracker.update_state(0.0)
        self.crop_coverage_reward_tracker.update_state(0.0)
        self.human_bg_penalty_tracker.update_state(0.0)
        self.human_disease_reward_tracker.update_state(0.0)

        self.compute_metrics(x, y, logits, sample_weight=None)

        results = {}
        for m in self.metrics:
            res = m.result()
            if isinstance(res, dict):
                results.update(res)
            else:
                results[m.name] = res
        return results
