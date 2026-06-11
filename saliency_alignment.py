import tensorflow as tf
import tensorflow.keras as keras

from config import (
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
    ):
        super().__init__()
        self.functional_model = functional_model
        self.backbone_name = backbone_name
        self.bg_weight = bg_weight
        self.sparsity_weight = sparsity_weight
        self.disease_reward_weight = disease_reward_weight
        self.disease_class_indices = disease_class_indices
        # When False (Phase 1, frozen backbone), all spatial
        # penalties are zeroed out so the head can train safely.
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

    def _compute_heatmap_penalties(
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
        # We calculate penalties per sample first, then average over the batch
        # because sparsity and reward are conditioned on whether the sample is diseased.

        # 1. Background Penalty (Energy fraction outside leaf)
        gc_bg = tf.reduce_sum(heatmap_grid * bg_mask_resized, axis=[1, 2, 3])
        gc_tot = tf.reduce_sum(heatmap_grid, axis=[1, 2, 3]) + 1e-5
        gradcam_bg_penalty = gc_bg / gc_tot

        # 2. Sparsity Penalty
        # Use spatial entropy of the normalized heatmap as a true sparsity metric.
        # Minimizing entropy forces the attention distribution to be highly peaked (focused)
        # rather than diffuse (attention everywhere).
        eps = 1e-5
        # Normalize heatmap to a probability distribution (sum to 1) per image
        gc_tot_keepdims = tf.reduce_sum(heatmap_grid, axis=[1, 2, 3], keepdims=True) + 1e-5
        hm_prob = heatmap_grid / gc_tot_keepdims
        # Compute entropy per image
        entropy = -tf.reduce_sum(hm_prob * tf.math.log(hm_prob + eps), axis=[1, 2, 3])
        
        # Only apply sparsity if the sample is diseased
        # Healthy samples shouldn't be forced to focus on a single spot.
        gradcam_sparsity = entropy * tf.cast(
            tf.squeeze(is_diseased, axis=[-1, -2, -3]), heatmap_grid.dtype
        )

        # 3. Disease Reward (Energy fraction inside anomaly mask)
        # Only reward if the sample is diseased
        gc_anomaly = tf.reduce_sum(
            heatmap_grid * anomaly_mask_resized, axis=[1, 2, 3]
        )
        gradcam_disease_reward = (gc_anomaly / gc_tot) * tf.cast(tf.squeeze(
            is_diseased, axis=[-1, -2, -3]
        ), heatmap_grid.dtype)

        # Mix with direct ViT Self-Attention if available
        if attn_scores is not None:
            # CLS token attention to patch tokens
            cls_attn = attn_scores[:, :, 0, 1:]
            mean_cls_attn = tf.reduce_mean(cls_attn, axis=1)
            attn_grid = tf.reshape(mean_cls_attn, (-1, 14, 14, 1))
            # Normalize self-attention map
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
            ) * tf.cast(tf.squeeze(is_diseased, axis=[-1, -2, -3]), attn_grid.dtype)

            attn_anomaly = tf.reduce_sum(
                attn_grid * anomaly_mask_resized, axis=[1, 2, 3]
            )
            vit_disease_reward = (attn_anomaly / attn_tot) * tf.cast(tf.squeeze(
                is_diseased, axis=[-1, -2, -3]
            ), attn_grid.dtype)

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

        # 1. Reconstruct original images for bg mask
        if self.backbone_name == "DINOv3":
            mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
            std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
            x_orig = (x * std + mean) * 255.0
        else:
            x_orig = x

        mean_val = tf.reduce_mean(x_orig, axis=-1, keepdims=True)
        variance = tf.reduce_mean(
            tf.square(x_orig - mean_val),
            axis=-1,
            keepdims=True,
        )
        std_val = tf.sqrt(variance + 1e-8)
        # Background: std <= 8.0 or mean brightness <= 20.0
        bg_mask = tf.cast(
            (std_val <= 8.0) | (mean_val <= 20.0),
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
        batch_size = tf.shape(y)[0]
        if self.disease_class_indices:
            disease_tensor = tf.constant(
                self.disease_class_indices, dtype=tf.int32
            )
            is_diseased_1d = tf.reduce_sum(
                tf.gather(y, disease_tensor, axis=1), axis=1
            )
        else:
            is_diseased_1d = tf.ones((batch_size,), dtype=y.dtype)
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

                # Predicted class score for Grad-CAM
                pred_idx = tf.argmax(logits, axis=-1)
                pred_scores = tf.reduce_sum(
                    logits * tf.one_hot(pred_idx, tf.shape(logits)[-1]),
                    axis=-1,
                )

            # Compute per-block penalties, then average
            bg_penalties = []
            sparsity_penalties = []
            disease_rewards = []

            for blk_i in range(len(target_activations)):
                act = target_activations[blk_i]
                attn = target_attn_list[blk_i]

                grads = inner.gradient(pred_scores, act)
                if grads is None:
                    continue

                if self.backbone_name == "DINOv3":
                    hm, hm_grid = self._gradcam_vit(grads, act)
                    bg_resized = tf.image.resize(
                        bg_mask,
                        (14, 14),
                        method="nearest",
                    )
                    bg_resized = tf.cast(
                        bg_resized,
                        dtype=hm_grid.dtype,
                    )
                    anomaly_resized = tf.image.resize(
                        anomaly_mask,
                        (14, 14),
                        method="nearest",
                    )
                    anomaly_resized = tf.cast(
                        anomaly_resized,
                        dtype=hm_grid.dtype,
                    )
                else:
                    hm_grid = self._gradcam_cnn(grads, act)
                    act_shape = tf.shape(act)
                    h_a, w_a = act_shape[1], act_shape[2]
                    bg_resized = tf.image.resize(
                        bg_mask,
                        (h_a, w_a),
                        method="nearest",
                    )
                    bg_resized = tf.cast(
                        bg_resized,
                        dtype=hm_grid.dtype,
                    )
                    anomaly_resized = tf.image.resize(
                        anomaly_mask,
                        (h_a, w_a),
                        method="nearest",
                    )
                    anomaly_resized = tf.cast(
                        anomaly_resized,
                        dtype=hm_grid.dtype,
                    )
                    attn = None  # CNN has no attn scores

                bp, sp, dr = self._compute_heatmap_penalties(
                    hm_grid, bg_resized, anomaly_resized, is_diseased, attn
                )
                bg_penalties.append(bp)
                sparsity_penalties.append(sp)
                disease_rewards.append(dr)

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
            else:
                bg_penalty = tf.constant(0.0, dtype=tf.float32)
                sparsity_penalty = tf.constant(0.0, dtype=tf.float32)
                disease_reward = tf.constant(0.0, dtype=tf.float32)

            # Combined total loss
            total_loss = (
                cls_loss
                + self.bg_weight * tf.cast(bg_penalty, tf.float32)
                + self.sparsity_weight * tf.cast(sparsity_penalty, tf.float32)
                - self.disease_reward_weight
                * tf.cast(disease_reward, tf.float32)
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

        for i in range(vit_encoder.num_layers):
            if i in target_indices:
                blk = vit_encoder.encoder_layers[i]
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

                # Watch activation for Grad-CAM
                tape.watch(val_x)
                activations.append(val_x)
                attn_list.append(attn_scores)
            else:
                val_x = vit_encoder.encoder_layers[i](val_x)

        final_vit_out = vit_encoder.layer_norm(val_x)
        pooled = pool(final_vit_out)

        x_head = pooled
        for layer in self.functional_model.layers[4:]:
            x_head = layer(x_head)
        logits = x_head

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

        self.compute_metrics(x, y, logits, sample_weight=None)

        results = {}
        for m in self.metrics:
            res = m.result()
            if isinstance(res, dict):
                results.update(res)
            else:
                results[m.name] = res
        return results
