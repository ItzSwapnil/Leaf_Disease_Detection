import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from train_model import resolve_backbone_factory

backbone_factory = resolve_backbone_factory("DINOv3")
base_model = backbone_factory(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

vit_encoder = base_model.get_layer("vit_encoder")
patch_embed = base_model.get_layer("vit_patching_and_embedding")

x = tf.random.normal((1, 224, 224, 3))

with tf.GradientTape() as tape:
    pe_out = patch_embed(x)
    
    if hasattr(vit_encoder, "dropout") and vit_encoder.dropout is not None:
        val_x = vit_encoder.dropout(pe_out)
    else:
        val_x = pe_out

    target_activation = None
    for i, blk in enumerate(vit_encoder.encoder_layers):
        val_x = blk(val_x, training=False)
        if i == 6:
            target_activation = val_x
            tape.watch(target_activation)

    if hasattr(vit_encoder, "layer_norm") and vit_encoder.layer_norm is not None:
        final_vit_out = vit_encoder.layer_norm(val_x)
    else:
        final_vit_out = val_x

    score = tf.reduce_sum(final_vit_out)

grads = tape.gradient(score, target_activation)
print("Are grads None?", grads is None)
if grads is not None:
    print("Grad shape:", grads.shape)
