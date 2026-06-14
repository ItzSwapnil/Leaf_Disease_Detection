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

print("encoder_layers:", vit_encoder.encoder_layers)
print("Are they keras.layers.Layer?", [isinstance(blk, tf.keras.layers.Layer) for blk in vit_encoder.encoder_layers])
