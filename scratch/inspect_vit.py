import os
import sys

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

print("vit_encoder type:", type(vit_encoder))
print("vit_encoder dir:")
print([a for a in dir(vit_encoder) if "layer" in a.lower()])

if hasattr(vit_encoder, "layers"):
    print("vit_encoder.layers:")
    for blk in vit_encoder.layers:
        print(f" - {blk.name} ({type(blk)})")
else:
    print("vit_encoder has no 'layers' attribute.")
