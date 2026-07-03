import os

os.environ["KERAS_BACKEND"] = "torch"
import keras

try:
    print("Testing keras.layers.TorchModuleWrapper")
    # Just checking if it exists
    wrapper = keras.layers.TorchModuleWrapper
    print("Wrapper exists!")
except AttributeError as e:
    print("Error:", e)
