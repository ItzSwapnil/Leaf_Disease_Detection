import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch


class Simple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

m = Simple()
inputs = keras.Input(shape=(10,))
wrapper = keras.layers.TorchModuleWrapper(m)
outputs = wrapper(inputs)
k_model = keras.Model(inputs, outputs)
k_model.save("test_wrap.keras")
print("Saved!")
