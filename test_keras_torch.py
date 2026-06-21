import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch

# 1. Define model
inputs = keras.Input(shape=(3, 224, 224))
x = keras.layers.Conv2D(32, 3, activation="relu")(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(10)(x)
model = keras.Model(inputs, outputs)

# 2. Check if it's an nn.Module
print("Is nn.Module?", isinstance(model, torch.nn.Module))

# 3. Train with pure PyTorch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

x = torch.randn(2, 3, 224, 224)
y = torch.tensor([1, 5])

for _ in range(3):
    optimizer.zero_grad()
    preds = model(x)
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()
    print("Loss:", loss.item())

# 4. Save both
model.save("test_model.keras")
torch.save(model.state_dict(), "test_model.pt")
print("Saved both!")
