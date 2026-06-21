import torch

print("=== GPU Diagnostic Test ===")
print(f"CUDA Available in PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

print("\nTesting Cast on GPU via PyTorch backend...")
try:
    x_torch = torch.tensor([1, 2], dtype=torch.int32).cuda()
    y_torch = x_torch.float()

    print("Success! Tensors successfully allocated and casted on the GPU.")
    print("PyTorch Tensor:", y_torch)
except Exception as e:
    print("Failed!")
    print(e)
