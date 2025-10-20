import torch

print(f"CUDA available: {torch.cuda.is_available()}")

device = torch.device("cuda")

tensor = torch.tensor((0, 0, 0)).to(device, dtype=torch.bfloat16)
print(tensor)
