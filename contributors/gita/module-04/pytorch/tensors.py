import sys
import torch

print(f"Python version: {sys.version}")
print(f"Pytorch version: {torch.__version__}")

print(f"CUDA avalaibility: {torch.cuda.is_available()}")

# TO SEE CUDA:

# # Uninstall everything first
# pip uninstall torch torchvision torchaudio -y

# # Reinstall with CUDA support (for CUDA 12.1 — works for most modern drivers)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if torch.cuda.is_available():
    print("CUDA version: {torch.version.cuda}")
    print("Number of gpu's: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        prop=torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {prop.name}")
    