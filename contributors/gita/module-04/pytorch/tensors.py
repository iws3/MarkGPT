import sys
import torch

print(f"Python version: {sys.version}")
print(f"Pytorch version: {torch.__version__}")

print(f"CUDA avalaibility: {torch.cuda.is_available()}")