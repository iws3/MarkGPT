# Lesson L01.4: Setting Up Your Learning Environment
## Day 1: Welcome & Orientation | Preparing Your Development Environment for LLM Training

### Lesson Overview
Before you can begin building neural networks, training language models, and eventually launching MarkGPT, you need a working development environment. This lesson walks you through the one-time setup: installing Python, creating a virtual environment, installing dependencies (NumPy, PyTorch, Jupyter), and optionally configuring GPU acceleration. This may seem tedious, but a properly configured environment saves hours of debugging later. By the end of this lesson, you should be able to run a simple PyTorch program that verifies your GPU (if available) and confirms all dependencies are installed.

## Table of Contents
- Hardware Requirements
- Software Prerequisites
- Python Environment Setup
- Installing Dependencies
- GPU Configuration
- Testing Your Setup
- Troubleshooting Common Issues
- Resources and Next Steps

---

## Hardware Requirements

To train LLMs effectively, you'll need adequate hardware. Minimum requirements include:

- CPU: Modern multi-core processor (8+ cores recommended)

- RAM: 16GB minimum, 32GB+ preferred

- Storage: 500GB SSD for datasets and models

- GPU: NVIDIA GPU with at least 8GB VRAM (RTX 3060 or better)

For larger models, more powerful GPUs or cloud instances may be necessary.

---

## Software Prerequisites

Before setting up Python, ensure you have:

- Operating System: Linux, macOS, or Windows 10/11

- Git for version control

- CUDA toolkit for GPU acceleration (if using NVIDIA GPU)

- Docker for containerized environments (optional but recommended)

- Text editor or IDE (VS Code, PyCharm, etc.)

---

## Python Environment Setup

Create a virtual environment to isolate dependencies:

```bash
python -m venv markgpt_env
source markgpt_env/bin/activate  # On Windows: markgpt_env\Scripts\activate
```

---

## Installing Dependencies

Install core dependencies via pip:

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib jupyter scikit-learn transformers datasets wandb
```

Verify the installation:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## GPU Configuration (Optional)

If you have an NVIDIA GPU:

1. Download CUDA Toolkit matching your PyTorch version
2. Download cuDNN from NVIDIA
3. Follow platform-specific installation guides
4. Verify with the test above

Note: GPU setup is optional for this curriculum. All exercises can run on CPU; GPU just makes training faster.

---

## Testing Your Setup

Create a file `test_env.py`:

```python
import torch
import numpy as np
import pandas as pd

print("✓ All imports successful!")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"GPU: {torch.cuda.is_available()}")
```

Run: `python test_env.py`

If all imports succeed, your environment is ready.

---

## Troubleshooting Common Issues

### CUDA not recognized
- Restart your terminal after installing CUDA
- Verify `nvcc --version` works from command line
- Check NVIDIA GPU Driver is up to date: `nvidia-smi`

### PyTorch not using GPU
- Reinstall PyTorch with correct CUDA version
- Verify GPU compute capability is compatible (RTX 3060+ recommended)

### Dependency conflicts
- Start fresh: `deactivate` and `rm -rf markgpt_env`
- Create new virtual environment and reinstall

---

## Resources and Next Steps

- Official PyTorch installation: https://pytorch.org/
- CUDA Installation Guide: https://docs.nvidia.com/cuda/
- Troubleshooting Guide: See `docs/ENVIRONMENT_VARIABLES.md`

---

## Closing Reflection: Your Environment is Your Laboratory

Software engineers often say: "The environment is everything." For you, right now, this is true in two senses.

First, technically: a properly configured environment prevents mysterious errors and hours of "works on my machine" debugging. You've just built that foundation.

Second, philosophically: your physical environment matters too. Find a quiet space. Have water nearby. Configure your IDE the way you like it. Small things compound. If you enjoy spending time in your development environment, you will write better code and experiment more.

You have now completed Day 1. You understand the history of AI, what language models are, where MarkGPT fits, and how to set up the tools. 

Tomorrow (Day 2), you begin learning how computers *see* language. You will learn tokenization, encoding, and linguistics. But first, rest. Have completed the foundation.

*Next: Module 01, Day 2 — Language as Data (Lesson L02.1)*