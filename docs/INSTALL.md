# Installation Guide

## Overview

Comprehensive installation instructions for MarkGPT on various platforms and configurations.

## System Requirements

### Minimum
- **OS**: Linux (Ubuntu 20.04+), macOS (10.14+), Windows 10+
- **Python**: 3.10 or higher
- **RAM**: 8GB (for Small model), 16GB+ for Base/Medium
- **Disk**: 20GB for data + models

### Recommended
- **OS**: Linux (Ubuntu 22.04)
- **Python**: 3.11+
- **RAM**: 32GB
- **Disk**: 100GB SSD for training
- **GPU**: NVIDIA GPU with compute capability ≥ 7.5 (V100, A100, RTX 30xx/40xx series)
- **CUDA**: 11.8+ (for NVIDIA GPUs)

## Installation Methods

### Method 1: Pip (Quickest)

```bash
# Clone repository
git clone https://github.com/iwstechnical/MarkGPT-LLM-Curriculum.git
cd MarkGPT-LLM-Curriculum

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Method 2: Conda (Recommended for Data Science)

```bash
# Create conda environment
conda create -n markgpt python=3.11 pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch -c nvidia

# Activate environment
conda activate markgpt

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import torch; torch.cuda.is_available()"
```

### Method 3: Docker

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-runtime-ubuntu22.04

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python"]
```

```bash
# Build image
docker build -t markgpt:latest .

# Run container
docker run --gpus all -v $(pwd)/data:/workspace/data markgpt:latest src/training/train.py
```

## Platform-Specific Setup

### Linux (Ubuntu 22.04)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3.11 python3-pip python3-venv build-essential

# GPU support (if using NVIDIA)
sudo apt-get install nvidia-driver-525 nvidia-cuda-toolkit-11.8

# Clone and setup
git clone https://github.com/iwstechnical/MarkGPT-LLM-Curriculum.git
cd MarkGPT-LLM-Curriculum
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### macOS (M1/M2 Apple Silicon)

```bash
# Homebrew
brew install python@3.11 git

# Virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies (PyTorch with MPS support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Note**: Apple Silicon benefits from MPS (Metal Performance Shaders). MarkGPT auto-detects and uses MPS.

### Windows

```powershell
# PowerShell (as Administrator)
python -m venv venv
venv\Scripts\Activate.ps1

# Install PyTorch for CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

**Note**: Ensure CUDA Toolkit compatible with your GPU driver.

## GPU Setup

### NVIDIA CUDA Setup

```bash
# Check GPU
nvidia-smi

# Install CUDA (if not present)
# Follow: https://developer.nvidia.com/cuda-11-8-0-download-archive

# Verify CUDA availability in PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### AMD ROCm (for AMD GPUs)

```bash
# Install ROCm
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
sudo apt install rocm-dkms

# Install PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

### Apple Metal Performance Shaders

Auto-detected if using PyTorch 1.12+.

```python
# In your code: automatically uses MPS on macOS
# No special setup needed beyond pip install torch
```

## Verification

After installation, run the verification script:

```bash
python scripts/verify_setup.py
```

**Expected output**:
```
✓ Python 3.11.0
✓ PyTorch 2.0.1
✓ CUDA available: True (v11.8)
✓ GPUs detected: 1 (NVIDIA A100)
✓ All dependencies installed
✓ Setup verified successfully!
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
pip install torch --upgrade
# or
conda install pytorch::pytorch -c pytorch
```

### Issue: `CUDA out of memory`

**Solution**:
```bash
# In your code or environment:
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export MARKGPT_BATCH_SIZE=32  # Reduce batch size
python src/training/train.py
```

### Issue: `ImportError: libcublas.so.11: cannot open shared object`

**Solution**: Ensure CUDA path is set:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

### Issue: `torch.cuda.is_available() returns False`

**Solution**:
1. Verify GPU installed: `nvidia-smi`
2. Check CUDA versions match: `nvcc --version` vs PyTorch requirement
3. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip install torch --upgrade --force-reinstall --no-cache-dir
   ```

## Development Setup (Optional)

For contributing to MarkGPT:

```bash
# Install with development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting and type checking
make lint
make typecheck

# Run tests
make test
```

## Docker Compose (Multi-Container Setup)

```yaml
version: '3.9'

services:
  markgpt:
    image: markgpt:latest
    volumes:
      - ./data:/workspace/data
      - ./checkpoints:/workspace/checkpoints
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - WANDB_API_KEY=${WANDB_API_KEY}
    command: python src/training/train.py --num-epochs 10

  tensorboard:
    image: tensorflow/tensorflow:latest
    ports:
      - "6006:6006"
    volumes:
      - ./checkpoints:/workspace/checkpoints
    command: tensorboard --logdir=/workspace/checkpoints --host=0.0.0.0
```

```bash
docker-compose up -d
```

## Next Steps

After successful installation:

1. **Download data**: `python scripts/download_data.py`
2. **Preprocess data**: `python scripts/preprocess_bible.py`
3. **Train model**: `python src/training/train.py --model-name markgpt-nano`
4. **Inference**: `python -m src.inference.api`

See [QUICKSTART.md](QUICKSTART.md) for next steps.

---

**Last Updated**: 2024
**Latest Version**: MarkGPT v1.0
**Tested**: Ubuntu 22.04, macOS 13, Windows 11
