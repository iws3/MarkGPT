# Environment Variables Reference

## Overview

This document lists all environment variables used in MarkGPT for configuration, debugging, and deployment.

## Python / PyTorch Configuration

### `PYTHONUNBUFFERED`
```bash
export PYTHONUNBUFFERED=1
```
**Purpose**: Stream Python output in real-time (important for Jupyter notebooks and Docker)
**Default**: 0
**Recommended**: 1 (for development); 0 (for production logging efficiency)

### `PYTORCH_CUDA_ALLOC_CONF`
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```
**Purpose**: Control GPU memory fragmentation
**Value**: max_split_size_mb in MB (256-1024)
**When to use**: OOM errors despite available memory
**Default**: Auto

### `CUDA_VISIBLE_DEVICES`
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```
**Purpose**: Select which GPUs to use
**Value**: Comma-separated GPU indices (0-indexed)
**Example**: 
  - `CUDA_VISIBLE_DEVICES=0` → Use only GPU 0
  - `CUDA_VISIBLE_DEVICES=1,3`: Use GPUs 1 and 3 (renumbered as 0, 1 in code)
**Default**: All available GPUs

### `CUDA_LAUNCH_BLOCKING`
```bash
export CUDA_LAUNCH_BLOCKING=1
```
**Purpose**: Synchronous GPU launches (easier debugging)
**Value**: 0 (async, default) or 1 (blocking)
**Performance Impact**: 5-20% slower with blocking
**When to use**: Debugging OOM errors, NaN losses

### `OMP_NUM_THREADS`
```bash
export OMP_NUM_THREADS=8
```
**Purpose**: Number of CPU threads for NumPy/PyTorch CPU operations
**Value**: Typically # of CPU cores (4, 8, 16, 32, etc.)
**Example**: On system with 32 cores: `OMP_NUM_THREADS=16`
**Default**: Auto-detect

### `TORCH_HOME`
```bash
export TORCH_HOME=/path/to/torch/cache
```
**Purpose**: Directory for PyTorch model cache
**Default**: ~/.cache/torch
**Use case**: Shared cluster where multiple users need same models

## MarkGPT Configuration

### `MARKGPT_DATA_DIR`
```bash
export MARKGPT_DATA_DIR=/data/markgpt
```
**Purpose**: Directory containing training/test data
**Structure**:
  ```
  /data/markgpt/
  ├── raw/
  │   ├── bible.txt
  │   └── banso.txt
  └── processed/
      ├── tokenizer_vocab.pkl
      └── train.bin
  ```
**Default**: ./data

### `MARKGPT_CHECKPOINT_DIR`
```bash
export MARKGPT_CHECKPOINT_DIR=/checkpoints/markgpt
```
**Purpose**: Directory for model checkpoints
**Default**: ./checkpoints
**Recommendation**: Use SSD for fast I/O; move to cold storage after training

### `MARKGPT_VOCAB_SIZE`
```bash
export MARKGPT_VOCAB_SIZE=10000
```
**Purpose**: BPE vocabulary size for tokenizer
**Value**: 1000-50000 (typical: 10000)
**Default**: 10000

### `MARKGPT_MODEL_SIZE`
```bash
export MARKGPT_MODEL_SIZE=small
```
**Purpose**: Pre-configured model variant
**Options**: nano, small, base, medium, large
**Configs**: See configs/ directory for YAML specs
**Default**: small

## Training Configuration

### `MARKGPT_BATCH_SIZE`
```bash
export MARKGPT_BATCH_SIZE=64
```
**Purpose**: Training batch size
**Value**: 8-512 (adjust to GPU memory)
**Memory guide**: Nano: 256, Small: 128, Base: 64, Medium: 32
**Default**: 64

### `MARKGPT_LEARNING_RATE`
```bash
export MARKGPT_LEARNING_RATE=0.0005
```
**Purpose**: Initial learning rate
**Value**: 1e-5 to 1e-3 (typical: 5e-4)
**Default**: 5e-4
**Fine-tuning**: Use 1-5x smaller (1e-4 to 2.5e-4)

### `MARKGPT_NUM_EPOCHS`
```bash
export MARKGPT_NUM_EPOCHS=10
```
**Purpose**: Number of training epochs
**Value**: 1-100
**Early stop**: Training may stop early if validation plateaus
**Default**: 10

### `MARKGPT_WARMUP_STEPS`
```bash
export MARKGPT_WARMUP_STEPS=1000
```
**Purpose**: Number of gradient updates before reaching peak learning rate
**Value**: Typically 5-10% of total updates
**Default**: 1000

### `MARKGPT_GRADIENT_ACCUMULATION_STEPS`
```bash
export MARKGPT_GRADIENT_ACCUMULATION_STEPS=4
```
**Purpose**: Accumulate gradients over N forward passes
**Effective batch size**: actual_batch_size × accumulation_steps
**Memory/speed tradeoff**: More accumulation = less memory, slower
**Default**: 1

### `MARKGPT_MAX_GRAD_NORM`
```bash
export MARKGPT_MAX_GRAD_NORM=1.0
```
**Purpose**: Maximum L2 norm for gradient clipping
**Value**: 0.5-10.0
**Default**: 1.0
**Prevent**: NaN losses from exploding gradients

### `MARKGPT_MIXED_PRECISION`
```bash
export MARKGPT_MIXED_PRECISION=true
```
**Purpose**: Enable automatic mixed precision (FP32 + FP16)
**Value**: true, false
**Speed improvement**: 2-3x faster on modern GPUs
**Default**: true

### `MARKGPT_USE_FLASH_ATTENTION`
```bash
export MARKGPT_USE_FLASH_ATTENTION=true
```
**Purpose**: Use NVIDIA Flash Attention for faster attention
**Value**: true, false
**Requirements**: CUDA compute capability >= 7.5 (A100, V100, etc.)
**Speed**: 2-4x faster attention
**Default**: true (auto-disabled if unavailable)

## Distributed Training

### `MASTER_ADDR`
```bash
export MASTER_ADDR=192.168.1.100
```
**Purpose**: IP of master node for distributed training
**Value**: Valid IP address
**Multi-node training**: Required for training across servers
**Single-node**: Set to 127.0.0.1 or localhost

### `MASTER_PORT`
```bash
export MASTER_PORT=1234
```
**Purpose**: Port for master node communication
**Value**: 1024-65535 (pick unused port)
**Default**: 1234

### `RANK`
```bash
export RANK=0
```
**Purpose**: Global rank of current process (0 = master)
**Value**: 0 to world_size-1
**Multi-node**: Set 0,1,2... for each machine

### `LOCAL_RANK`
```bash
export LOCAL_RANK=0
```
**Purpose**: Local rank on current machine
**Value**: 0 to num_gpus-1
**Use case**: Binding process to GPU (typically: `--gpu ${LOCAL_RANK}`)

### `WORLD_SIZE`
```bash
export WORLD_SIZE=8
```
**Purpose**: Total number of processes
**Value**: num_nodes × num_gpus_per_node
**Example**: 2 nodes × 4 GPUs = WORLD_SIZE=8

## Evaluation and Logging

### `WANDB_API_KEY`
```bash
export WANDB_API_KEY=your_key_here
```
**Purpose**: API key for Weights & Biases experiment tracking
**Where to find**: https://wandb.ai/settings/api
**Usage**: Logging metrics, gradients, system stats during training
**Default**: None (optional; training works without)

### `WANDB_PROJECT`
```bash
export WANDB_PROJECT=markgpt-training
```
**Purpose**: W&B project name for organizing runs
**Default**: markgpt
**Multiple projects**: Change per experiment

### `WANDB_ENTITY`
```bash
export WANDB_ENTITY=yourteam
```
**Purpose**: W&B team name for shared logging
**Value**: Your username or team name
**Default**: Personal account

### `MARKGPT_LOG_LEVEL`
```bash
export MARKGPT_LOG_LEVEL=DEBUG
```
**Purpose**: Python logging level
**Options**: DEBUG, INFO, WARNING, ERROR, CRITICAL
**Default**: INFO

### `MARKGPT_EVAL_EVERY_N_STEPS`
```bash
export MARKGPT_EVAL_EVERY_N_STEPS=1000
```
**Purpose**: Run validation every N training steps
**Value**: 100-5000
**Higher = faster training, less frequent checkpoints
**Default**: 1000

## Deployment

### `MARKGPT_API_PORT`
```bash
export MARKGPT_API_PORT=8000
```
**Purpose**: Port for inference API (Flask/FastAPI)
**Value**: 1024-65535
**Default**: 8000

### `MARKGPT_API_WORKERS`
```bash
export MARKGPT_API_WORKERS=4
```
**Purpose**: Number of worker processes for API
**Value**: 1-32 (consider: num_cpus / 2)
**Default**: 4

### `MARKGPT_MODEL_PRECISION`
```bash
export MARKGPT_MODEL_PRECISION=fp16
```
**Purpose**: Precision for inference (fp32, fp16, int8)
**fp16**: 2x faster, same quality on most hardware
**int8**: 4x smaller model, quantization artifacts
**fp32**: Baseline accuracy, slower
**Default**: fp32

### `MARKGPT_MAX_BATCH_SIZE_INFERENCE`
```bash
export MARKGPT_MAX_BATCH_SIZE_INFERENCE=64
```
**Purpose**: Maximum batch size for concurrent inference
**Memory limited**: Smaller for smaller GPUs
**Default**: 64 (A100); reduce to 16-32 for consumer GPUs

## Debug / Development

### `MARKGPT_DEBUG`
```bash
export MARKGPT_DEBUG=true
```
**Purpose**: Enable debug mode (verbose logging, error details)
**Value**: true, false
**Default**: false

### `MARKGPT_PROFILE`
```bash
export MARKGPT_PROFILE=true
```
**Purpose**: Enable profiling (timing, memory tracking)
**Value**: true, false
**Output**: results/profile_*.json
**Default**: false

### `MARKGPT_SEED`
```bash
export MARKGPT_SEED=42
```
**Purpose**: Random seed for reproducibility
**Value**: Any integer (42 is conventional)
**Sets**: np.random.seed, torch.manual_seed, etc.
**Default**: Auto

### `MARKGPT_DRY_RUN`
```bash
export MARKGPT_DRY_RUN=true
```
**Purpose**: Run 1 step of training without saving
**Use case**: Verify setup before launching full training
**Default**: false

## Quick Start Commands

### Development Setup
```bash
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export MARKGPT_MODEL_SIZE=small
export MARKGPT_BATCH_SIZE=32
export MARKGPT_LOG_LEVEL=DEBUG
python src/training/train.py
```

### Production Training
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export MARKGPT_MODEL_SIZE=base
export MARKGPT_BATCH_SIZE=128
export MARKGPT_MIXED_PRECISION=true
export WANDB_PROJECT=markgpt-prod
export MARKGPT_EVAL_EVERY_N_STEPS=500
python src/training/train.py --num-epochs 20
```

### Multi-GPU Training
```bash
export WORLD_SIZE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=1234
python -m torch.distributed.launch --nproc_per_node=8 src/training/train.py
```

### Inference API
```bash
export MARKGPT_API_PORT=8000
export MARKGPT_API_WORKERS=4
export MARKGPT_MODEL_PRECISION=fp16
python -m gunicorn --workers 4 --bind 0.0.0.0:8000 src/inference/api:app
```

---

**Last Updated**: 2024
**Applies to**: MarkGPT v1.0+
**Tested with**: PyTorch 2.0.1, CUDA 11.8, Python 3.10+
