# Health Check & Diagnostics

Quick diagnostic tools to verify MarkGPT installation and setup.

## System Health Check

### Python Environment

```bash
# Check Python version
python --version  # Should be 3.10+

# Check installed packages
pip list | grep torch  # PyTorch
pip list | grep numpy  # NumPy
```

### GPU Diagnostics

```bash
# Check NVIDIA GPU
nvidia-smi

# PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# CUDA version
python -c "import torch; print(torch.version.cuda)"

# GPU memory
python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
```

### Model Configuration Test

```python
# Test basic model loading
from src.model.markgpt import MarkGPT, MarkGPTConfig
import torch

config = MarkGPTConfig(vocab_size=10000, d_model=256, num_layers=6)
model = MarkGPT(config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Test forward pass
x = torch.randint(0, 10000, (2, 128)).to(device)
output = model(x)
print(f"Model output shape: {output.shape}")  # Should be (2, 128, 10000)
```

## Common Issues & Fast Fixes

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Diagnosis**: PyTorch not installed
**Fix**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: `torch.cuda.is_available()` returns False

**Diagnosis**: CUDA not properly configured
**Steps**:
1. Check NVIDIA driver: `nvidia-smi`
2. Check PyTorch CUDA version: `python -c "import torch; print(torch.version.cuda)"`
3. Reinstall with correct CUDA version:
   ```bash
   pip install torch --upgrade --force-reinstall
   ```

### Issue: `CUDA out of memory` during training

**Diagnosis**: Batch too large for GPU
**Intermediate Fixes**:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export MARKGPT_BATCH_SIZE=32
```

**Permanent Fix**: Reduce batch size in config

### Issue: Model loads but inference is slow

**Diagnosis**: Running on CPU or no Flash Attention
**Check**:
```python
print(f"Model device: {next(model.parameters()).device}")
print(f"Flash Attention: {model.config.flash_attention}")
```

**Fix**: Use GPU and enable Flash Attention:
```bash
export CUDA_VISIBLE_DEVICES=0
export MARKGPT_USE_FLASH_ATTENTION=true
```

## Performance Baseline

Expected performance on standard hardware:

### Single GPU (NVIDIA A100, 40GB)

| Task | Time | Notes |
|------|------|-------|
| Model initialization | <1s | Loading weights |
| First inference (Nano) | 1ms | Forward pass only |
| Batch inference (64) | 200ms | 64 sequences, seq_len=1024 |
| Training epoch (Small, 1B tokens) | ~8 hours | On A100 with mixed precision |

### CPU Baseline

| Task | Time |
|------|------|
| Inference (Nano, CPU) | 50ms |
| Inference (Small, CPU) | 200ms |
| Training | Not recommended |

## Verification Checklist

- [ ] Python 3.10+ installed
- [ ] PyTorch 2.0+ installed
- [ ] CUDA available (optional, CPU works)
- [ ] Model initializes without error
- [ ] Forward pass completes (<1s)
- [ ] Tokenizer loads and encodes text
- [ ] Inference produces reasonable output
- [ ] Tests pass: `python -m pytest tests/`

## Logging and Debugging

### Enable Debug Logging

```bash
export MARKGPT_LOG_LEVEL=DEBUG
python src/training/train.py
```

### Profiling

```bash
export MARKGPT_PROFILE=true
python src/training/train.py
# Generates profile_*.json files
```

### Dry Run (No I/O)

```bash
export MARKGPT_DRY_RUN=true
python src/training/train.py
# Runs 1 batch without saving
```

## Support Resources

If diagnostics indicate issues:

1. **Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)** for common solutions
2. **Review [INSTALL.md](INSTALL.md)** for platform-specific setup
3. **Open GitHub Issue** with:
   - Error message
   - System info (OS, Python, PyTorch versions)
   - Output of diagnostics above
   - Reproducible minimal example

---

**Diagnostics Version**: 1.0
**Last Updated**: 2024
**Maintained by**: MarkGPT DevOps
