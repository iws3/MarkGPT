# Troubleshooting Guide

## Common Issues and Solutions

### Setup & Installation

#### "ModuleNotFoundError: No module named torch"

**Cause:** PyTorch not installed

**Solution:**
```bash
# Reinstall PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.__version__)"
```

#### "No module named 'src'"

**Cause:** Running from wrong directory

**Solution:**
```bash
# ❌ Wrong
cd src/
python training/train.py

# ✓ Correct
cd MarkGPT-LLM-Curriculum/
python src/training/train.py
```

---

### Training Issues

#### "CUDA out of memory"

**Cause:** Model/batch too large for GPU

**Solutions (try in order):**

```python
# 1. Reduce batch size
python train.py --batch_size 8  # instead of 32

# 2. Clear GPU cache
torch.cuda.empty_cache()

# 3. Use gradient accumulation (simulate larger batch)
python train.py --gradient_accumulation_steps 4

# 4. Enable mixed precision (float16)
python train.py --mixed_precision

# 5. Use smaller model
python train.py --config configs/markgpt_nano.yaml
```

**If still failing:**
```bash
# Use CPU (slow but works)
CUDA_VISIBLE_DEVICES="" python train.py
```

---

#### "Loss becomes NaN"

**Cause:** Gradient explosion or numerical instability

**Solutions:**

```python
# 1. Lower learning rate
python train.py --learning_rate 0.0001  # instead of 0.001

# 2. Enable gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Check for invalid data
assert not torch.isnan(batch['input_ids']).any()
assert batch['input_ids'].max() < vocab_size
```

---

#### "Loss plateaus (stops decreasing)"

**Cause:** Learning rate too low or model capacity issues

**Solutions:**

```python
# 1. Increase learning rate
python train.py --learning_rate 0.01  # try 10x higher

# 2. Use learning rate scheduler
python train.py --lr_scheduler cosine

# 3. Increase model size
python train.py --config configs/markgpt_small.yaml

# 4. Train longer
python train.py --num_epochs 10  # instead of 3
```

---

#### "Validation loss worse than training loss"

**Cause:** Overfitting or validation set mismatch

**Solutions:**

```python
# 1. Add regularization (dropout)
model = MarkGPT(config, dropout_rate=0.1)

# 2. Reduce model size
python train.py --config configs/markgpt_nano.yaml

# 3. Increase training data
# Add more Bible verses to data/raw/

# 4. Use early stopping
python train.py --early_stopping_patience 3
```

---

### Data Issues

#### "FileNotFoundError: data/bible_tokens.bin"

**Cause:** Data not preprocessed

**Solution:**
```bash
# Download and preprocess data
python scripts/download_data.py
python scripts/preprocess_bible.py
```

---

#### "Tokenizer vocabulary size mismatch"

**Cause:** Model vocab_size doesn't match tokenizer

**Solution:**
```python
# Verify they match
model_vocab_size = model.config.vocab_size
tokenizer_vocab_size = tokenizer.get_vocab_size()

assert model_vocab_size == tokenizer_vocab_size, \
    f"Mismatch: {model_vocab_size} vs {tokenizer_vocab_size}"
```

---

#### "UnicodeDecodeError: 'utf-8' codec can't decode byte..."

**Cause:** File encoding issue (often with Banso text)

**Solution:**
```python
# Open with correct encoding
with open("file.txt", encoding="utf-8-sig") as f:  # -sig = BOM
    text = f.read()

# For Banso: ensure file was saved as UTF-8
# In VS Code: Bottom right → "Change End of Line Sequence" → UTF-8
```

---

### Inference Issues

#### "Model generates same token repeatedly"

**Cause:** Model underfitting or broken beam search

**Solution:**

```python
# 1. Check model was trained (not random)
# Verify loss decreased during training

# 2. Increase temperature
generated = model.generate(
    tokens,
    temperature=0.7,  # instead of 0.0 (deterministic)
    top_p=0.9          # nucleus sampling
)

# 3. Check tokenizer
tokens = tokenizer.encode("In the beginning")
print(tokens)  # Should not be all 0s or all same value
```

---

#### "Generation is gibberish"

**Cause:** Model undertrained or tokenizer misaligned

**Solution:**

```python
# 1. Train longer
# Increase num_epochs or num_steps in config

# 2. Verify model loaded correctly
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()

# 3. Check tokenizer matches training
# Must use same tokenizer as training

# 4. Lower temperature (more deterministic)
model.generate(tokens, temperature=0.1)
```

---

### Jupyter/Notebook Issues

#### "Kernel keeps dying"

**Cause:** Out of memory in notebook

**Solution:**

```python
# Restart kernel (⚠️ loses all variables)
# In Jupyter: Kernel → Restart Kernel

# Or manually clear
del model  # Delete large objects
torch.cuda.empty_cache()  # Clear GPU

# Reduce batch size
batch_size = 4  # instead of 32
```

---

#### "ModuleNotFoundError in Jupyter but works in terminal"

**Cause:** Wrong Python environment or sys.path issue

**Solution:**

```python
# In first Jupyter cell:
import sys
sys.path.insert(0, '/path/to/MarkGPT-LLM-Curriculum')

import src
```

---

### Git/Version Control

#### "Git merge conflicts"

**Cause:** Multiple people editing same file

**Solution:**
```bash
# View conflict
git status

# Manually resolve in file (look for <<<< and >>>>)
# Then:
git add resolved_file.py
git commit -m "resolve merge conflict"
```

---

#### "Accidental commit to wrong branch"

**Solution:**
```bash
# Create new branch from current state
git branch feature/my-work

# Reset main to previous commit
git reset --hard HEAD~1

# Switch to feature branch
git checkout feature/my-work
```

---

### Performance Issues

#### "Training very slow"

**Diagnosis:**

```python
# 1. Check if using GPU
print(next(model.parameters()).device)  # Should show cuda:0, not cpu

# 2. Profile training
import cProfile
cProfile.run('train_epoch()')  # Shows time-consuming functions

# 3. Check batch size
# Increase batch_size to saturate GPU (until OOM)

# 4. Check I/O
# Verify dataloader not bottleneck:
for batch in dataloader:
    print(f"Batch loaded: {batch['input_ids'].shape}")
```

**Solutions:**

```python
# 1. Use GPU
model = model.to('cuda:0')
batch = batch.to('cuda:0')

# 2. Increase batch size (until OOM)
# Larger batch = better GPU utilization

# 3. Use mixed precision (float16)
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model(batch)
scaler.scale(loss).backward()

# 4. Profile and optimize hot spots
```

---

## Getting Help

1. **Check this guide** (you're doing it!)
2. **Search GitHub issues**: https://github.com/yourusername/MarkGPT-LLM-Curriculum/issues
3. **Post question** with:
   - Error message and traceback
   - Your environment (Python 3.10? GPU model?)
   - Minimal reproducible example
4. **Join Discord**: [To be created]

---

## Still Stuck?

Email: support@markgpt.org (or see README for actual contact)
