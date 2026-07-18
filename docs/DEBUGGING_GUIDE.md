# Debugging & Profiling Guide

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB.
```

**Solutions** (in order):

1. **Reduce Batch Size**
```python
# Before
batch_size = 128
model = load_model()

# After
batch_size = 32  # Reduce by 2-4x
# or
batch_size = max(1, batch_size // 2)  # Adaptive reduction
```

2. **Enable Gradient Checkpointing**
```python
model.gradient_checkpointing_enable()
# Trades compute for memory (saves ~50% memory, costs ~20% speed)
```

3. **Use Mixed Precision**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():  # FP16 autocast
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

4. **Enable Memory-Efficient Attention**
```python
# For Transformers library
model = AutoModel.from_pretrained(model_id, attention_implementation="flash_attention_2")

# Or manual attention implementation
from torch.nn.functional import scaled_dot_product_attention
```

**Debug Script**:
```python
def diagnose_oom():
    """Identify memory bottleneck."""
    
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
    print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
    print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.1f} GB")
    
    # Empty cache and retry
    torch.cuda.empty_cache()
    print("\nCache cleared. Retrying...")
```

### Issue 2: NaN Loss During Training

**Symptoms**:
```
loss = nan
```

**Root Causes & Solutions**:

1. **Learning Rate Too High**
```python
# Before: lr=0.1 → exploding gradients
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# After: use learning rate warmup
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=num_steps)

for step, batch in enumerate(train_loader):
    loss = model(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Gradually increase then decrease
```

2. **Unstable Initial State**
```python
# Initialize weights carefully
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)

model.apply(init_weights)
```

3. **Gradient Explosion Check**
```python
def check_gradients(model):
    """Detect gradient scaling issues."""
    
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    
    if total_norm > 100:
        print(f"⚠️  Warning: Large gradients detected ({total_norm:.2f})")
    
    return total_norm

for batch in train_loader:
    loss = model(batch)
    loss.backward()
    norm = check_gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

### Issue 3: Model Not Learning

**Symptoms**:
- Loss plateaus at random value
- Validation loss never improves
- Metrics don't change epoch-to-epoch

**Diagnostics**:

```python
def diagnose_learning_failure(model, train_loader, val_loader):
    """Comprehensive learning diagnostics."""
    
    print("=" * 50)
    print("LEARNING FAILURE DIAGNOSTICS")
    print("=" * 50)
    
    # 1. Check model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total Parameters: {total_params:,}")
    
    # 2. Check data
    batch = next(iter(train_loader))
    print(f"✓ Batch Shape: {batch['input_ids'].shape}")
    print(f"✓ Label Range: [{batch['labels'].min()}, {batch['labels'].max()}]")
    
    # 3. Check model output
    with torch.no_grad():
        out = model(batch['input_ids'])
        print(f"✓ Output Range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"✓ Output NaN: {torch.isnan(out).any()}")
    
    # 4. Check loss computation
    loss = criterion(out.view(-1, vocab_size), batch['labels'].view(-1))
    print(f"✓ Loss Value: {loss.item():.4f}")
    print(f"✓ Loss NaN: {torch.isnan(loss)}")
    
    # 5. Check gradients
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"✓ Gradient Range: [{min(grad_norms):.4f}, {max(grad_norms):.4f}]")
    print(f"✓ Zero gradients: {sum(1 for g in grad_norms if g == 0)}")
    
    print("=" * 50)

diagnose_learning_failure(model, train_loader, val_loader)
```

## Profiling

### Memory Profiling

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    for batch in train_loader:
        model(batch)

# Print top memory consumers
print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
```

### Speed Profiling

```python
import time

def timeit(func, *args, **kwargs):
    """Simple timing utility."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

# Usage
_, inference_time = timeit(model, batch)
print(f"Inference time: {inference_time*1000:.2f}ms")

# Batch timing
times = []
for batch in train_loader:
    _, t = timeit(model, batch)
    times.append(t)

avg_time = sum(times) / len(times)
print(f"Avg batch time: {avg_time*1000:.2f}ms")
print(f"Throughput: {32/avg_time:.0f} examples/sec")  # assuming batch_size=32
```

### PyTorch Profiler

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(train_loader):
        model.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        prof.step()

print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=20))
```

## Health Checks

### Pre-Training Checklist

```python
def pre_training_checklist(model, train_loader, config):
    """Verify setup before training."""
    
    checks = {}
    
    # 1. Device availability
    checks['device'] = torch.cuda.is_available()
    
    # 2. Model parameters
    checks['frozen_params'] = sum(1 for p in model.parameters() if not p.requires_grad)
    checks['total_params'] = sum(p.numel() for p in model.parameters())
    
    # 3. Data
    batch = next(iter(train_loader))
    checks['batch_size'] = batch['input_ids'].shape[0]
    checks['seq_length'] = batch['input_ids'].shape[1]
    
    # 4. Forward pass
    try:
        with torch.no_grad():
            out = model(batch['input_ids'])
        checks['forward_pass'] = True
        checks['output_shape'] = tuple(out.shape)
    except Exception as e:
        checks['forward_pass'] = False
        checks['error'] = str(e)
    
    # 5. Loss computation
    try:
        loss = criterion(out.view(-1, vocab_size), batch['labels'].view(-1))
        checks['loss_computation'] = True
        checks['loss_value'] = loss.item()
    except Exception as e:
        checks['loss_computation'] = False
        checks['error'] = str(e)
    
    # 6. Backward pass
    try:
        loss.backward()
        checks['backward_pass'] = True
    except Exception as e:
        checks['backward_pass'] = False
        checks['error'] = str(e)
    
    # Print results
    print("\n" + "="*50)
    print("PRE-TRAINING CHECKLIST")
    print("="*50)
    for check, result in checks.items():
        symbol = "✓" if result else "✗"
        print(f"{symbol} {check}: {result}")
    print("="*50 + "\n")
    
    return all(checks.values())

# Run before training
assert pre_training_checklist(model, train_loader, config)
```

---

**Guide Version**: 1.0
**Last Updated**: 2024
