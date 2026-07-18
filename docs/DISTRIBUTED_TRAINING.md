# Distributed Training: Multi-GPU Setup

## Why Distributed Training?

Single GPU limits:
- V100: 32 GB memory
- 200M model in fp32: 800 MB
- Gradients + optimizer states: ~3.2 GB per parameter
- **Total for 200M params**: ~3.2 GB (fits!)

But with batch_size=32:
- Activations: 32 × seq_len × d_model × 4 bytes
- seq_len=2048, d_model=512 → 64 MB per batch
- **Practical**: 3-4 GPUs needed for reasonable throughput

---

## Data Parallel (Simplest)

### Concept

```
Model copy on GPU 0
Model copy on GPU 1
Model copy on GPU 2

Batch split: [0:16 samples] → GPU 0
             [16:32] → GPU 1
             [32:48] → GPU 2

Each GPU:
  1. Forward pass
  2. Backward pass
  3. Compute gradients

Aggregate gradients → Average → Update all models

Synchronization point ↓

All GPUs updated ✓
```

### Implementation (PyTorch)

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

# Single GPU version
model = TransformerLM(config)
model = model.to('cuda:0')

# Make it data parallel (simplest)
model = DataParallel(model, device_ids=[0, 1, 2, 3])

# Training loop (same as single GPU!)
for epoch in range(epochs):
    for batch in loader:
        input_ids = batch['input_ids'].to('cuda')
        targets = batch['targets'].to('cuda')
        
        outputs = model(input_ids)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Memory distribution:
# GPU 0: Model + 25% of batch
# GPU 1: Model + 25% of batch
# GPU 2: Model + 25% of batch
# GPU 3: Model + 25% of batch
```

### Issues with DataParallel

- Redundant model copies (4×8GB = 32GB for 4 V100s)
- GPU 0 does gradient aggregation (bottleneck)
- Slow for cross-GPU communication

---

## DistributedDataParallel (Better)

Modern approach:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

# Must initialize before creating model
dist.init_process_group(backend='nccl')  # NCCL for GPUs

# Create model
model = TransformerLM(config).to(local_rank)

# Wrap with DDP
model = DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank
)

# Use DistributedSampler (splits data across ranks)
sampler = DistributedSampler(train_dataset, shuffle=True)
loader = DataLoader(train_dataset, sampler=sampler, batch_size=32)

# Training
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # Reshuffle data across ranks
    
    for batch in loader:
        input_ids = batch['input_ids'].to(local_rank)
        targets = batch['targets'].to(local_rank)
        
        outputs = model(input_ids)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

### Launch Script

```bash
# Launch on 4 GPUs
torchrun --nproc_per_node=4 train.py

# Or manually with torch.distributed.launch (deprecated but works)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    --config markgpt_small.yaml
```

### Inside train.py

```python
import torch.distributed as dist

def setup(rank, world_size):
    """Initialize distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

# In main
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    
    # Each GPU gets a process
    torch.multiprocessing.spawn(
        train_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

def train_worker(rank, world_size, config):
    setup(rank, world_size)
    
    # Training code here
    model = TransformerLM(config).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    
    # ... train ...
    
    cleanup()
```

---

## Pipeline Parallelism (Large Models)

For models that don't fit on single GPU, split across GPUs:

```
Layer 0-6: GPU 0   (first half of transformer)
Layer 7-12: GPU 1  (second half of transformer)

Forward pass (pipeline):
  ├─ GPU 0 processes seq
  ├─ Output → GPU 1
  ├─ GPU 1 processes
  └─ Output to loss
```

### GPipe (Gradient Checkpointing + Pipeline)

```python
from torch.nn.parallel import SequentialDataParallel

# Define model as sequential blocks
model = nn.Sequential(
    *[TransformerBlock(config) for _ in range(24)]
)

# Partition across 4 GPUs (6 blocks each)
model = SequentialDataParallel(
    model,
    chunks=4  # Number of pipeline stages
)

# Forward pass automatically pipelines
output = model(input_ids)
```

**Trade-off**: Some GPUs idle while others work (pipeline bubble), but enables training of 500M+ on 4×GPUs.

---

## Checkpointing Strategy for MarkGPT

### Multi-GPU MarkGPT (Recommended)

```
Setup: 8 × A100 (80GB each)

DDP Configuration:
  - Distributed across 8 GPUs
  - Batch size per GPU: 16
  - Total batch size: 128
  - Gradient accumulation: 4 steps
    (Effective: 512)

Expected throughput:
  - ~5000 tokens/sec per GPU
  - ~40,000 tokens/sec total
  - 10B tokens → ~278 hours (~11.5 days)
  - Cost: $20,000 on Lambda/Vast (rough)
```

### Scaling Script

```python
class DistributedTrainer:
    def __init__(self, config):
        self.config = config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
    
    def train(self):
        model = TransformerLM(self.config).to(self.rank)
        model = DistributedDataParallel(model, device_ids=[self.rank])
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
        
        for epoch in range(self.config.epochs):
            for step, batch in enumerate(self.train_loader):
                # Forward
                logits = model(batch['input_ids'].to(self.rank))
                loss = criterion(logits, batch['targets'].to(self.rank))
                
                # Backward
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Logging (only rank 0)
                if self.rank == 0:
                    if step % 100 == 0:
                        print(f"Step {step}: Loss = {loss.item():.4f}")
            
            # Synchronize across ranks before next epoch
            dist.barrier()

# Main execution
if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    trainer = DistributedTrainer(config)
    trainer.train()
    dist.destroy_process_group()
```

---

## Multi-Node Training

For 16-32 GPUs (across machines):

```python
# Set environment variables (typically via launcher)
os.environ['MASTER_ADDR'] = '192.168.1.100'  # One master node
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = os.environ.get('SLURM_PROCID', 0)  # From SLURM/PBS
os.environ['WORLD_SIZE'] = os.environ.get('SLURM_NTASKS', 1)

# Initialize (same as single-node)
dist.init_process_group(backend='nccl')
```

### SLURM Script (HPC Cluster)

```bash
#!/bin/bash
#SBATCH --job-name=markgpt_training
#SBATCH --nodes=4                    # 4 physical machines
#SBATCH --ntasks=32                  # 8 GPUs per node × 4 = 32
#SBATCH --gres=gpu:8                 # 8 GPUs per node
#SBATCH --time=10:00:00

srun python train.py --config markgpt_base.yaml
```

---

## Monitoring Distributed Training

```python
def log_training_metrics(metrics, epoch, step, rank=0):
    """Log metrics (only from rank 0 to avoid duplicates)."""
    
    if rank == 0:
        print(f"Epoch {epoch}, Step {step}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Learning Rate: {metrics['lr']:.2e}")
        print(f"  Throughput: {metrics['tokens_per_sec']:.0f} tokens/sec")

def synchronize_gradients(model, world_size):
    """Verify gradients synchronized (debugging)."""
    for param in model.parameters():
        if param.grad is not None:
            # Sum gradients across all ranks
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)

# In training loop
if step % log_interval == 0:
    metrics = {
        'loss': loss.item(),
        'lr': optimizer.param_groups[0]['lr'],
        'tokens_per_sec': tokens_seen / elapsed_time,
    }
    log_training_metrics(metrics, epoch, step, rank)
```

---

## Troubleshooting DDP

### Issue: Ranks Hang (Deadlock)

```python
# ✓ Correct: All ranks must hit barrier
dist.barrier()  # All wait here

# ✗ Wrong: Only rank 0 computes
if rank == 0:
    dist.barrier()  # Rank 0 waits forever (others never arrive)
```

### Issue: Gradient Mismatch

```python
# Debug: Check gradient norms are similar
grad_norms = []
for param in model.parameters():
    if param.grad is not None:
        grad_norms.append(param.grad.norm().item())

grad_mean = np.mean(grad_norms)
grad_std = np.std(grad_norms)

# Should be similar across ranks (within 5%)
if grad_std / grad_mean > 0.05:
    print("⚠️ Warning: Gradient divergence detected")
```

---

**Distributed Training v1.0**
**Last Updated**: 2024
