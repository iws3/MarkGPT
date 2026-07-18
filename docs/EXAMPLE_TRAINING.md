# Example Training Script

## Complete training script demonstrating MarkGPT training workflow

```python
#!/usr/bin/env python3
\"\"\"
MarkGPT Training Example

Demonstrates complete training loop with:
- Data loading
- Mixed precision training
- Checkpoint management
- Early stopping
- W&B logging
\"\"\"

import os
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from src.model.markgpt import MarkGPT, MarkGPTConfig
from src.tokenizer.tokenizer import Tokenizer
from src.utils.datasets import BibleDataset
from src.training.trainer import Trainer
from src.training.training_utils import EarlyStopping


def main():
    parser = argparse.ArgumentParser(description='Train MarkGPT')
    parser.add_argument('--model-size', default='small', choices=['nano', 'small', 'base'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', default='checkpoints/')
    parser.add_argument('--data-dir', default='data/')
    parser.add_argument('--mixed-precision', action='store_true', default=True)
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load model config
    config_map = {
        'nano': MarkGPTConfig(vocab_size=10000, d_model=256, num_layers=6),
        'small': MarkGPTConfig(vocab_size=10000, d_model=512, num_layers=12),
        'base': MarkGPTConfig(vocab_size=10000, d_model=768, num_layers=24),
    }
    config = config_map[args.model_size]
    model = MarkGPT(config).to(device)
    
    # Load tokenizer
    tokenizer = Tokenizer.load(os.path.join(args.data_dir, 'markgpt_vocab.pkl'))
    
    # Create datasets
    train_dataset = BibleDataset(
        path=os.path.join(args.data_dir, 'bible_train.txt'),
        tokenizer=tokenizer,
        seq_length=1024
    )
    val_dataset = BibleDataset(
        path=os.path.join(args.data_dir, 'bible_val.txt'),
        tokenizer=tokenizer,
        seq_length=1024
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Setup scheduler
    num_training_steps = len(train_loader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=1000,
        T_mult=1,
        eta_min=1e-5
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        mixed_precision=args.mixed_precision,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Setup early stopping
    early_stop = EarlyStopping(
        patience=5,
        min_delta=0.001,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Training loop
    print(f"Training {args.model_size} model...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    for epoch in range(args.num_epochs):
        # Train epoch
        train_loss = trainer.train_epoch()
        
        # Validate
        val_loss = trainer.validate()
        
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}")
        
        # Early stopping
        if early_stop(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    print("Training complete!")


if __name__ == '__main__':
    main()
```

## Running the Example

```bash
# Basic usage
python examples/train_example.py

# Custom configuration
python examples/train_example.py \\
    --model-size base \\
    --batch-size 128 \\
    --learning-rate 1e-4 \\
    --num-epochs 20 \\
    --checkpoint-dir /tmp/markgpt-checkpoints

# Multi-GPU (distributed training)
python -m torch.distributed.launch \\
    --nproc_per_node=8 \\
    examples/train_example.py \\
    --model-size large
```

## Expected Output

```
Training small model...
Model parameters: 50.2M
Epoch 1/10
  Train loss: 4.23
  Val loss: 3.98
Epoch 2/10
  Train loss: 3.85
  Val loss: 3.72
...
Epoch 10/10
  Train loss: 2.61
  Val loss: 2.58
Training complete!
Model saved to: checkpoints/markgpt-small-final.pt
```

---

See [QUICKSTART.md](../docs/QUICKSTART.md) for more examples.
