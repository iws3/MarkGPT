"""
MarkGPT Training Loop
======================
A production-quality training script with everything a serious 
training run requires: logging, checkpointing, learning rate scheduling,
gradient clipping, and mixed precision.

This is the file you will run to train MarkGPT. By the time you use it
(Module 07), you will understand every line from first principles learned
in Modules 03–06.

Usage:
    python train.py --config configs/markgpt_small.yaml
    python train.py --config configs/markgpt_nano.yaml --quick-test
"""

import os
import time
import math
import yaml
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Our own modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from model.markgpt import MarkGPT, MarkGPTConfig, markgpt_nano, markgpt_small
from utils.data_loader import BibleDataset, create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """All hyperparameters governing the training process."""
    
    # ── Data ────────────────────────────────────────────────────────────────
    data_path: str = "data/processed/kjv_tokenized.bin"
    val_split: float = 0.05          # 5% of data reserved for validation
    
    # ── Model ────────────────────────────────────────────────────────────────
    model_size: str = "small"        # "nano", "small", or "base"
    vocab_size: int = 8000
    
    # ── Optimization ────────────────────────────────────────────────────────
    batch_size: int = 32             # Number of sequences per gradient step
    block_size: int = 512            # Context window (sequence length)
    max_iters: int = 50000           # Total training iterations
    eval_interval: int = 500         # How often to evaluate on validation set
    log_interval: int = 10           # How often to log training loss
    eval_iters: int = 200            # Number of batches to average for eval loss
    
    # ── Learning Rate Schedule ───────────────────────────────────────────────
    # We use a warmup + cosine decay schedule.
    # During warmup, LR grows linearly from 0 to max_lr.
    # After warmup, it decays following a cosine curve down to min_lr.
    # This is the "standard" schedule for Transformer training.
    learning_rate: float = 3e-4      # Peak learning rate (max_lr)
    min_lr: float = 3e-5             # Minimum LR at end of cosine decay (= 10% of peak)
    warmup_iters: int = 2000         # Number of steps to ramp up from 0 to learning_rate
    lr_decay_iters: int = 50000      # Steps at which cosine decay reaches min_lr
    
    # ── Regularization ───────────────────────────────────────────────────────
    weight_decay: float = 0.1        # L2 regularization coefficient (AdamW)
    grad_clip: float = 1.0           # Max gradient norm (clip if exceeded)
    
    # ── System ───────────────────────────────────────────────────────────────
    device: str = "auto"             # "cpu", "cuda", "mps", or "auto"
    compile: bool = False            # torch.compile() for ~20% speedup (requires PyTorch 2.0+)
    dtype: str = "bfloat16"          # "float32", "float16", or "bfloat16"
    
    # ── Checkpointing ────────────────────────────────────────────────────────
    out_dir: str = "checkpoints"
    checkpoint_interval: int = 1000  # Save checkpoint every N iterations
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    
    # ── Logging ──────────────────────────────────────────────────────────────
    wandb_log: bool = False          # Whether to log to Weights & Biases
    wandb_project: str = "markgpt"
    run_name: str = "markgpt-training"


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(it: int, cfg: TrainingConfig) -> float:
    """
    Compute the learning rate for iteration `it` using warmup + cosine decay.
    
    This three-phase schedule is used by most modern LLM training runs:
    
    Phase 1 — Linear warmup (0 to warmup_iters):
        LR grows from 0 to learning_rate.
        Why warmup? At initialization, model weights are random and gradients
        are unreliable. Starting with a small LR prevents early instability.
    
    Phase 2 — Cosine decay (warmup_iters to lr_decay_iters):
        LR follows a cosine curve from learning_rate down to min_lr.
        The smooth cosine shape (vs. linear decay) is a common heuristic
        that often improves final loss by a small but consistent margin.
    
    Phase 3 — Floor (after lr_decay_iters):
        LR stays at min_lr. We don't want LR to hit zero.
    """
    # Phase 1: Linear warmup
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    
    # Phase 3: After decay, hold at minimum
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    
    # Phase 2: Cosine decay between warmup and decay endpoint
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0 <= decay_ratio <= 1
    # cos(π * decay_ratio) goes from 1 to -1, so the coeff goes from 1 to 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: MarkGPT, val_loader: DataLoader, cfg: TrainingConfig) -> dict:
    """
    Estimate validation loss by averaging over eval_iters batches.
    
    We compute multiple batches and average because a single batch gives
    a noisy estimate. We're not computing the exact validation loss —
    just a reliable estimate of it.
    """
    model.eval()
    losses = []
    
    for i, (x, y) in enumerate(val_loader):
        if i >= cfg.eval_iters:
            break
        x, y = x.to(cfg.device), y.to(cfg.device)
        with torch.amp.autocast(device_type=cfg.device, dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    
    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)  # Perplexity = e^(cross-entropy loss)
    
    model.train()
    return {"val_loss": avg_loss, "val_perplexity": perplexity}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainingConfig):
    """
    The main training function. This runs the full training loop for MarkGPT.
    
    The training loop follows this pattern for each iteration:
      1. Sample a batch of (input, target) token sequences
      2. Forward pass: compute predictions and loss
      3. Backward pass: compute gradients via backpropagation
      4. Clip gradients (prevent exploding gradients)
      5. Optimizer step: update weights in the direction that reduces loss
      6. Repeat
    
    After every eval_interval iterations, we measure validation loss
    and save a checkpoint.
    """
    # ── Device Setup ─────────────────────────────────────────────────────────
    if cfg.device == "auto":
        if torch.cuda.is_available():
            cfg.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cfg.device = "mps"  # Apple Silicon GPU
        else:
            cfg.device = "cpu"
    
    logger.info(f"Training on: {cfg.device}")
    
    # ── Model Creation ────────────────────────────────────────────────────────
    if cfg.model_size == "nano":
        model = markgpt_nano(vocab_size=cfg.vocab_size)
    else:
        model = markgpt_small(vocab_size=cfg.vocab_size)
    
    model = model.to(cfg.device)
    
    # Optionally compile the model for speed (PyTorch 2.0+)
    if cfg.compile:
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model)
    
    # ── Optimizer ────────────────────────────────────────────────────────────
    # We use AdamW (Adam with decoupled weight decay).
    # Weight decay only applies to weight matrices, not biases or LayerNorm params.
    # This is a standard practice from the original GPT-2 paper.
    decay_params = [p for n, p in model.named_parameters() 
                    if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() 
                      if p.requires_grad and p.dim() < 2]
    
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': cfg.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ], lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    
    # ── Data Loading ──────────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(
        data_path=cfg.data_path,
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split
    )
    
    # ── Resume from Checkpoint ────────────────────────────────────────────────
    iter_num = 0
    best_val_loss = float('inf')
    
    if cfg.resume_from is not None:
        logger.info(f"Resuming from checkpoint: {cfg.resume_from}")
        checkpoint = torch.load(cfg.resume_from, map_location=cfg.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Resumed at iteration {iter_num}, best val loss: {best_val_loss:.4f}")
    
    # ── Training Loop ──────────────────────────────────────────────────────────
    os.makedirs(cfg.out_dir, exist_ok=True)
    model.train()
    
    scaler = torch.amp.GradScaler(enabled=(cfg.dtype == 'float16'))
    train_iter = iter(train_loader)
    
    t0 = time.time()
    local_iter = 0
    
    logger.info(f"Starting training: {model.count_parameters()/1e6:.1f}M params, "
                f"{cfg.max_iters} total iterations")
    
    while iter_num < cfg.max_iters:
        
        # ── Learning Rate Update ──────────────────────────────────────────────
        lr = get_lr(iter_num, cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # ── Evaluation ────────────────────────────────────────────────────────
        if iter_num % cfg.eval_interval == 0:
            metrics = evaluate(model, val_loader, cfg)
            logger.info(
                f"Iter {iter_num:6d} | "
                f"val_loss: {metrics['val_loss']:.4f} | "
                f"val_ppl: {metrics['val_perplexity']:.1f} | "
                f"lr: {lr:.2e}"
            )
            
            # Save best checkpoint
            if metrics['val_loss'] < best_val_loss:
                best_val_loss = metrics['val_loss']
                ckpt_path = os.path.join(cfg.out_dir, 'markgpt_best.pt')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': cfg,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }, ckpt_path)
                logger.info(f"  New best! Checkpoint saved to {ckpt_path}")
        
        # ── Regular Checkpoint ────────────────────────────────────────────────
        if iter_num % cfg.checkpoint_interval == 0 and iter_num > 0:
            ckpt_path = os.path.join(cfg.out_dir, f'markgpt_iter{iter_num}.pt')
            torch.save({
                'model': model.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }, ckpt_path)
        
        # ── Training Step ──────────────────────────────────────────────────────
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)  # Restart when dataset is exhausted
            x, y = next(train_iter)
        
        x, y = x.to(cfg.device), y.to(cfg.device)
        
        # Forward pass with automatic mixed precision
        # Using bfloat16 roughly halves memory usage and can double training speed
        # while having minimal impact on loss convergence.
        with torch.amp.autocast(device_type=cfg.device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        
        # Backward pass
        # scaler.scale(loss) scales the loss to prevent underflow in fp16.
        # In bfloat16 mode, scaling is effectively a no-op.
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # Gradient clipping: if the gradient norm exceeds grad_clip,
        # scale all gradients down so the norm equals grad_clip.
        # This prevents "exploding gradients" which can destabilize training.
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # ── Logging ────────────────────────────────────────────────────────────
        if iter_num % cfg.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = cfg.batch_size * cfg.block_size * cfg.log_interval / dt
            logger.info(
                f"Iter {iter_num:6d} | "
                f"loss: {loss.item():.4f} | "
                f"grad_norm: {grad_norm:.2f} | "
                f"lr: {lr:.2e} | "
                f"{tokens_per_sec:.0f} tok/s"
            )
            t0 = t1
        
        iter_num += 1
        local_iter += 1
    
    logger.info(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to {os.path.join(cfg.out_dir, 'markgpt_best.pt')}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MarkGPT")
    parser.add_argument("--config", type=str, default="configs/markgpt_small.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run a quick 100-step test to verify everything works")
    args = parser.parse_args()
    
    # Load config from YAML if provided
    cfg = TrainingConfig()
    if os.path.exists(args.config):
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    
    if args.quick_test:
        logger.info("Running quick test (100 iterations)...")
        cfg.max_iters = 100
        cfg.eval_interval = 50
        cfg.log_interval = 10
        cfg.model_size = "nano"
    
    train(cfg)
