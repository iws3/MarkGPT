"""Training utilities for MarkGPT model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional


def gradient_accumulation_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    loss_fn,
    accumulation_steps: int,
    step: int,
) -> float:
    """Compute loss for gradient accumulation.
    
    Args:
        model: Model to train
        batch: (input_ids, target_ids)
        loss_fn: Loss function
        accumulation_steps: Number of steps to accumulate
        step: Current training step
    
    Returns:
        Loss value
    """
    x, y = batch
    logits, _ = model(x, y)
    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
    
    # Scale loss by accumulation steps
    loss = loss / accumulation_steps
    loss.backward()
    
    return loss.item()


class EarlyStopping:
    """Early stopping callback for training."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        """Initialize early stopping.
        
        Args:
            patience: Number of iterations to wait before stopping
            min_delta: Minimum improvement threshold
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss: float) -> bool:
        """Check if should stop training.
        
        Args:
            val_loss: Current validation loss
        
        Returns:
            True if should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
