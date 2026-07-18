"""Learning rate finder to determine optimal LR (Smith, 2017)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


def find_learning_rate(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn,
    device: str = "cpu",
    start_lr: float = 1e-4,
    end_lr: float = 1e0,
    num_iterations: int = 100,
) -> tuple:
    """Run learning rate range test.
    
    Args:
        model: Model to test
        train_loader: Training dataloader
        loss_fn: Loss function
        device: Device to run on
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iterations: Number of batches to test
    
    Returns:
        Tuple of (lrs, losses)
    """
    model.to(device)
    model.train()
    
    # Exponentially spaced learning rates
    lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iterations)
    losses = []
    
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    
    for iteration, (x, y) in enumerate(train_loader):
        if iteration >= num_iterations:
            break
        
        x, y = x.to(device), y.to(device)
        
        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrs[iteration]
        
        # Forward/backward
        logits, _ = model(x, y)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return lrs[:len(losses)], losses
