"""Gradient flow monitoring for training diagnostics."""

import logging
import torch.nn as nn
from typing import Dict


logger = logging.getLogger(__name__)


class GradientMonitor:
    """Monitor gradient statistics during training.
    
    Tracks gradient norms per layer to diagnose:
    - Vanishing gradients (norms too small)
    - Exploding gradients (norms too large)
    - Dead neurons (zero gradients)
    """

    def __init__(self, model: nn.Module, log_frequency: int = 100):
        """Initialize monitor.
        
        Args:
            model: Model to monitor
            log_frequency: Log every N iterations
        """
        self.model = model
        self.log_frequency = log_frequency
        self.iteration = 0

    def log_gradients(self) -> Dict[str, float]:
        """Compute and log gradient statistics.
        
        Returns:
            Dictionary mapping layer names to gradient norms
        """
        grad_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms[name] = grad_norm
                
                if self.iteration % self.log_frequency == 0:
                    logger.info(f"Gradient norm {name}: {grad_norm:.6f}")
        
        self.iteration += 1
        return grad_norms
