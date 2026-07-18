"""Checkpoint management utilities for training resumption and inference.

Handles saving and loading of model state, optimizer state, and training
metadata for resuming training or deploying trained models.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import logging


logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints for training and inference.
    
    Attributes:
        checkpoint_dir (Path): Directory to save checkpoints
    """

    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory where checkpoints will be saved
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> Path:
        """Save model checkpoint with full training state.
        
        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            scheduler: Learning rate scheduler (optional)
            step: Current training step
            metrics: Dictionary of metrics to save
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Save to timestamped checkpoint
        checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_step_{step:06d}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

        return checkpoint_path

    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load model checkpoint and restore training state.
        
        Args:
            model: Model to load state into
            checkpoint_path: Path to checkpoint file
            optimizer: Optimizer to restore state to (optional)
            scheduler: Scheduler to restore state to (optional)
            strict: Whether to require exact state dict match
            
        Returns:
            Checkpoint metadata (step, metrics, etc.)
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Load model state
        model.load_state_dict(
            checkpoint["model_state_dict"], strict=strict
        )

        # Restore optimizer state
        if (
            optimizer is not None
            and "optimizer_state_dict" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler state
        if (
            scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(step {checkpoint['step']})"
        )

        # Return metadata for training loop
        return {
            "step": checkpoint["step"],
            "metrics": checkpoint.get("metrics", {}),
        }

    def list_checkpoints(self) -> list:
        """List all saved checkpoints, sorted by step.
        
        Returns:
            List of checkpoint paths
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt")
        )
        return checkpoints

    def cleanup_old_checkpoints(self, keep_last: int = 3) -> None:
        """Remove old checkpoints, keeping only the last N.
        
        Args:
            keep_last: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) > keep_last:
            to_remove = checkpoints[:-keep_last]
            for path in to_remove:
                path.unlink()
                logger.info(f"Removed old checkpoint: {path}")


def save_training_config(config: Dict[str, Any], path: Path) -> None:
    """Save training configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        path: Path to save config
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved training config to {path}")


def load_training_config(path: Path) -> Dict[str, Any]:
    """Load training configuration from JSON file.
    
    Args:
        path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(path, "r") as f:
        config = json.load(f)
    logger.info(f"Loaded training config from {path}")
    return config
