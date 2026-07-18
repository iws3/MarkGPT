"""Unit tests for training utilities.

Tests gradient accumulation, early stopping, learning rate scheduling,
and checkpoint management.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from tempfile import TemporaryDirectory

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.training_utils import EarlyStopping
from training.checkpoint import CheckpointManager, save_training_config, load_training_config


class SimpleModel(nn.Module):
    """Simple test model."""
    
    def __init__(self, input_size=10, hidden_size=32, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestEarlyStopping:
    """Test suite for EarlyStopping callback."""
    
    def test_early_stopping_with_improvement(self):
        """Test that early stopping doesn't trigger with improvement."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01)
        
        # Losses improving each step
        losses = [1.0, 0.95, 0.90, 0.85, 0.80]
        for loss in losses:
            should_stop = early_stop(loss)
            assert not should_stop, "Should not stop when improving"
    
    def test_early_stopping_patience_threshold(self):
        """Test that early stopping triggers after patience exceeded."""
        early_stop = EarlyStopping(patience=2, min_delta=0.01)
        
        # Loss improves then plateaus
        losses = [1.0, 0.98, 0.95, 0.95, 0.95, 0.95]
        for i, loss in enumerate(losses):
            should_stop = early_stop(loss)
            if i < 4:  # First 4 steps should not stop
                assert not should_stop
            else:  # After patience exceeded, should stop
                assert should_stop
    
    def test_early_stopping_min_delta(self):
        """Test that min_delta threshold is respected."""
        early_stop = EarlyStopping(patience=2, min_delta=0.1)
        
        # Loss improves but less than min_delta
        losses = [1.0, 0.99, 0.98, 0.98]
        for loss in losses:
            should_stop = early_stop(loss)
            assert not should_stop, "Improvements below min_delta don't count"
    
    def test_early_stopping_reset_on_best(self):
        """Test that counter resets when new best found."""
        early_stop = EarlyStopping(patience=2, min_delta=0.01)
        
        # Loss plateaus, then improves
        losses = [1.0, 0.95, 0.95, 0.95, 0.90, 0.90]
        for i, loss in enumerate(losses):
            should_stop = early_stop(loss)
            if i == 5:  # After final improvement, should reset counter
                assert not should_stop


class TestCheckpointManager:
    """Test suite for CheckpointManager."""
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading model checkpoint."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir)
            
            # Create and save model
            model = SimpleModel()
            original_state = model.state_dict().copy()
            
            checkpoint_path = manager.save_checkpoint(
                model, step=100, metrics={"loss": 0.5}
            )
            assert checkpoint_path.exists()
            
            # Modify model
            for param in model.parameters():
                param.data.fill_(0)
            assert not torch.allclose(
                model.fc1.weight, 
                torch.tensor(original_state["fc1.weight"])
            )
            
            # Load checkpoint
            metadata = manager.load_checkpoint(model, checkpoint_path)
            assert metadata["step"] == 100
            assert metadata["metrics"]["loss"] == 0.5
            
            # Verify model restored
            for name, param in model.named_parameters():
                assert torch.allclose(
                    param.data,
                    torch.tensor(original_state[name])
                ), f"Parameter {name} not restored correctly"
    
    def test_save_with_optimizer(self):
        """Test saving checkpoint with optimizer state."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir)
            
            model = SimpleModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Run one optimization step to populate optimizer state
            loss = model(torch.randn(4, 10)).sum()
            loss.backward()
            optimizer.step()
            saved_opt_state = optimizer.state_dict()
            
            # Save checkpoint with optimizer
            checkpoint_path = manager.save_checkpoint(
                model, optimizer=optimizer, step=1
            )
            
            # Clear optimizer state
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            original_state = optimizer.state_dict()
            
            # Load and verify optimizer state restored
            manager.load_checkpoint(
                model, checkpoint_path, optimizer=optimizer
            )
            assert optimizer.state_dict() == saved_opt_state
    
    def test_best_checkpoint_save(self):
        """Test that best checkpoint is saved separately."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir)
            
            model = SimpleModel()
            
            # Save as best checkpoint
            manager.save_checkpoint(model, step=10, is_best=True)
            
            best_path = checkpoint_dir / "checkpoint_best.pt"
            assert best_path.exists()
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir)
            
            model = SimpleModel()
            
            # Save multiple checkpoints
            for step in [100, 200, 300]:
                manager.save_checkpoint(model, step=step)
            
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 3
            # Should be sorted by step
            assert "checkpoint_step_000100" in str(checkpoints[0])
            assert "checkpoint_step_000300" in str(checkpoints[2])
    
    def test_cleanup_old_checkpoints(self):
        """Test cleanup of old checkpoints."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir)
            
            model = SimpleModel()
            
            # Save 5 checkpoints
            for step in [100, 200, 300, 400, 500]:
                manager.save_checkpoint(model, step=step)
            
            # Keep only last 2
            manager.cleanup_old_checkpoints(keep_last=2)
            
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 2


class TestTrainingConfig:
    """Test suite for training config save/load."""
    
    def test_save_and_load_config(self):
        """Test saving and loading training config."""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            config = {
                "model_size": "small",
                "learning_rate": 0.001,
                "batch_size": 32,
                "num_epochs": 100
            }
            
            # Save config
            save_training_config(config, config_path)
            assert config_path.exists()
            
            # Load config
            loaded_config = load_training_config(config_path)
            assert loaded_config == config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
