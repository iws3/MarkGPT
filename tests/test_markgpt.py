"""Tests for MarkGPT model forward pass and basic functionality."""

import torch
import pytest
import math

from src.model.markgpt import MarkGPT, MarkGPTConfig
from src.utils.model_factory import markgpt_nano, markgpt_small


class TestMarkGPTForwardPass:
    """Test model forward pass functionality."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return markgpt_nano()

    @pytest.fixture
    def batch(self):
        """Create test batch."""
        batch_size = 4
        seq_len = 64
        return torch.randint(0, 8000, (batch_size, seq_len))

    def test_forward_shape(self, model, batch):
        """Test output shape matches input batch structure."""
        logits, loss = model(batch)
        
        assert logits.shape == (batch.shape[0], batch.shape[1], model.config.vocab_size)
        assert loss is None  # No targets provided

    def test_forward_with_targets(self, model, batch):
        """Test loss computation when targets provided."""
        targets = batch.clone()
        logits, loss = model(batch, targets)
        
        assert loss is not None
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_loss_sanity_check(self, model, batch):
        """Test that initial loss is approximately log(vocab_size)."""
        targets = batch.clone()
        _, loss = model(batch, targets)
        
        # At random initialization, cross-entropy should be ~log(vocab_size)
        expected_loss = math.log(model.config.vocab_size)
        ratio = loss.item() / expected_loss
        
        # Allow some variance
        assert 0.5 < ratio < 2.0, f"Loss ratio {ratio} not near 1.0"

    def test_causal_mask(self, model, batch):
        """Test that attention respects causal masking.
        
        Changing tokens at position i+1 and beyond should not affect
        the output at position i.
        """
        model.eval()
        
        with torch.no_grad():
            # Generate logits normally
            logits1, _ = model(batch)
            
            # Modify tokens at positions after i 
            batch_modified = batch.clone()
            batch_modified[:, 32:] = 0  # Zero out second half
            logits2, _ = model(batch_modified)
        
        # Logits for first 32 positions should be identical
        assert torch.allclose(logits1[:, :32, :], logits2[:, :32, :], rtol=1e-5)

    def test_weight_tying(self, model):
        """Test that token embeddings and LM head share weights."""
        wte_weight = model.transformer['wte'].weight
        lm_head_weight = model.lm_head.weight
        
        # Should be the exact same tensor (shared)
        assert wte_weight is lm_head_weight


class TestModelVariants:
    """Test different model sizes."""

    def test_nano_model(self):
        """Test Nano model creation and properties."""
        model = markgpt_nano()
        
        assert model.config.n_embd == 128
        assert model.config.n_layer == 4
        assert model.config.n_head == 4
        assert model.count_parameters() < 5e6  # Should be ~2M

    def test_small_model(self):
        """Test Small model creation and properties."""
        model = markgpt_small()
        
        assert model.config.n_embd == 256
        assert model.config.n_layer == 6
        assert model.config.n_head == 8
        assert 5e6 < model.count_parameters() < 20e6  # Should be ~10M

    @pytest.mark.skip(reason="Base model too large for quick tests")
    def test_base_model(self):
        """Test Base model creation (skipped for speed)."""
        pass


class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_head_dim(self):
        """Test that n_embd must be divisible by n_head."""
        with pytest.raises(AssertionError):
            config = MarkGPTConfig(n_embd=256, n_head=7)
            config.head_size  # Should raise

    def test_config_parameter_count(self):
        """Test parameter count estimation."""
        config = MarkGPTConfig(
            vocab_size=8000,
            n_embd=256,
            n_layer=6,
            n_head=8,
        )
        
        # Shouldn't raise
        est_params = config.parameter_count()
        assert est_params > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
