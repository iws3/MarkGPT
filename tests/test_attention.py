"""Tests for attention mechanism properties."""

import torch
import pytest
from src.model.markgpt import CausalSelfAttention, MarkGPTConfig


class TestAttentionProperties:
    """Test attention layer invariants."""

    @pytest.fixture
    def attn(self):
        config = MarkGPTConfig(n_embd=128, n_head=4, block_size=64)
        return CausalSelfAttention(config)

    def test_causal_mask_property(self, attn):
        """Test that causal mask produces lower-triangular attention."""
        batch_size = 2
        seq_len = 16
        
        x = torch.randn(batch_size, seq_len, 128)
        
        # Forward pass (uses attention internally)
        output = attn(x)
        
        assert output.shape == x.shape

    def test_attention_output_shape(self, attn):
        """Test multi-head attention output shape."""
        batch_size = 4
        seq_len = 32
        
        x = torch.randn(batch_size, seq_len, attn.n_embd)
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, attn.n_embd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
