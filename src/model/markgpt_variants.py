"""MarkGPT model architecture variants for ablation studies.

Educational variants for understanding architecture design choices.
"""

import torch.nn as nn
from src.model.markgpt import MarkGPTConfig, FeedForward


class TransformerBlockPostNorm(nn.Module):
    """Transformer block with POST-NORMALIZATION (original paper design).
    
    LayerNorm applied after residual connections instead of before.
    Generally requires careful learning rate tuning.
    """

    def __init__(self, config: MarkGPTConfig):
        super().__init__()
        from src.model.markgpt import CausalSelfAttention
        
        self.attn = CausalSelfAttention(config, use_flash_attn=config.use_flash_attn)
        self.ffn = FeedForward(config)
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        # Post-norm: norm is applied AFTER residual
        x = x + self.attn(x)
        x = self.ln_1(x)
        
        x = x + self.ffn(x)
        x = self.ln_2(x)
        
        return x


class TransformerBlockRMSNorm(nn.Module):
    """Transformer block using RMSNorm instead of LayerNorm.
    
    RMSNorm (Zhang & Sennrich, 2019) normalizes based on RMS instead of variance.
    Used in T5 and recent Llama models.
    """

    def __init__(self, config: MarkGPTConfig):
        super().__init__()
        from src.model.markgpt import CausalSelfAttention
        
        self.attn = CausalSelfAttention(config, use_flash_attn=config.use_flash_attn)
        self.ffn = FeedForward(config)
        
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x
