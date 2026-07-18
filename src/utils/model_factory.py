"""Factory functions for creating MarkGPT models with different sizes.

Provides preset configurations and utilities for model instantiation and analysis.
"""

from src.model.markgpt import MarkGPT, MarkGPTConfig
import torch
import yaml
from pathlib import Path
from typing import Dict


def get_model_size_config(size: str) -> MarkGPTConfig:
    """Get preset configuration for a model size.
    
    Args:
        size: Model size ('nano', 'small', 'base')
    
    Returns:
        MarkGPTConfig instance
    """
    configs = {
        'nano': MarkGPTConfig(
            vocab_size=8000,
            block_size=512,
            n_embd=128,
            n_layer=4,
            n_head=4,
            dropout=0.1,
            use_flash_attn=True,
        ),
        'small': MarkGPTConfig(
            vocab_size=8000,
            block_size=512,
            n_embd=256,
            n_layer=6,
            n_head=8,
            dropout=0.1,
            use_flash_attn=True,
        ),
        'base': MarkGPTConfig(
            vocab_size=16000,
            block_size=512,
            n_embd=512,
            n_layer=12,
            n_head=8,
            dropout=0.1,
            use_flash_attn=True,
        ),
    }
    
    if size not in configs:
        raise ValueError(f"Unknown model size: {size}. Options: {list(configs.keys())}")
    
    return configs[size]


def markgpt_nano() -> MarkGPT:
    """Create MarkGPT-Nano model (~2M parameters).
    
    Small model for development and experimentation.
    
    Returns:
        MarkGPT model instance
    """
    config = get_model_size_config('nano')
    return MarkGPT(config)


def markgpt_small() -> MarkGPT:
    """Create MarkGPT-Small model (~10M parameters).
    
    Production-quality model for training on single GPU.
    
    Returns:
        MarkGPT model instance
    """
    config = get_model_size_config('small')
    return MarkGPT(config)


def markgpt_base() -> MarkGPT:
    """Create MarkGPT-Base model (~85M parameters).
    
    Larger model for multi-GPU training.
    
    Returns:
        MarkGPT model instance
    """
    config = get_model_size_config('base')
    return MarkGPT(config)


def markgpt_from_config(config_path: str) -> MarkGPT:
    """Load MarkGPT from YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        MarkGPT model instance
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = MarkGPTConfig(**config_dict)
    return MarkGPT(config)


def print_parameter_breakdown(model: MarkGPT) -> None:
    """Print detailed parameter breakdown for a model.
    
    Shows parameters by component: embeddings, attention, ffn, output.
    
    Args:
        model: MarkGPT instance
    """
    params = {}
    
    # Token embedding
    params['Token Embeddings'] = sum(
        p.numel() for name, p in model.named_parameters() 
        if 'wte' in name
    )
    
    # Position embedding  
    params['Position Embeddings'] = sum(
        p.numel() for name, p in model.named_parameters()
        if 'wpe' in name
    )
    
    # Attention layers (per layer)
    attention_params = 0
    for name, p in model.named_parameters():
        if 'attn' in name or 'c_attn' in name or 'c_proj' in name:
            attention_params += p.numel()
    params['Attention Total'] = attention_params
    
    # FFN layers
    ffn_params = 0
    for name, p in model.named_parameters():
        if ('net' in name or 'net.0' in name or 'net.2' in name) and 'ffn' in name:
            ffn_params += p.numel()
    params['Feed-Forward Total'] = ffn_params
    
    # LayerNorms
    ln_params = 0
    for name, p in model.named_parameters():
        if 'ln' in name:
            ln_params += p.numel()
    params['LayerNorms'] = ln_params
    
    # Output head
    params['Output Head'] = sum(
        p.numel() for name, p in model.named_parameters()
        if 'lm_head' in name
    )
    
    # Print breakdown
    print("\n" + "=" * 60)
    print("MarkGPT Parameter Breakdown")
    print("=" * 60)
    
    total = 0
    for component, count in params.items():
        pct = 100.0 * count / model.count_parameters() if count > 0 else 0
        print(f"{component:.<40} {count:>12,} ({pct:>5.1f}%)")
        total += count
    
    print("-" * 60)
    print(f"{'TOTAL':.<40} {total:>12,} (100.0%)")
    print("=" * 60 + "\n")


def model_summary(model: MarkGPT) -> Dict[str, int]:
    """Get summary statistics about the model.
    
    Args:
        model: MarkGPT instance
    
    Returns:
        Dictionary with various statistics
    """
    return {
        'total_params': model.count_parameters(),
        'vocab_size': model.config.vocab_size,
        'embedding_dim': model.config.n_embd,
        'n_layers': model.config.n_layer,
        'n_heads': model.config.n_head,
        'head_size': model.config.head_size,
        'context_length': model.config.block_size,
        'using_flash_attn': model.config.use_flash_attn,
    }
