"""Model surgery utilities: parameter counting, freezing, model inspection."""

import torch.nn as nn
from typing import Dict, Set


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters broken down by component.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary mapping component names to parameter counts
    """
    params = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        components = name.split('.')
        comp_name = components[0] if components else 'other'
        params[comp_name] = params.get(comp_name, 0) + param.numel()
    return params


def freeze_layers(model: nn.Module, layer_names: Set[str]) -> None:
    """Freeze specified layers (disable gradients).
    
    Args:
        model: PyTorch model
        layer_names: Set of layer names to freeze
    """
    for name, param in model.named_parameters():
        if any(target in name for target in layer_names):
            param.requires_grad = False


def print_model_summary(model: nn.Module) -> None:
    """Print summary of model architecture.
    
    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total_params - trainable
    
    print(f"\nModel Summary:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen: {frozen:,}")


def save_model(model: nn.Module, path: str, metadata: dict = None) -> None:
    """Save model with config metadata.
    
    Args:
        model: Model to save
        path: Save path
        metadata: Optional metadata dictionary
    """
    import json
    state = {
        'model_state_dict': model.state_dict(),
        'config': metadata or {}
    }
    torch.save(state, path)
