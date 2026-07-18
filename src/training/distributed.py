"""Distributed training utilities for multi-GPU training.

Provides helper functions for setting up DistributedDataParallel and
managing multi-GPU training workflows.

References:
    - PyTorch DistributedDataParallel: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
    - DeepSpeed: https://www.deepspeed.ai/
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging


logger = logging.getLogger(__name__)


def setup_distributed_training(rank: int, world_size: int) -> None:
    """Initialize distributed training environment.
    
    Args:
        rank: Process rank in the distributed group
        world_size: Total number of processes
    """
    if rank >= 0:
        # Set CUDA device for this process
        torch.cuda.set_device(rank)
        # Initialize the process group
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )


def wrap_model_distributed(
    model: nn.Module,
    rank: int,
    find_unused_parameters: bool = False,
) -> nn.parallel.DistributedDataParallel:
    """Wrap model with DistributedDataParallel.
    
    Args:
        model: Model to wrap
        rank: Current process rank
        find_unused_parameters: Whether to allow unused parameters
        
    Returns:
        DistributedDataParallel wrapped model
    """
    model = model.to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=find_unused_parameters,
    )
    return ddp_model


def get_sampler_distributed(
    dataset: torch.utils.data.Dataset,
    shuffle: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> torch.utils.data.DistributedSampler:
    """Create DistributedSampler for dataset.
    
    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle samples
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        DistributedSampler instance
    """
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )
    return sampler


def cleanup_distributed() -> None:
    """Cleanup distributed training resources."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


class DistributedTrainerConfig:
    """Configuration for distributed training.
    
    Attributes:
        world_size (int): Number of processes
        rank (int): Current process rank
        master_addr (str): Master node address
        master_port (int): Master node port
        backend (str): Distributed backend ('nccl', 'gloo', 'mpi')
    """

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        master_addr: str = "localhost",
        master_port: int = 29500,
        backend: str = "nccl",
    ):
        """Initialize distributed training config."""
        self.world_size = world_size
        self.rank = rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.backend = backend

    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary."""
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "backend": self.backend,
        }


# TODO: Implement gradient accumulation across multiple GPUs
# TODO: Implement FSDP (Fully Sharded Data Parallel) for large model training
# TODO: Add ZeRO optimizer stage support for memory-efficient training
