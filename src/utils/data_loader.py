"""Bible corpus data loading utilities for efficient training.

Provides:
- BibleDataset: Memory-mapped dataset for fast access to binary token files
- create_dataloaders(): Factory function for train/val/test splits
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BibleDataset(Dataset):
    """Memory-mapped dataset for Bible corpus text.

    Loads pre-tokenized Bible text from a binary .bin file and yields
    (input_ids, target_ids) pairs of configurable length.

    The dataset memory-maps the file, enabling efficient loading of very large
    corpora without loading everything into RAM.

    Attributes:
        data_file: Path to the .bin file containing uint16 token IDs
        block_size: Context length (input sequence length)
        data: Memory-mapped file handle
        length: Total number of sequences available

    Example:
        >>> dataset = BibleDataset("data/processed/kjv_train.bin", block_size=512)
        >>> loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        >>> for x, y in loader:
        ...     print(x.shape, y.shape)  # (32, 512), (32, 512)
    """

    def __init__(self, data_file: str, block_size: int = 512):
        """Initialize dataset.

        Args:
            data_file: Path to binary .bin file with uint16 token IDs
            block_size: Context window length for model input/output pairs

        Raises:
            FileNotFoundError: If data_file doesn't exist
            ValueError: If data file is too small
        """
        self.data_file = Path(data_file)
        self.block_size = block_size

        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Memory-map the file for efficient access
        self.data = np.memmap(str(self.data_file), dtype=np.uint16, mode="r")

        # Calculate number of sequences
        # Each sequence uses block_size + 1 tokens (input + target)
        if len(self.data) < block_size + 1:
            raise ValueError(
                f"Data file too small. Need at least {block_size + 1} tokens, "
                f"got {len(self.data)}"
            )

        self.length = len(self.data) - block_size

    def __len__(self) -> int:
        """Return number of sequences in dataset.

        Returns:
            Number of sequences
        """
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a (input, target) pair.

        The target is the input shifted by one position. This allows the model
        to learn next-token prediction: given tokens [0, 1, 2, ..., n-1], predict
        tokens [1, 2, 3, ..., n].

        Args:
            idx: Sequence index

        Returns:
            Tuple of (input_ids, target_ids), each of shape (block_size,)

        Raises:
            IndexError: If idx is out of bounds
        """
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range [0, {self.length})")

        # Read block_size + 1 tokens starting at position idx
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y


def create_dataloaders(
    data_dir: str = "data/processed",
    block_size: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders for Bible corpus.

    Creates three DataLoaders from pre-processed binary files:
    - Training: Shuffled, uses dropout/random cropping for augmentation
    - Validation: Not shuffled, deterministic ordering
    - Test: Not shuffled, deterministic ordering

    Args:
        data_dir: Directory containing train.bin, val.bin, test.bin
        block_size: Context window length (input sequence length)
        batch_size: Number of sequences per batch
        num_workers: Number of worker processes for data loading
                     (0 = synchronous loading, >0 = parallel)
        pin_memory: If True, pin batches to GPU memory for faster transfer
                    (only beneficial if using GPU)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Raises:
        FileNotFoundError: If any .bin file is missing
        ValueError: If block_size is too large for data

    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     data_dir="data/processed",
        ...     block_size=512,
        ...     batch_size=64,
        ... )
        >>> for batch_x, batch_y in train_loader:
        ...     print(batch_x.shape)  # (64, 512)
        ...     break  # Just show first batch
    """
    data_dir = Path(data_dir)

    # Create datasets
    train_dataset = BibleDataset(str(data_dir / "train.bin"), block_size=block_size)
    val_dataset = BibleDataset(str(data_dir / "val.bin"), block_size=block_size)
    test_dataset = BibleDataset(str(data_dir / "test.bin"), block_size=block_size)

    # Create dataloaders with appropriate settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for test
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_batch(
    loader: DataLoader, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get next batch from dataloader.

    Convenience function for training loops.

    Args:
        loader: A PyTorch DataLoader instance
        device: Device to move batch to ("cpu" or "cuda")

    Returns:
        Tuple of (input_ids, target_ids) on specified device

    Example:
        >>> x, y = get_batch(train_loader, device="cuda")
        >>> outputs = model(x)
    """
    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    return x, y
