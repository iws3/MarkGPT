"""Tests for Bible corpus data loading utilities.

Tests BibleDataset and DataLoader functionality including:
- Shape verification
- Target is input shifted by one
- Token ID bounds
- Multiple sequences correctness
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.data_loader import BibleDataset, create_dataloaders


@pytest.fixture
def sample_data():
    """Create temporary binary data file for testing.
    
    Yields:
        Path to temporary .bin file with 1000 random token IDs
    """
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        # Create sample token data: random uint16 values between 0 and 8000
        n_tokens = 1000
        data = np.random.randint(0, 8000, size=n_tokens, dtype=np.uint16)
        data.tofile(f.name)
        yield f.name

    # Cleanup
    Path(f.name).unlink()


class TestBibleDataset:
    """Test BibleDataset class."""

    def test_initialization(self, sample_data):
        """Test dataset initialization.
        
        Args:
            sample_data: Path to sample binary data
        """
        dataset = BibleDataset(sample_data, block_size=512)
        assert dataset.data is not None
        assert dataset.block_size == 512
        assert dataset.length > 0

    def test_length(self, sample_data):
        """Test dataset length calculation.
        
        Args:
            sample_data: Path to sample binary data
        """
        block_size = 512
        dataset = BibleDataset(sample_data, block_size=block_size)
        # Length should be (total tokens - block_size)
        assert len(dataset) == 1000 - block_size

    def test_getitem_shape(self, sample_data):
        """Test that getitem returns correct shapes.
        
        Args:
            sample_data: Path to sample binary data
        """
        block_size = 256
        dataset = BibleDataset(sample_data, block_size=block_size)

        for idx in range(min(10, len(dataset))):
            x, y = dataset[idx]
            assert x.shape == (block_size,), f"Input shape mismatch at idx {idx}"
            assert y.shape == (block_size,), f"Target shape mismatch at idx {idx}"

    def test_target_is_shifted_input(self, sample_data):
        """Test that target is input shifted by one position.
        
        This is the key property for next-token prediction.
        
        Args:
            sample_data: Path to sample binary data
        """
        dataset = BibleDataset(sample_data, block_size=128)

        for idx in range(min(5, len(dataset))):
            x, y = dataset[idx]

            # y[i] should equal x[i+1] for all i
            # This means y[0] = x[1], y[1] = x[2], etc.
            expected_y = x[1:]
            assert torch.allclose(y[:-1], expected_y), f"Shift property failed at idx {idx}"

    def test_token_ids_in_bounds(self, sample_data):
        """Test that all returned token IDs are within valid range.
        
        Args:
            sample_data: Path to sample binary data
        """
        vocab_size = 8000
        dataset = BibleDataset(sample_data, block_size=256)

        for idx in range(min(10, len(dataset))):
            x, y = dataset[idx]

            assert torch.all(x >= 0), f"Negative token ID at idx {idx}"
            assert torch.all(x < vocab_size), f"Token ID >= vocab_size at idx {idx}"
            assert torch.all(y >= 0), f"Negative target ID at idx {idx}"
            assert torch.all(y < vocab_size), f"Target ID >= vocab_size at idx {idx}"

    def test_indexing_bounds(self, sample_data):
        """Test that indexing respects bounds.
        
        Args:
            sample_data: Path to sample binary data
        """
        dataset = BibleDataset(sample_data, block_size=256)

        # Valid indices
        dataset[0]
        dataset[len(dataset) - 1]

        # Invalid indices
        with pytest.raises(IndexError):
            dataset[len(dataset)]

        with pytest.raises(IndexError):
            dataset[-1]

    def test_file_not_found(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            BibleDataset("nonexistent_file.bin", block_size=256)

    def test_file_too_small(self):
        """Test that data file smaller than block_size raises error."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            # Create a file with fewer tokens than block_size
            data = np.random.randint(0, 1000, size=10, dtype=np.uint16)
            data.tofile(f.name)

            try:
                with pytest.raises(ValueError):
                    BibleDataset(f.name, block_size=256)
            finally:
                Path(f.name).unlink()


class TestDataLoaderFactory:
    """Test create_dataloaders factory function."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with train/val/test splits.
        
        Yields:
            Path to temporary data directory
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create train/val/test binary files
            for split in ["train", "val", "test"]:
                data = np.random.randint(0, 8000, size=2000, dtype=np.uint16)
                data.tofile(str(data_dir / f"{split}.bin"))

            yield str(data_dir)

    def test_create_dataloaders(self, temp_data_dir):
        """Test create_dataloaders creates three loaders.
        
        Args:
            temp_data_dir: Path to temporary data directory
        """
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=temp_data_dir, block_size=256, batch_size=32
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_batch_shapes(self, temp_data_dir):
        """Test that batches have correct shapes.
        
        Args:
            temp_data_dir: Path to temporary data directory
        """
        block_size = 256
        batch_size = 32
        train_loader, _, _ = create_dataloaders(
            data_dir=temp_data_dir, block_size=block_size, batch_size=batch_size
        )

        # Get first batch
        for x, y in train_loader:
            assert x.shape == (batch_size, block_size)
            assert y.shape == (batch_size, block_size)
            break

    def test_train_loader_shuffled(self, temp_data_dir):
        """Test that training loader shuffles data.
        
        Args:
            temp_data_dir: Path to temporary data directory
        """
        train_loader, _, _ = create_dataloaders(
            data_dir=temp_data_dir, block_size=256, batch_size=32
        )

        # Get first batch twice (should be different due to shuffling)
        batches = []
        for _ in range(2):
            for x, y in train_loader:
                batches.append(x.clone())
                break

        # Batches should be different (with high probability)
        # (extremely unlikely they're the same by chance)
        assert not torch.allclose(batches[0], batches[1])

    def test_val_loader_not_shuffled(self, temp_data_dir):
        """Test that validation loader returns same order consistently.
        
        Args:
            temp_data_dir: Path to temporary data directory
        """
        _, val_loader, _ = create_dataloaders(
            data_dir=temp_data_dir, block_size=256, batch_size=32
        )

        # Get first batch twice
        batches = []
        for _ in range(2):
            for x, y in val_loader:
                batches.append(x.clone())
                break

        # Batches should be identical (no shuffling)
        assert torch.allclose(batches[0], batches[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
