"""Mixed-language dataset for bilingual pre-training."""

import torch
from torch.utils.data import Dataset
from typing import Tuple


class MixedLanguageDataset(Dataset):
    """Dataset that interleaves Bible (English) and Banso text.
    
    Useful for creating multilingual models that can generate in both languages.
    """

    def __init__(
        self,
        english_data: torch.Tensor,
        banso_data: torch.Tensor,
        block_size: int = 512,
        english_ratio: float = 0.8,
    ):
        """Initialize mixed dataset.
        
        Args:
            english_data: English (KJV) token tensor
            banso_data: Banso corpus token tensor
            block_size: Context window size
            english_ratio: Fraction of batches from English (0-1)
        """
        self.english_data = english_data
        self.banso_data = banso_data
        self.block_size = block_size
        self.english_ratio = english_ratio
        
        self.english_length = len(english_data) - block_size
        self.banso_length = len(banso_data) - block_size

    def __len__(self) -> int:
        """Return total length (use English as primary)."""
        return self.english_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get mixed-language sample.
        
        Returns:
            (input_ids, target_ids, language_label)
        """
        import random
        
        if random.random() < self.english_ratio:
            # Sample from English
            data = self.english_data
            length = self.english_length
            language = "english"
        else:
            # Sample from Banso
            data = self.banso_data
            length = self.banso_length
            language = "banso"
        
        # Ensure valid index
        idx = idx % length
        
        chunk = data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y, language
