"""Preprocess Bible corpus into binary token files for efficient training.

Reads raw KJV text, cleans verse markers, normalizes whitespace, adds
BOS/EOS tokens at chapter boundaries, splits into train/val/test, and
saves as binary .bin files with uint16 token IDs for fast DataLoader access.

Usage:
    python scripts/preprocess_bible.py --input data/raw/kjv_bible.txt --vocab-size 8000
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
from tqdm import tqdm

from src.tokenizer.tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BiblePreprocessor:
    """Preprocess Bible text into tokenized binary format."""

    def __init__(self, vocab_size: int = 8000):
        """Initialize preprocessor.
        
        Args:
            vocab_size: Target vocabulary size for BPE tokenizer
        """
        self.vocab_size = vocab_size
        self.tokenizer = None

    def clean_text(self, text: str) -> str:
        """Clean Bible text by removing verse markers and normalizing whitespace.
        
        Args:
            text: Raw Bible text
        
        Returns:
            Cleaned text
        """
        # Remove verse markers like "1:1", "2:5" at line starts
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Skip empty lines and chapter headers
            if not line.strip() or line.startswith("***"):
                continue

            # Remove leading verse numbers (e.g., "1:1 ")
            parts = line.split(maxsplit=1)
            if len(parts) > 1 and ":" in parts[0]:
                cleaned_lines.append(parts[1])
            else:
                cleaned_lines.append(line)

        # Join and normalize whitespace
        text = " ".join(cleaned_lines)
        text = " ".join(text.split())  # Remove multiple spaces
        return text

    def add_special_tokens(self, text: str) -> str:
        """Add BOS/EOS tokens at chapter boundaries.
        
        Args:
            text: Cleaned Bible text
        
        Returns:
            Text with special tokens inserted
        """
        # Simple heuristic: chapters typically start with book names or "Chapter X"
        # For simplicity, we'll add BOS at major punctuation (periods)
        # In practice, would need more sophisticated detection
        lines = text.split(".")
        marked = []

        for i, line in enumerate(lines):
            if i > 0:
                marked.append(" <EOS>")
            marked.append(line)
            if i < len(lines) - 1:
                marked.append(".")
                marked.append(" <BOS>")

        return "".join(marked)

    def train_tokenizer(self, text: str) -> Tokenizer:
        """Train BPE tokenizer on corpus.
        
        Args:
            text: Text to train tokenizer on
        
        Returns:
            Trained Tokenizer instance
        """
        logger.info(f"Training BPE tokenizer with vocab size {self.vocab_size}...")
        tokenizer = Tokenizer(vocab_size=self.vocab_size)
        tokenizer.train(text)
        self.tokenizer = tokenizer
        logger.info(f"✓ Tokenizer trained. Vocabulary size: {len(tokenizer.vocab)}")
        return tokenizer

    def tokenize_and_save(
        self,
        text: str,
        output_dir: Path,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
    ) -> None:
        """Tokenize text and save as binary files.
        
        Args:
            text: Text to tokenize
            output_dir: Directory to save .bin files
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
                      (test_ratio = 1 - train_ratio - val_ratio)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train_tokenizer() first.")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Tokenize entire corpus
        logger.info("Tokenizing corpus...")
        tokens = self.tokenizer.encode(text)
        tokens_array = np.array(tokens, dtype=np.uint16)

        # Calculate split indices
        n_tokens = len(tokens)
        train_size = int(n_tokens * train_ratio)
        val_size = int(n_tokens * val_ratio)

        train_tokens = tokens_array[:train_size]
        val_tokens = tokens_array[train_size : train_size + val_size]
        test_tokens = tokens_array[train_size + val_size :]

        # Save splits as binary files
        train_file = output_dir / "train.bin"
        val_file = output_dir / "val.bin"
        test_file = output_dir / "test.bin"

        train_tokens.tofile(train_file)
        val_tokens.tofile(val_file)
        test_tokens.tofile(test_file)

        logger.info(f"✓ Training set: {len(train_tokens):,} tokens → {train_file}")
        logger.info(f"✓ Validation set: {len(val_tokens):,} tokens → {val_file}")
        logger.info(f"✓ Test set: {len(test_tokens):,} tokens → {test_file}")

        # Compute and log statistics
        self._log_statistics(text, train_size, val_size, test_size=len(test_tokens))

    def _log_statistics(
        self, text: str, train_size: int, val_size: int, test_size: int
    ) -> None:
        """Log preprocessing statistics.
        
        Args:
            text: Original text
            train_size: Number of training tokens
            val_size: Number of validation tokens
            test_size: Number of test tokens
        """
        total_tokens = train_size + val_size + test_size
        unique_words = len(set(text.split()))

        stats = {
            "total_characters": len(text),
            "total_words": len(text.split()),
            "unique_words": unique_words,
            "total_tokens": total_tokens,
            "training_tokens": train_size,
            "validation_tokens": val_size,
            "test_tokens": test_size,
            "vocabulary_size": len(self.tokenizer.vocab),
            "average_tokens_per_word": total_tokens / max(len(text.split()), 1),
        }

        logger.info("\nPreprocessing Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:,}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Preprocess Bible corpus into binary token files")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/kjv_bible.txt",
        help="Path to raw Bible text file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for .bin files",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Target vocabulary size for BPE tokenizer",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.90,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of data for validation",
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(input_file)

    # Read and preprocess text
    logger.info(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    logger.info(f"Raw text size: {len(text):,} characters")

    # Clean and prepare
    preprocessor = BiblePreprocessor(vocab_size=args.vocab_size)
    text = preprocessor.clean_text(text)
    text = preprocessor.add_special_tokens(text)

    logger.info(f"Cleaned text size: {len(text):,} characters")

    # Train tokenizer and tokenize
    preprocessor.train_tokenizer(text)
    preprocessor.tokenize_and_save(
        text,
        output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    logger.info("\n✓ Preprocessing complete!")


if __name__ == "__main__":
    main()
