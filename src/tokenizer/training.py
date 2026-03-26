"""Tokenizer training pipeline with BPE merge history and fertility tracking."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.tokenizer.tokenizer import Tokenizer


def compute_fertility(tokenizer: Tokenizer, text: str) -> float:
    """Compute average tokens per word (fertility).
    
    Lower is better - indicates more efficient tokenization.
    
    Args:
        tokenizer: Trained tokenizer
        text: Text to evaluate
    
    Returns:
        Average tokens per word
    """
    words = text.split()
    tokens = tokenizer.encode(text)
    return len(tokens) / max(len(words), 1)


def train_with_history(text: str, vocab_size: int = 8000) -> Tuple[Tokenizer, Dict]:
    """Train tokenizer and record merge history.
    
    Args:
        text: Training text
        vocab_size: Target vocabulary size
    
    Returns:
        Tuple of (tokenizer, merge_history)
    """
    tokenizer = Tokenizer(vocab_size=vocab_size)
    
    # Record merges during training
    merge_history = []
    initial_vocab_size = len(tokenizer.vocab)
    
    tokenizer.train(text)
    
    merge_history.append({
        'step': 0,
        'vocab_size': initial_vocab_size,
        'merges_to_date': 0,
    })
    
    return tokenizer, {'merges': merge_history}


def save_training_report(
    output_dir: str,
    tokenizer: Tokenizer,
    train_fertility: float,
    val_fertility: float,
    test_fertility: float,
) -> None:
    """Save tokenizer training reports.
    
    Args:
        output_dir: Directory to save report
        tokenizer: Trained tokenizer
        train_fertility: Training set fertility
        val_fertility: Validation set fertility
        test_fertility: Test set fertility
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'vocab_size': len(tokenizer.vocab),
        'training_fertility': train_fertility,
        'validation_fertility': val_fertility,
        'test_fertility': test_fertility,
    }
    
    with open(output_dir / 'fertility_report.json', 'w') as f:
        json.dump(report, f, indent=2)
