"""Evaluation metrics for language models.

Provides:
- compute_perplexity(): Per-token perplexity on a dataset
- compute_bleu(): BLEU score against reference texts
- compute_self_bleu(): BLEU score for diversity measurement
"""

import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_perplexity(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> float:
    """Compute per-token perplexity on a dataset.

    Perplexity measures how "surprised" the model is by the test data.
    Lower is better. A perplexity of P means the model assigns on average
    1/P probability to the actual next token.

    The perplexity is computed as exp(average_loss), where loss is cross-entropy.

    Args:
        model: A language model with a forward() method that returns logits
               of shape (batch_size, seq_len, vocab_size)
        dataloader: A DataLoader yielding (input_ids, target_ids) batches
        device: Device to run inference on ("cpu" or "cuda")

    Returns:
        The average perplexity across all batches

    Raises:
        ValueError: If dataloader is empty
        RuntimeError: If model fails forward pass

    Example:
        >>> model.eval()
        >>> ppl = compute_perplexity(model, val_loader, device="cuda")
        >>> print(f"Validation perplexity: {ppl:.2f}")

    References:
        - Jurafsky & Martin (2023): Speech and Language Processing, Ch. 3
        - Merity et al. (2017): "Regularizing and Optimizing LSTM Language Models"
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    if len(dataloader) == 0:
        raise ValueError("DataLoader is empty")

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            try:
                logits = model(x)
            except Exception as e:
                raise RuntimeError(f"Model forward pass failed at batch {batch_idx}: {e}")

            # Reshape for loss computation
            # logits: (batch_size, seq_len, vocab_size)
            # y: (batch_size, seq_len)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    perplexity = float(np.exp(avg_loss))

    logger.info(f"Perplexity: {perplexity:.2f} (avg loss: {avg_loss:.4f})")

    return perplexity


def compute_bleu(
    generated_texts: List[str],
    reference_texts: List[str],
    max_ngram: int = 4,
) -> float:
    """Compute BLEU score against reference texts.

    BLEU (Bilingual Evaluation Understudy) measures how many n-grams in the
    generated text match reference translations. It ranges from 0 to 1.

    This is a simplified BLEU implementation. For production use, consider
    sacrebleu: https://github.com/mjpost/sacrebleu

    Args:
        generated_texts: List of generated text strings
        reference_texts: List of reference text strings (same length as generated)
        max_ngram: Maximum n-gram length to consider (typically 4)

    Returns:
        BLEU score between 0 and 1

    Raises:
        ValueError: If input lists have different lengths or are empty
        AssertionError: If max_ngram < 1

    Example:
        >>> generated = ["the cat is on the mat"]
        >>> reference = ["the cat is on the mat"]
        >>> bleu = compute_bleu(generated, reference)
        >>> print(f"BLEU: {bleu:.4f}")  # 1.0 (perfect match)

    References:
        - Papineni et al. (2002): "BLEU: a Method for Automatic Evaluation"
        - https://github.com/mjpost/sacrebleu
    """
    if len(generated_texts) != len(reference_texts):
        raise ValueError("Generated and reference text lists must have same length")

    if len(generated_texts) == 0:
        raise ValueError("Input lists are empty")

    assert max_ngram >= 1, "max_ngram must be >= 1"

    def _get_ngrams(text: str, n: int) -> dict:
        """Extract n-grams from text.
        
        Args:
            text: Input text string
            n: N-gram length
        
        Returns:
            Dictionary mapping n-grams to counts
        """
        tokens = text.lower().split()
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    # Compute BLEU score for each n-gram
    scores = []

    for n in range(1, max_ngram + 1):
        matches = 0
        total = 0

        for gen_text, ref_text in zip(generated_texts, reference_texts):
            gen_ngrams = _get_ngrams(gen_text, n)
            ref_ngrams = _get_ngrams(ref_text, n)

            for ngram, count in gen_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
                total += count

        if total == 0:
            scores.append(0.0)
        else:
            scores.append(matches / total)

    # Compute geometric mean of n-gram scores
    bleu = float(np.exp(np.mean(np.log(np.array(scores) + 1e-10))))

    logger.info(f"BLEU score: {bleu:.4f}")

    return bleu


def compute_self_bleu(
    generated_texts: List[str],
    max_ngram: int = 4,
) -> float:
    """Compute self-BLEU score for diversity measurement.

    Self-BLEU measures how much generated samples resemble each other. Lower
    self-BLEU indicates more diverse generation. It's computed by treating
    each sample as a reference and computing BLEU against all others.

    Args:
        generated_texts: List of generated text strings
        max_ngram: Maximum n-gram length to consider

    Returns:
        Self-BLEU score (lower = more diverse)

    Raises:
        ValueError: If fewer than 2 samples provided

    Example:
        >>> texts = [
        ...     "the quick brown fox",
        ...     "a quick brown fox"
        ... ]
        >>> self_bleu = compute_self_bleu(texts)
        >>> print(f"Self-BLEU: {self_bleu:.4f}")

    References:
        - Zhu et al. (2018): "Texygen: A Benchmarking Platform"
    """
    if len(generated_texts) < 2:
        raise ValueError("Need at least 2 samples to compute self-BLEU")

    def _get_ngrams(text: str, n: int) -> dict:
        """Extract n-grams from text."""
        tokens = text.lower().split()
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    total_bleu = 0.0

    # For each text, compute BLEU against all others
    for i, test_text in enumerate(generated_texts):
        ref_texts = generated_texts[:i] + generated_texts[i + 1 :]

        scores = []
        for n in range(1, max_ngram + 1):
            matches = 0
            total = 0

            test_ngrams = _get_ngrams(test_text, n)

            for ref_text in ref_texts:
                ref_ngrams = _get_ngrams(ref_text, n)

                for ngram, count in test_ngrams.items():
                    matches += min(count, ref_ngrams.get(ngram, 0))
                    total += count

            if total == 0:
                scores.append(0.0)
            else:
                scores.append(matches / total)

        bleu = float(np.exp(np.mean(np.log(np.array(scores) + 1e-10))))
        total_bleu += bleu

    self_bleu = total_bleu / len(generated_texts)

    logger.info(f"Self-BLEU (diversity): {self_bleu:.4f}")

    return self_bleu
