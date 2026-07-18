"""Vocabulary analysis for tokenizer evaluation.

Analyzes vocabulary statistics, token frequency, and cross-language coverage.
This script produces analysis suitable for visualization in notebooks.
"""

import json
from collections import Counter
from typing import Dict, List


def analyze_vocabulary(
    tokens: List[int],
    tokenizer,
    text: str = None,
) -> Dict:
    """Analyze vocabulary statistics.
    
    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer instance
        text: Optional source text for context
    
    Returns:
        Dictionary with analysis results
    """
    # Token frequency
    token_freq = Counter(tokens)
    most_common = token_freq.most_common(10)
    
    # Vocabulary coverage
    unique_tokens = len(set(tokens))
    vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else 0
    coverage = unique_tokens / vocab_size if vocab_size > 0 else 0
    
    analysis = {
        'total_tokens': len(tokens),
        'unique_tokens': unique_tokens,
        'vocab_size': vocab_size,
        'coverage': coverage,
        'most_common_tokens': most_common,
    }
    
    return analysis


def cross_language_coverage(
    english_text: str,
    banso_text: str,
    tokenizer,
) -> Dict:
    """Measure tokenizer coverage across languages.
    
    Args:
        english_text: English text sample
        banso_text: Banso text sample
        tokenizer: Tokenizer to evaluate
    
    Returns:
        Coverage statistics per language
    """
    eng_tokens = tokenizer.encode(english_text)
    banso_tokens = tokenizer.encode(banso_text)
    
    eng_unique = len(set(eng_tokens))
    banso_unique = len(set(banso_tokens))
    
    return {
        'english': {
            'total': len(eng_tokens),
            'unique': eng_unique,
        },
        'banso': {
            'total': len(banso_tokens),
            'unique': banso_unique,
        },
    }
