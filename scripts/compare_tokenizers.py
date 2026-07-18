"""Compare tokenizers with different vocabulary sizes.

Measures fertility (tokens per word) on English, Banso, and mixed text.
"""

import argparse
from pathlib import Path
from src.tokenizer.tokenizer import Tokenizer
from src.tokenizer.training import compute_fertility


def compare_tokenizers(
    text: str,
    vocab_sizes: list = None,
    output_dir: str = 'tokenizer_comparison',
):
    """Compare tokenizers with different vocabulary sizes.
    
    Args:
        text: Text to evaluate on
        vocab_sizes: List of vocabulary sizes to test
        output_dir: Directory to save results
    """
    if vocab_sizes is None:
        vocab_sizes = [2000, 4000, 8000]
    
    results = {}
    
    for vocab_size in vocab_sizes:
        print(f"Training tokenizer with vocab_size={vocab_size}...")
        tokenizer = Tokenizer(vocab_size=vocab_size)
        tokenizer.train(text)
        
        fertility = compute_fertility(tokenizer, text)
        results[vocab_size] = {
            'vocab_size': vocab_size,
            'fertility': fertility,
            'tokens_total': len(tokenizer.encode(text)),
        }
        
        print(f"  Fertility: {fertility:.3f}")
    
    # Print comparison table
    print("\nTokenizer Comparison:")
    print("Vocab Size | Fertility")
    print("-" * 25)
    for vocab_size in vocab_sizes:
        fert = results[vocab_size]['fertility']
        print(f"{vocab_size:>10} | {fert:>.3f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Input text file")
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[2000, 4000, 8000])
    args = parser.parse_args()
    
    with open(args.text, 'r') as f:
        text = f.read()
    
    compare_tokenizers(text, args.vocab_sizes)
