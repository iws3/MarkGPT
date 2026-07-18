"""Visualize tokenization in the terminal."""

import argparse
from src.tokenizer.tokenizer import Tokenizer


def visualize_tokenization(text: str, tokenizer: Tokenizer):
    """Display tokenization with color-coding.
    
    Args:
        text: Text to visualize
        tokenizer: Tokenizer to use
    """
    tokens = tokenizer.encode(text)
    decoded_tokens = [tokenizer.decode([t]) for t in tokens]
    
    print("Tokenization Visualization:")
    print("=" * 60)
    print(f"Original: {text}")
    print(f"Total tokens: {len(tokens)}")
    print("\nTokens:")
    for i, (token, text_repr) in enumerate(zip(tokens, decoded_tokens)):
        print(f"  [{i:3d}] {token:5d} → {text_repr!r}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()
    
    tokenizer = Tokenizer(vocab_size=8000)
    # In practice, would load pretrained tokenizer
    visualize_tokenization(args.text, tokenizer)
