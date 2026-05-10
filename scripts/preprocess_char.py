"""Preprocess text into character-level binary token files.

Usage:
    python scripts/preprocess_char.py --input data/raw/john.txt --output data/processed/john_char
"""

import argparse
import os
from pathlib import Path
import json
from array import array

def main():
    parser = argparse.ArgumentParser(description="Character-level preprocessing")
    parser.add_argument("--input", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    # Save metadata
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(output_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    # Encode
    ids = [stoi[c] for c in text]
    
    # Split
    n = len(ids)
    train_size = int(n * 0.9)
    val_size = int(n * 0.05)

    train_ids = ids[:train_size]
    val_ids = ids[train_size:train_size+val_size]
    test_ids = ids[train_size+val_size:]

    # Save splits as binary files (H for uint16)
    with open(output_dir / 'train.bin', 'wb') as f:
        array('H', train_ids).tofile(f)
    with open(output_dir / 'val.bin', 'wb') as f:
        array('H', val_ids).tofile(f)
    with open(output_dir / 'test.bin', 'wb') as f:
        array('H', test_ids).tofile(f)

    print(f"Saved to {output_dir}")
    print(f"Train: {len(train_ids)} tokens")


if __name__ == "__main__":
    main()
