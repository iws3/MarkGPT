"""
Day 25 Exercise: BPE Tokenization from Scratch
Module 05: NLP Foundations
===============================================

Objective:
- Implement Byte-Pair Encoding (BPE) from scratch.
- Understand the trade-off between vocabulary size and sequence length (Fertility).
- Train BPE on the KJV Bible and observe common merges.

Tasks:
1. Implement pair counting.
2. Implement pair merging.
3. Build the full BPE training loop.
4. Measure 'Fertility' on sample text.
"""

from collections import Counter
import re

# --- PART 1: Data Preparation ---

def get_stats(vocab):
    """Count frequencies of adjacent character pairs in the vocabulary."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """Merge all occurrences of the most frequent pair in the vocabulary."""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# --- PART 2: Implementation ---

def train_bpe(text, num_merges=10):
    """
    Train BPE on text.
    
    Args:
        text (str): The raw corpus.
        num_merges (int): Number of merge operations to perform.
        
    Returns:
        merges (dict): Mapping of pairs to merged strings.
        vocab (dict): Final dictionary of merged words.
    """
    # 1. Initialize: Split text into words and then characters with spaces
    # Example: "low" -> "l o w </w>"
    words = text.strip().split()
    vocab = Counter()
    for word in words:
        # We add </w> to mark the end of a word
        char_word = ' '.join(list(word)) + ' </w>'
        vocab[char_word] += 1
        
    merges = {}
    
    print(f"Initial Vocabulary Size (Characters): {len(set(''.join(vocab.keys()).split()))}")
    
    # 2. Iteratively merge common pairs
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
            
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges[best] = ''.join(best)
        
        if (i + 1) % 5 == 0 or i == 0:
            print(f"Merge {i+1}: {best} -> {''.join(best)} (Count: {pairs[best]})")
            
    return merges, vocab

# --- PART 3: Encoding & Fertility ---

def encode(word, merges):
    """Encode a single word using the learned merges."""
    if not word: return ""
    
    # Start with characters
    word_chars = list(word) + ['</w>']
    
    # Apply merges in the order they were learned
    for (p1, p2), merged in merges.items():
        i = 0
        while i < len(word_chars) - 1:
            if word_chars[i] == p1 and word_chars[i+1] == p2:
                word_chars = word_chars[:i] + [merged] + word_chars[i+2:]
            else:
                i += 1
    return word_chars

def calculate_fertility(corpus, merges):
    """Calculate Fertility: Average number of tokens per word."""
    words = corpus.split()
    total_tokens = 0
    
    for word in words:
        tokens = encode(word, merges)
        total_tokens += len(tokens)
        
    return total_tokens / len(words)

# --- PART 4: Execution ---

def main():
    # Load KJV Bible sample
    workspace_root = r'c:\Users\the eye informatique\Desktop\ML\AI\MarkGPT'
    bible_path = os.path.join(workspace_root, 'data', 'raw', 'kjv_bible.txt')
    
    if os.path.exists(bible_path):
        with open(bible_path, 'r', encoding='utf-8') as f:
            # Take a small sample for speed
            text = f.read(50000)
    else:
        text = "in the beginning was the word and the word was with god and the word was god"
        print("KJV Bible not found, using toy text.")

    # Train BPE
    num_merges = 50
    print(f"\nTraining BPE for {num_merges} merges...")
    merges, final_vocab = train_bpe(text, num_merges)
    
    # Sample Encoding
    test_word = "beginning"
    encoded = encode(test_word, merges)
    print(f"\nEncoding test: '{test_word}' -> {encoded}")
    
    # Calculate Fertility
    fertility = calculate_fertility(text, merges)
    print(f"Fertility (Avg tokens per word): {fertility:.2f}")
    
    # TODO: Try training on Banso text (if available) and compare fertility!
    # Hint: Banso text is in data/banso-vernacular/

if __name__ == "__main__":
    import os
    main()
