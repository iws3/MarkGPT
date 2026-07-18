# Module 5 Exercise: BPE Tokenization from Scratch

## Objective
Implement Byte-Pair Encoding (BPE) to understand tokenization.

## Background
Text doesn't naturally fit neural networks (discrete tokens). Tokenization bridges this gap. BPE builds a vocabulary by iteratively merging common byte pairs—a simple but powerful algorithm used in GPT-2, GPT-3, and MarkGPT.

## Part 1: Count Byte Pairs

```python
from collections import Counter

text = "the bible speaks truth the bible god"

# TODO: Convert text to characters + special END marker
# Example: "the" → ['t', 'h', 'e', '</w>']
tokens = None

print(tokens)
# Should be: ['t', 'h', 'e', '</w>', 'b', 'i', 'b', 'l', 'e', '</w>', ...]

# TODO: Find all adjacent pairs and count them
pair_counts = None

# Hint: zip(tokens[:-1], tokens[1:]) gives you all adjacent pairs
# Use Counter to count occurrences

print(pair_counts.most_common(5))
# Should show: [('e', '</w>'), ('b', 'i'), ('i', 'b'), ('l', 'e'), ...]
```

## Part 2: Merge Most Common Pair

```python
def merge_pair(tokens, pair):
    """Merge a pair throughout the token list"""
    new_tokens = []
    i = 0
    while i < len(tokens):
        # TODO: If next two tokens form the pair, merge them
        # Otherwise, keep token as-is
        # Example: tokens=['a','b','c'], pair=('a','b') → ['ab','c']
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            new_tokens.append(tokens[i] + tokens[i+1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

# Test
tokens = ['t','h','e','</w>','t','h','e','</w>']
merged = merge_pair(tokens, ('h', 'e'))
print(merged)
# Should be: ['t','he','</w>','t','he','</w>']
```

## Part 3: Full BPE Algorithm

```python
def train_bpe(text, vocab_size=100):
    """Train BPE vocabulary"""
    
    # Initialize: each character is a token
    tokens = list(text) + ['</w>']
    vocab = set(tokens)
    
    # TODO: Repeatedly merge most common pair until vocab_size reached
    while len(vocab) < vocab_size:
        # Count all adjacent pairs
        pair_counts = Counter()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pair_counts[pair] += 1
        
        if not pair_counts:
            break
        
        # Find most common
        most_common = pair_counts.most_common(1)[0][0]
        
        # Merge throughout
        new_token = most_common[0] + most_common[1]
        tokens = merge_pair(tokens, most_common)
        vocab.add(new_token)
    
    return sorted(vocab), tokens

vocab, final_tokens = train_bpe("the bible speaks truth", vocab_size=50)
print(f"Vocabulary size: {len(vocab)}")
print(f"Final tokens: {final_tokens}")
print(f"Sample vocab: {sorted(vocab)[:20]}")
```

## Part 4: Encode New Text Using Learned Vocab

```python
def encode_text(text, vocab, merges):
    """Encode new text using trained BPE"""
    
    # Initialize with characters
    tokens = list(text) + ['</w>']
    
    # TODO: Apply merges in order to encode
    # (This is where efficiency matters: we only apply learned merges)
    for merge_pair in merges:
        tokens = merge_pair(tokens, merge_pair)
    
    return tokens

# After training BPE...
# new_text = "the god is"
# encoded = encode_text(new_text, vocab, merges)
```

## Challenge: Measure Fertility

```python
def measure_fertility(texts, vocab_size):
    """
    Fertility = average tokens per word
    Lower fertility = better compression
    """
    
    vocab, _ = train_bpe(" ".join(texts), vocab_size)
    
    total_tokens = 0
    total_words = 0
    
    for text in texts:
        tokens = encode_text(text, vocab, merges)
        words = len(text.split())
        
        total_tokens += len(tokens)
        total_words += words
    
    fertility = total_tokens / total_words
    return fertility

# Compare different vocab sizes
for vs in [100, 1000, 10000]:
    fert = measure_fertility(bible_texts, vs)
    print(f"Vocab size {vs}: fertility {fert:.2f} tokens/word")
    
# Expected: larger vocab → lower fertility (fewer tokens needed)
```

## Real-World Application: MarkGPT Tokenization

```python
# MarkGPT uses sentencepiece library, but principle is same

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers

# Initialize with BPE model
tokenizer = Tokenizer(models.BPE())

# Train on Bible corpus
trainer = trainers.BpeTrainer(vocab_size=4000, min_frequency=2)
tokenizer.train_from_iterator(bible_texts, trainer=trainer)

# Encode
tokens = tokenizer.encode("In the beginning was the Word").ids
print(tokens)  # E.g., [19, 43, 81, 5, 12, ...]
```

## Key Insights

- ✅ BPE is simple: iteratively merge common pairs
- ✅ Low vocabulary: many tokens (high fertility)
- ✅ High vocabulary: fewer tokens (low fertility)
- ✅ Trade-off: computational cost vs. compression ratio
- ✅ Unicode/UTF-8: each byte is a token initially (allows any text!)

## References

- Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation of Rare Words with Subword Units." *ACL*.
- OpenAI Tokenizer: https://github.com/openai/tiktoken
