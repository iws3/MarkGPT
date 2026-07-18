# Tokenization & BPE (Byte-Pair Encoding)

## Why Tokenization?

Neural networks need fixed-dimensional inputs. Text is variable-length strings.

Solution: Break text into tokens (subword units) that map to fixed vocabulary.

```
Text: "The MarkGPT model is great"
     ↓ Tokenization
Tokens: [1256, 42, 1047, 19, 521]
     ↓ Embedding
Embeddings: (batch, seq_len, d_model)
```

---

## Character-Level Tokenization

### Simplest Approach

```python
def char_tokenize(text, vocab=None):
    """Map each character to ID."""
    
    if vocab is None:
        vocab = {
            '<pad>': 0, '<unk>': 1, '\n': 2,  # Special
            **{chr(i): i+3 for i in range(32, 127)}  # ASCII printable
        }
    
    return [vocab.get(char, vocab['<unk>']) for char in text]

# Example
vocab = build_char_vocab()
tokens = char_tokenize("Hi!", vocab)  # [72, 105, 33] (ASCII values)
```

### Issues

- Sequence length explodes: "hello world" = 11 tokens vs 1 word
- Doesn't capture linguistic units ("ing", "ed" as suffixes)

---

## Word-Level Tokenization

### Simple Approach

```python
def word_tokenize(text, vocab=None):
    """Tokenize to words (space-separated)."""
    
    words = text.split()
    
    if vocab is None:
        vocab = build_word_vocab(text)
    
    return [vocab.get(word, vocab['<unk>']) for word in words]

# Example
words = word_tokenize("The cat sat on the mat")
# ['The', 'cat', 'sat', 'on', 'the', 'mat']
```

### Issues

- **Vocabulary explosion**: Need entry for every word
  - English: ~170,000 words
  - Include names, rare words, new slang: 500K+
- **Out-of-vocabulary (OOV)**: Unseen words → `<unk>`
- **Doesn't share morphology**: "running" and "run" are separate

---

## Byte-Pair Encoding (BPE)

### Motivation

Subword tokenization: Compromise between character and word level.

- Vocabulary: ~10K-50K tokens
- Sequence length: Reasonable
- Handles OOV naturally

### Algorithm

**Step 1: Start with character tokens**

```
Text: "low low low"
Initial: [l, o, w, l, o, w, l, o, w]
Merges: {}
```

**Step 2: Find most frequent pair, merge**

```
Frequency count:
  (l, o): 3 ✓ Most frequent
  (o, w): 3
  ...

Merge (l, o) → lo
Text: [lo, w, lo, w, lo, w]
Merges: {(l, o): lo}
```

**Step 3: Repeat**

```
Frequency count:
  (lo, w): 3 ✓
  ...

Merge (lo, w) → low
Text: [low, low, low]
Vocabulary: [l, o, w, lo, low]
```

### Implementation

```python
import re
from collections import Counter

def bpe_tokenize(text, merges=None, num_merges=1000):
    """Byte-pair encoding."""
    
    if merges is None:
        merges = {}
    
    # Start with characters
    words = text.split()
    
    for word in words:
        # Character-level tokens
        tokens = list(word) + ['</w>']  # </w> marks word boundary
        
        # Perform merges
        for _ in range(num_merges):
            # Find most frequent pair
            pair_counts = Counter()
            for token_seq in words:
                for i in range(len(token_seq) - 1):
                    pair = (token_seq[i], token_seq[i+1])
                    pair_counts[pair] += 1
            
            if not pair_counts:
                break
            
            most_common = pair_counts.most_common(1)[0][0]
            
            # Merge this pair
            new_token = ''.join(most_common)
            merges[most_common] = new_token
            
            # Update sequences
            # (Implementation: apply merge to all words)
    
    return merges

# Usage
merges = bpe_tokenize("low low low", num_merges=100)
```

### MarkGPT Tokenization

```python
class MarkGPTTokenizer:
    def __init__(self, vocab_size=10000, merges=None):
        self.vocab_size = vocab_size
        self.merges = merges or {}  # Pre-computed BPE merges
        self.vocab = self.build_vocab()
    
    def encode(self, text):
        """Text → token IDs."""
        tokens = []
        for word in text.split():
            word_tokens = self._encode_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def decode(self, token_ids):
        """Token IDs → text."""
        tokens = [self.vocab[tid] for tid in token_ids]
        return ''.join(tokens).replace('</w>', ' ').strip()
    
    def _encode_word(self, word):
        """Encode single word using BPE."""
        # Start with characters
        chars = list(word) + ['</w>']
        
        # Apply merges
        for merge in self.merges.values():
            chars = self._apply_merge(chars, merge)
        
        # Map to IDs
        return [self.vocab[token] for token in chars]
    
    def _apply_merge(self, tokens, merge_pair):
        """Merge a pair in token sequence."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == merge_pair:
                new_tokens.append(''.join(merge_pair))
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
```

---

## SentencePiece (Google's Approach)

Instead of BPE pairs, train on raw bytes + statistics:

```python
import sentencepiece as spm

# Train tokenizer
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='markgpt',
    vocab_size=10000,
    model_type='bpe'
)

# Load and use
tokenizer = spm.SentencePieceProcessor('markgpt.model')
tokens = tokenizer.encode("Hello world")  # [1, 2, 3]
text = tokenizer.decode(tokens)           # "Hello world"
```

---

## Multilingual Tokenization (MarkGPT)

### Challenge

English and Banso have different character encodings.

```
English (Latin): A-Z a-z 0-9
Banso (if Xhosa, Zulu): Click consonants (ǰ, ǀ, ǁ, ǂ)
```

### Solution: Unified BPE Vocabulary

```python
class MultilingualTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        
        # Reserve tokens
        special_tokens = 256  # <pad>, <unk>, <bos>, etc.
        
        # Split vocabulary
        self.english_tokens = vocab_size // 2  # 5000
        self.banso_tokens = vocab_size // 2    # 5000
        
        self.english_vocab_start = special_tokens
        self.banso_vocab_start = special_tokens + self.english_tokens
    
    def tokenize(self, text, language='en'):
        """Tokenize with language awareness."""
        if language == 'en':
            tokens = self.bpe_tokenize(text, 'english')
            tokens = [t + self.english_vocab_start for t in tokens]
        elif language == 'banso':
            tokens = self.bpe_tokenize(text, 'banso')
            tokens = [t + self.banso_vocab_start for t in tokens]
        
        return tokens
```

### Data Format

```
<BOS> English text here </EOS> <BOS> Banso text here </EOS>
```

---

## Tokenization Statistics

```python
def analyze_tokenization(text, tokenizer):
    """Understand tokenization efficiency."""
    
    tokens = tokenizer.encode(text)
    
    # Compression ratio
    char_count = len(text)
    token_count = len(tokens)
    compression = char_count / token_count
    
    print(f"Text length: {char_count} chars")
    print(f"Token count: {token_count} tokens")
    print(f"Compression: {compression:.2f} (chars per token)")
    
    # Common tokens
    from collections import Counter
    token_freq = Counter(tokens)
    
    print("\nMost common tokens:")
    for token_id, count in token_freq.most_common(10):
        token_text = tokenizer.decode([token_id])
        print(f"  {token_id}: '{token_text}' ({count} times)")
```

---

## Padding & Masking

For batching variable-length sequences:

```python
def batch_encode(texts, tokenizer, max_length=512):
    """Encode multiple texts with padding."""
    
    batch_tokens = []
    batch_mask = []
    
    for text in texts:
        tokens = tokenizer.encode(text)
        
        # Truncate if too long
        tokens = tokens[:max_length]
        
        # Pad to max_length
        padding = max_length - len(tokens)
        tokens = tokens + [tokenizer.pad_token_id] * padding
        
        # Attention mask (1 for real tokens, 0 for padding)
        mask = [1] * (len(tokens) - padding) + [0] * padding
        
        batch_tokens.append(tokens)
        batch_mask.append(mask)
    
    return torch.tensor(batch_tokens), torch.tensor(batch_mask)

# In model forward:
logits = model(input_ids, attention_mask=attention_mask)
```

---

## Tokenization Pipeline

```python
class TokenizationPipeline:
    def __init__(self, vocab_size=10000, max_length=512):
        self.tokenizer = MarkGPTTokenizer(vocab_size)
        self.max_length = max_length
    
    def process(self, text, padding=True, truncation=True):
        """Full preprocessing pipeline."""
        
        # Normalize text
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces → single
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate
        if truncation and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Pad
        if padding:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        # Attention mask
        mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in tokens]
        
        return {
            'input_ids': torch.tensor(tokens),
            'attention_mask': torch.tensor(mask)
        }
```

---

**Tokenization & BPE v1.0**
**Last Updated**: 2024
