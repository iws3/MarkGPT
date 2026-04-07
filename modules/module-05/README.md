# Module 05 — NLP Foundations: Text as Data
## Days 25–30 | Intermediate

---

## Module Overview

How do machines see language? This module teaches the practical skills for converting text into numbers: tokenization, embeddings, and the craft of feature engineering. You'll also learn about pre-Transformer language models that ruled the era 2018–2019.

By the end of Module 05, you will:
- Implement tokenization from Byte Pair Encoding
- Train word embeddings and visualize them
- Build a text classifier
- Understand ELMo and the shift toward contextual representations

## Learning Objectives

- Understand core ML concepts
- Implement algorithms from scratch
- Relate theory to MarkGPT architecture
- Complete hands-on exercises

## Structure

```
lessons/       - Conceptual explanations with code examples
exercises/     - Practical implementation exercises
projects/      - Larger projects (optional)
resources/     - Additional readings and links
```

## Time Estimate

- Lessons: 4-6 hours
- Exercises: 4-6 hours
- **Total: 8-12 hours per module**

## Key Concepts

[See lesson files for detailed content]

## Completion Checklist

- [ ] Read all lessons (L*_*.md files)
- [ ] Complete all exercises (day*_*.md files)
- [ ] Pass the module quiz (if provided)
- [ ] Understand connections to MarkGPT

## Resources

- Lesson references contain links to papers and tutorials
- http://markgpt-docs.com (forthcoming)
- GitHub discussions: https://github.com/yourusername/MarkGPT-LLM-Curriculum/discussions

## Next Module

See ../module-0$((i+1))/README.md for the next module.
## Tokenization Fundamentals

### What is Tokenization?

Converting text → tokens (numbers)
Token: Smallest unit (word, subword, character)
Required step for all NLP models
Quality crucial: Affects downstream tasks
Trade-off: Granularity vs vocabulary size

### Character-level Tokenization

Simplest: Each character = token
Alphabet size: 26 + digits + punct ≈ 100
Pros: Handles any text (misspellings, OOV)
Cons: Sequences very long, harder to learn
Example: "Hello" → [H,e,l,l,o]
Used in: Character-level language models

### Word-level Tokenization

Split on whitespace and punctuation
Vocab size: 10K-100K typical
Pros: Interpretable, reasonable length
Cons: OOV problem (unknown words)
Example: "Hello world!" → ["Hello", "world", "!"]
Problem: "hello" vs "Hello" = different tokens

### Subword Tokenization

Middle ground: Parts of words
Vocab size: 1K-100K
Examples: Byte-Pair Encoding, WordPiece
Balances word and character levels
Standard in modern NLP
"Hello" → ["He", "llo"] or ["Hel", "lo"]

## Byte-Pair Encoding (BPE)

### Algorithm

1. Start with characters + special symbols
2. Count all adjacent pairs
3. Merge most frequent pair
4. Repeat until vocab size reached
Simple greedy algorithm
Very effective in practice

### BPE Example

Text: "hello hello"
Initial: [h,e,l,l,o, ,h,e,l,l,o]
Step 1: "l" "l" frequent → [h,e,ll,o, ,h,e,ll,o]
Step 2: "h" "e" frequent → [he,ll,o, ,he,ll,o]
Step 3: "he" "ll" frequent → [hell,o, ,hell,o]
Result: [hell, o, </s>, hell, o]

### BPE Advantages

Handles misspellings: "helo" → ["he", "lo"]
Compression: Frequent words = single token
Vocabulary: Finite size (predictable memory)
Language independent: Works on any language
Reversible: Can decode back
Reproducible: Same text → same tokens

## WordPiece Tokenization

### Differences from BPE

Merge criterion: Likelihood maximization
Not just frequency
Used in: BERT, RoBERTa
Similar results to BPE
Slightly different algorithm
Both work well in practice

## SentencePiece

### Language Agnostic

Works on raw text (no preprocessing)
No language-specific logic
Combines BPE and unigram language model
Treats space as token
Great for non-Latin scripts
Used in: T5, mBERT, many recent models

## Vocabulary Size Impact

### Small Vocab (1K)

Sequence length: Very long
Memory per sample: High
Training time: Slow
Parameter count: Lower (embedding matrix)
Typical: Character-level models

### Large Vocab (100K)

Sequence length: Short
Memory per sample: Low
Training time: Fast
Parameter count: Very high
Typical: BERT, GPT
Trade-off: Memory vs speed

## Special Tokens

### Standard Special Tokens

[CLS]: Classification token (start)
[SEP]: Separator (between sentences)
[PAD]: Padding (fill short sequences)
[UNK]: Unknown (OOV words)
[MASK]: Masked token (BERT pre-training)
</s>: End of sequence
<s>: Start of sequence

### Custom Tokens

Task-specific: [QUESTION], [ANSWER]
Entity types: [PER], [LOC], [ORG]
Domain-specific: [CODE], [EQUATION]
Improves performance
Requires fine-tuning
Common in production systems

## Handling OOV Words

### Problem

Word not in vocabulary → [UNK]
Loses information
Subword tokenization helps
"unrecognizable" → ["unrecognizable"]
BPE: ["unrecogniz", "able"]
Preserves information!

### Solutions

1. Subword tokenization (BPE, WordPiece)
2. Character-level fallback
3. Morphological analysis
4. Expand vocabulary
5. Back-off smoothing (pre-training trick)
Best: Combine approaches

## Tokenization Quality Metrics

### Compression Ratio

Average tokens per word
1.0: Perfect (1 token per word)
1.3: Good (3 tokens per 10 words)
2.0: Poor (half as many words)
Impact: Memory and compute
Typical: 1.1-1.3 for English

### Vocabulary Coverage

% of corpus tokens that are in-vocabulary
BERT (30K vocab): 98%+
GPT-2 (50K vocab): 99%+
Smaller vocab: Lower coverage
Affects performance
Trade-off: Size vs coverage

## Contextual Tokenization

### Problem: Ambiguity

"bank" = financial vs river bank
Single tokenization misses context
Morphologically: Same
Solution: Same token, different embeddings
Transformers learn contextual meaning

## Tokenization in Code

### Using HuggingFace

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base')
tokens = tokenizer.encode('Hello world')
# [101, 7592, 2088, 102]
text = tokenizer.decode(tokens)
# 'Hello world'
```

### Custom Tokenizers

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())
# Train on your data
tokenizer.train_from_iterator(texts, vocab_size=50000)
# Use it
encoding = tokenizer.encode('Your text')
```

## Multilingual Tokenization

### Challenges

100+ languages, different scripts
Chinese: Words not separated
Arabic: Right-to-left
Agglutinative: Turkish, Finnish
Solution: Universal tokenizers
SentencePiece: Language-agnostic

### mBERT Tokenization

Shared tokenizer across 104 languages
110K vocabulary
WordPiece trained on multilingual corpus
Works reasonably well
Trade-off: Per-language quality
Enables zero-shot cross-lingual transfer

## Tokenization Speed

### Practical Performance

BERT tokenizer: 10K tokens/s
SentencePiece: 100K tokens/s
Character-level: 1M tokens/s
Bottleneck: Usually data loading
Cache tokens during preprocessing
Pre-tokenize for speed

## Preprocessing Pipeline

### Best Practices

1. Normalize: Lowercasing (task-dependent)
2. Remove: Accents, special chars (careful!)
3. Tokenize: Use standard tokenizer
4. Truncate: Limit length
5. Pad: Make same length for batching
6. Convert to IDs: Vocabulary lookup

## Tokenization Errors

### Common Issues

1. **Case sensitivity**: "Hello" ≠ "hello"
   Solution: Lowercase before tokenizing

2. **Punctuation**: Attached vs separate
   Solution: Check tokenizer behavior

3. **Contractions**: "don't" → [
   Solution: Expand or handle in tokenizer

4. **Whitespace**: Multiple spaces
   Solution: Normalize whitespace

## Vocabulary Learning

### From Unlabeled Data

Train on large corpus
No labels needed
Learn frequent patterns
Adapt to domain
Example: Train on Wikipedia → Reddit
Learn Reddit-specific slang

## Word Embeddings Fundamentals

### What are Word Embeddings?

Dense vectors representing words
Dimension: 50-300 typical
Learned from large text corpus
Similar words → similar vectors
Foundation of modern NLP
Input to neural networks

### Distributional Hypothesis

"You shall know a word by the company it keeps"
Context determines meaning
Words in similar contexts → similar meanings
Learning principle: Co-occurrence statistics
Basis for all embedding methods
Remarkably effective!

