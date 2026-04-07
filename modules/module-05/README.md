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

