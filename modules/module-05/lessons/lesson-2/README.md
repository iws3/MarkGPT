# Lesson 2: Tokenization Deep Dive
## Breaking Text into Model-Friendly Units

## Table of Contents
- What Is Tokenization?
- Tokenization Strategies
- Subword Tokenization (BPE, WordPiece)
- Tokenization and Vocabulary
- Tokenization for Multiple Languages
- Tokenization Tools
- Tokenization Pitfalls
- Practical Examples

---

## What Is Tokenization?

Tokenization is the process of splitting text into meaningful units (tokens) that a model can process. A good tokenizer balances expressivity, vocabulary size, and efficiency.

Tokenization is a critical preprocessing step for any language model.

---

## Quick Exercise

Try tokenizing a sentence with different tokenizers (e.g., whitespace, BPE, WordPiece) and compare the token count. Note how subword tokenization handles rare words.

This exercise will help you see the practical tradeoffs between vocabulary size and tokenization granularity.
