# Lesson 3: Positional Encoding and Sequences
## Giving Transformers a Sense of Order

## Table of Contents
- Why Position Matters
- Sinusoidal Positional Encoding
- Learnable Positional Embeddings
- Relative vs Absolute Position
- Sequence Lengths and Memory
- Positional Encoding in Practice

---

## Why Position Matters

Transformers process tokens in parallel, so they need a way to know the order of tokens. Positional encodings inject this information into token representations.

This lesson explores techniques for conveying sequence order to transformer models.

---

## Implementation Note

If you use learnable positional embeddings, make sure you handle longer sequences by either extending the embedding table or using relative positions. Otherwise, your model may fail on longer inputs.
