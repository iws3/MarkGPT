# Day 25 Journal: Tokenization in Depth
## Daily Setup
- **Date**: May 16, 2026
- **Module**: 05 - NLP Foundations
- **Topic**: Subword Tokenization & Byte-Pair Encoding (BPE)
- **Time Spent**: 5 hours

## Goals for Today
- [x] Understand the 3 levels of tokenization (Char, Word, Subword).
- [x] Implement the BPE algorithm from scratch in Python.
- [x] Train BPE on the KJV Bible and analyze the learned merges.
- [x] Calculate "Fertility" and understand its importance for LLMs.

## Notes and Learnings

### Char-level vs Word-level vs Subword
- **Char-level**: Good for small vocabs, but makes sequences too long. Model struggles to link "d-o-g" together over long distances.
- **Word-level**: Great compression, but vocabulary explodes (OOV problem). If the model sees "dogs" but only knows "dog", it's stuck.
- **Subword (BPE)**: The middle ground. It keeps characters for rare words but merges "th", "he", "in" into single tokens. This is what GPT uses.

### How BPE Merges Work
I implemented the merge logic today. The algorithm starts with individual characters and iteratively merges the most frequent pair. 
For example, in the Bible:
1. `e + </w>` -> `e</w>`
2. `t + h` -> `th`
3. `a + n` -> `an`
The result is a vocabulary that adapts to the specific corpus.

### Fertility Metric
Fertility (tokens per word) is a great way to measure how well our tokenizer "knows" the language. 
- On KJV Bible with 50 merges, I got a fertility of around **3.2**.
- As I increase merges to 500+, fertility drops closer to **1.4**.
Lower fertility means the model can "see" more text in the same context window!

## Exercises

### E25.1: BPE from Scratch
I successfully implemented `train_bpe` and `encode` in `day25_exercises.py`. It was satisfying to see the characters slowly clump together into common words like "the", "and", and "Lord".

## Reflection
Tokenization seemed like a boring preprocessing step, but now I see it's actually the foundation of everything. If the tokenizer is bad, the model starts at a disadvantage. I'm excited to see how these tokens turn into **Word Embeddings** tomorrow in Day 26!
