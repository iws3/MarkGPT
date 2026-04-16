# Day 05: N-Gram Models: The Pre-Neural Era
**Date:** 2026-04-16
**Theme:** Language modeling before neural networks

## 📖 Complete Lesson Guide

### L05.1: Unigrams, Bigrams, Trigrams
Moving beyond single words to sequences of length $N$.
- **Trigrams:** Predict $P(w_3 | w_1, w_2)$.
- **Intuition:** More context ($N$) usually means better predictions, but leads to **data sparsity** issues.
- **Challenge:** As $N$ increases, the number of possible sequences grows exponentially, and most will never appear in the training data.

### L05.2: Smoothing Techniques
How to handle "zero frequency" sequences.
- **Laplace (Add-one) Smoothing:** Adds a small constant $\alpha$ to all counts.
- **Formula:** $P(w_n | w_{n-1}) = \frac{\text{Count}(w_{n-1}, w_n) + \alpha}{\text{Count}(w_{n-1}) + \alpha |V|}$
- **Alternative:** Kneser-Ney smoothing (more advanced, uses the diversity of histories).

### L05.3: The Limits of N-Gram Models
- **Long-range dependencies:** N-grams cannot bridge dependencies that span more than $N-1$ words.
- **Syntactic Ignorance:** They don't understand sentence structure, only sequence frequency.
- **The "Colorless green ideas" problem:** Perfectly grammatical but nonsensical sentences receive zero probability.

### L05.4: Why These Limits Led to Neural Networks
Neural networks (Transformers, LSTMs) address these by:
1. Learning **distributed representations** (embeddings) instead of discrete counts.
2. Capturing **long-range dependencies** without exponential parameter growth.
3. Generalizing across similar contexts.

---

## ✍️ Exercises

### E05.1: Trigram Model (In Python)
- **Implementation:** Created `day05_solutions.py` with `TrigramLanguageModel`.
- **Observation:** Trigram generation is noticeably more coherent than bigrams ("and god called the light that it was so"), but still fails on long passages.

### E05.2: Laplace Smoothing Experiment
- **Without Smoothing:** Test perplexity was near infinity (**58,846.15**) due to unseen trigrams.
- **With Smoothing (α=1.0):** Test perplexity dropped to **40.90**.
- **Insight:** Smoothing is essential for evaluation but can make generation "noisy" on small datasets.

### E05.3: "Colorless green ideas" ESSAY
- **Core Argument:** N-gram models capture *distributional probability* (what was said) rather than *generative grammar* (what *could* be said). This is why they fail on novel yet grammatically correct sentences.

---

## 🧠 Reflection
**1. What surprised me today?**
How drastically the perplexity improves with even simple Laplace smoothing. It's the difference between a model that works and one that crashes on the first new word.

**2. What is the biggest hurdle for N-Grams in low-resource settings?**
Data sparsity. If you only have a small corpus (like Banso proverbs), you will hit "zero counts" constantly. Neural models that can share information between related words (embeddings) are likely much more suited for this.
