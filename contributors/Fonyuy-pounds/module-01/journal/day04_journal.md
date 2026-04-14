# Day 04: Probability and Information Theory
**Date:** 2026-04-14
**Theme:** Language models are probability machines

## 📖 Complete Lesson Guide

### L04.1: Conditional Probability
Language models don't just predict words; they predict **continuations**.
- **Question:** What is the probability of word $B$ given that we just saw word $A$?
- **Math:** $P(B|A) = \frac{\text{Count}(A, B)}{\text{Count}(A)}$
- **Example:** If "beginning" appears 100 times, and "beginning God" appears 40 times, $P(\text{God}|\text{beginning}) = 0.4$.

### L04.2: The Chain Rule of Probability
To find the probability of a whole sentence, we multiply the conditional probabilities:
$$P(w_1, w_2, w_3) = P(w_1) \times P(w_2|w_1) \times P(w_3|w_1, w_2)$$
In a **Bigram Model**, we simplify this by assuming $w_3$ only depends on $w_2$, not $w_1$. This is the "Markov Assumption".

### L04.3: Shannon Entropy
Entropy ($H$) measures **uncertainty** or **surprise**.
- **High Entropy:** The model is confused (many words are equally likely).
- **Low Entropy:** The model is confident (one word is very likely).
- **Formula:** $H(X) = - \sum p(x) \log_2 p(x)$ (measured in **bits**).

### L04.4: Cross-Entropy Loss
This is how we grade the model during training.
- We compare the model's predicted probability ($q$) with the actual next word ($p$).
- If the model gives low probability to the correct word, the **Loss** is high.
- **Goal:** Minimize cross-entropy loss by adjusting model parameters.

### L04.5: Perplexity
Perplexity (PPL) is a more intuitive version of cross-entropy.
- It tells you: "On average, how many words is the model choosing between?"
- **Math:** $PPL = 2^H$ (where $H$ is cross-entropy).
- **Example:** If $PPL = 10$, the model is as confused as if it had to pick 1 from 10 equal options at every step. Lower is better!

---

## ✍️ Exercises

### E04.1: Entropy Calculations (Manual)
1.  **Fair Coin ($0.5/0.5$):** $H = 1.0$ bit.
2.  **Loaded Coin ($0.9/0.1$):** $H \approx 0.47$ bits.
3.  **4-sided Die ($0.25$ each):** $H = 2.0$ bits.

### E04.2: Bigram Model (In Python)
- **Goal:** Implement the logic in `day04_bigram_starter.py`.
- **Logic:** Count word pairs, normalize to probabilities, and generate text.

### E04.3: Perplexity (In Python)
- **Goal:** Calculate how "surprised" your bigram model is by the last 10% of Genesis.

---

## 🧠 Reflection
**1. Why is a bigram model limited?**
It only "sees" one word back. It can't remember that a sentence started with a character name 5 words ago.

**2. What happens to perplexity with Laplace smoothing?**
Smoothing gives a tiny bit of probability to "impossible" transitions. This prevents the model from hitting $\infty$ perplexity when it sees a new word pair, making it more robust.
