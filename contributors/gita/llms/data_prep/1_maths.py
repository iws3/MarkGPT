# Simulated text corpus with realistic word frequencies
data = (
    ["the"] * 45 + ["a"] * 35 + ["is"] * 30 + ["and"] * 28 +
    ["to"] * 25 + ["in"] * 22 + ["of"] * 20 + ["cat"] * 18 +
    ["dog"] * 15 + ["on"] * 12 + ["sat"] * 10 + ["ran"] * 8 +
    ["house"] * 7 + ["with"] * 6 + ["quick"] * 5 + ["brown"] * 4 +
    ["jumps"] * 3 + ["lazy"] * 2 + ["fox"] * 2 + ["over"] * 3
)

vocab = [
    "the", "a", "is", "and", "to", "in", "of", "cat", "dog", "on",
    "sat", "ran", "house", "with", "quick", "brown", "jumps", "lazy", "fox", "over"
]

import numpy as np

# Convert words to numerical indices for processing
vocab_size = len(vocab)  # Total number of unique words (20)
ids = np.array([vocab.index(w) for w in data])  # Map each word to its vocab index
freq = np.bincount(ids, minlength=vocab_size) / len(ids)  # True frequency distribution

# Initialize logits (raw scores before softmax) - start with no preference
z = np.zeros(vocab_size)  # All zeros = uniform distribution after softmax

# Learning rate: controls how big our update steps are
lr = 0.5

# Training loop: learn the word frequency distribution
for step in range(201):
    # === Forward pass: convert logits to probabilities ===
    e = np.exp(z - z.max())  # Exponentiate (subtract max for numerical stability)
    p = e / e.sum()  # Normalize to get probability distribution (softmax)

    # === Compute loss: how surprised are we by the actual data? ===
    loss = -np.log(p[ids]).mean()  # Cross-entropy loss (average negative log-likelihood)

    # === Backward pass: compute gradient ===
    grad = p - freq  # Gradient tells us how to adjust z to match freq

    # === Update step: gradient descent ===
    z -= lr * grad  # Move logits in direction that reduces loss

    # Print progress at key steps to see convergence
    if step in (0, 1, 2, 5, 10, 50, 200):
        print(f"Step {step}: Loss = {round(loss, 4)}")
        print(f"  Top 5 words: {[(vocab[i], round(p[i], 4)) for i in np.argsort(p)[-5:][::-1]]}\n")
