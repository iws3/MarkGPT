# Day 5 Exercises: N-Gram Models

## Overview
Day 5 focused on N-gram statistical language models, smoothing techniques, and their limitations. I implemented a Trigram model and compared its performance with and without Laplace smoothing.

## Results

### Model Performance (Perplexity)
The following metrics were calculated on a held-out test set (10% of the small Genesis sample):

| Model Type | Perplexity | Notes |
|---|---|---|
| **No Smoothing** | 58,846.15 | Extremely high due to zero-probability events for unseen trigrams. |
| **Laplace Smoothing (α=1.0)** | 40.90 | significantly improved by assigning probability to unseen sequences. |

### Text Generation Samples
Generated from seed: `"and god"`

**Raw Model (No Smoothing):**
> `and god called the light that it was so`

**Smoothed Model:**
> `and god saw spirit void it grass the land which yielding place was said from beginning of under whose of created without good seed from he which evening herb second night darkness`

### Key Learning
While Laplace smoothing is essential for making statistical models functional on unseen data (by avoiding the product-of-probabilities being zero), it can introduce "noise" in generation. In small datasets, the total probability mass assigned to the large number of unseen trigrams can outweigh seen sequences, leading to less coherent generation.

## Reflections
The "Colorless green ideas sleep furiously" experiment (Exercise E05.3) highlighted that language has a structural depth that simple frequency statistics cannot fully capture. This realization marks the transition point in the curriculum from classical statistical methods to modern neural networks.
