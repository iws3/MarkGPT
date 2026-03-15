# Lesson 4: Measuring Similarity
## Distance Metrics and Their Meaning

## Table of Contents
- Why Similarity Matters
- Cosine Similarity
- Euclidean Distance
- Dot Product
- Choosing the Right Metric
- Similarity in High Dimensions
- Applications in Retrieval
- Experimenting with Metrics

---

## Why Similarity Matters

In embedding space, similarity measures help us find related words, sentences, or documents. Choosing the right metric affects retrieval quality, clustering, and downstream tasks.

This lesson explains common metrics and their tradeoffs.

---

## Example: Cosine Similarity Calculation

Given two vectors, cosine similarity is:

```
cosine_sim(a, b) = (a ⋅ b) / (||a|| * ||b||)
```

This value ranges from -1 (opposite) to 1 (identical). It is widely used in embedding-based systems.