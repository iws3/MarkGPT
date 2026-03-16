# Lesson 5: Training Embeddings
## How Models Learn Representations

## Table of Contents
- Supervised vs Unsupervised Learning
- Word2Vec and Skip-Gram
- GloVe and Co-Occurrence
- Contrastive Learning
- Fine-tuning Embeddings
- Embedding Quality Metrics
- Data Considerations

---

## Supervised vs Unsupervised Learning

Embeddings can be trained in supervised settings (with labels) or unsupervised settings (from raw text).

Word2Vec and GloVe are classic unsupervised methods; modern approaches often use self-supervised objectives to learn richer representations.

---

## Practical Tip

When training embeddings, start with a small vocabulary and a clean dataset. Inspect your embeddings by projecting them into 2D (e.g., using t-SNE) to ensure that semantically similar items cluster together.
