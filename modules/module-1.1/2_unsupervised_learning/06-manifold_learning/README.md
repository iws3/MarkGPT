# Manifold Learning (t-SNE, UMAP)

## Fundamentals

Manifold learning algorithms assume data lies on a low-dimensional manifold embedded in high-dimensional space. t-SNE and UMAP excel at visualization by preserving local and global structure. Unlike PCA (linear), manifold learning methods can capture non-linear structure. These methods are primarily for exploration and visualization rather than preprocessing for supervised learning.

## Key Concepts

- **t-SNE**: Local structure preservation, perplexity
- **UMAP**: Unified Manifold Approximation and Projection
- **Non-linear Dimensionality Reduction**: Beyond linear projections
- **Visualization**: 2D/3D representation of high-dimensional data
- **Local vs Global Structure**: Trade-offs in preservation

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Non-linear Dimensionality Reduction and Manifold Hypothesis

Manifold learning encompasses techniques that assume high-dimensional data lies near a lower-dimensional manifold—a continuous surface embedded in high-dimensional space. Unlike PCA which finds linear subspaces, manifold learning methods discover non-linear structure. The manifold hypothesis posits that high-dimensional data with high geometric complexity may have low intrinsic dimensionality when measured along the manifold. Manifold learning is motivated by phenomena like human perception: images of faces or handwritten digits, though high-dimensional (thousands of pixels), have low intrinsic dimensionality because only a few factors of variation (pose, lighting, digit style) control them. Discovering these factors and reducing to a low-dimensional representation that preserves manifold structure is the goal of manifold learning. This is valuable when data lies on complex non-linear structure that linear methods like PCA fail to capture.