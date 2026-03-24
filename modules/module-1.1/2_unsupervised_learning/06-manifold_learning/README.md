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

### t-SNE: Non-linear Dimensionality Reduction for Visualization

t-Distributed Stochastic Neighbor Embedding (t-SNE) is a popular manifold learning technique designed for visualization. t-SNE converts high-dimensional point similarities to low-dimensional Euclidean distances while preserving local neighborhood structure. In high dimensions, similarities are computed as P(j|i) = exp(-||x_i - x_j||²/(2σ_i²)) / Σ_k≠i exp(-||x_i - x_k||²/(2σ_i²)), balancing local and global structure. In low dimensions, t-SNE uses the t-distribution for robustness: Q(j|i) = (1 + ||y_i - y_j||²)^(-1) / Σ_k≠i (1 + ||y_i - y_k||²)^(-1). Kullback-Leibler divergence between P and Q is minimized using gradient descent. t-SNE excels at discovering cluster structure and reproducing local neighborhoods but doesn't preserve global distances; close points in the original space remain close in t-SNE space, but distant points may or may not remain distant. Therefore, t-SNE is most suitable for visualization rather than building features for downstream learning.