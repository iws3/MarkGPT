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

### Isomap and Locally Linear Embedding

Isomap extends multidimensional scaling by using geodesic distances (distances along the manifold) rather than Euclidean distances. A k-nearest neighbor graph is constructed, with edge weights as Euclidean distances. Shortest paths in this graph approximate geodesic distances. Classical multidimensional scaling is applied to the geodesic distance matrix, producing a low-dimensional embedding that preserves manifold structure. Isomap discovers the underlying manifold structure better than PCA when the manifold has non-trivial topology. Locally Linear Embedding (LLE) assumes each point and its k-nearest neighbors lie on a locally linear patch of the manifold. LLE reconstructs each point as a linear combination of its neighbors: x_i ≈ Σ_j w_{ij}·x_j, minimizing reconstruction error. The same weights are applied in low dimensions to preserve local linear structure. LLE captures manifold structure through local neighborhoods and often produces interpretable embeddings.

### Applications and Computational Considerations

Manifold learning is valuable for visualization (t-SNE), feature learning (Isomap, LLE), and data exploration in high-dimensional spaces. These methods have revealed interesting structure in digit images, face images, natural language documents, and many other domains. However, manifold learning methods lack theoretical guarantees and have hyperparameters (neighborhood size for LLE/Isomap, perplexity for t-SNE) requiring careful selection. Computational complexity ranges from O(n²) to O(n³), making these methods less scalable than PCA. t-SNE has no inverse mapping, so new points cannot be projected. Recent advances include parametric t-SNE and parametric UMAP enabling inverse mappings. UMAP (Uniform Manifold Approximation and Projection) provides faster alternatives to t-SNE with better theoretical foundations. Despite computational costs, manifold learning remains essential for exploratory data analysis and has contributed significantly to understanding structure in complex high-dimensional datasets.