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

### t-SNE Perplexity and Convergence Tuning

t-SNE has two key hyperparameters: perplexity (effective neighborhood size, typically 5-50) and learning rate. Perplexity roughly equals: effective neighborhood size for distance computations. Small perplexity (5-10) focuses on local structure; large perplexity (50+) emphasizes global structure. The trade-off is non-trivial: small perplexity often produces fragmented clusters; large perplexity can blur cluster boundaries. Typical choice is perplexity = 30, though tuning by visualization is common. t-SNE is sensitive to hyperparameters; slightly different settings produce noticeably different visualizations. Early exaggeration (multiplying attractive forces initially) helps separate clusters. Multiple runs with different random seeds produce different layouts; variations are expected. t-SNE usually requires hundreds of iterations to converge; monitoring loss (KL-divergence) reveals convergence. Computation is O(n²), making t-SNE slow for n > 10k (approximate versions like Barnes-Hut speed it up to O(n log n)).

### Isomap and Geodesic Distances

Isomap addresses PCA's limitation when data lies on non-linear manifolds. Computing geodesic distances (along the manifold) rather than Euclidean distances (through space) better captures intrinsic dimensionality. Procedure: (1) Build k-NN graph (connect each point to k nearest neighbors); (2) Compute shortest paths using Dijkstra's algorithm; (3) Apply classical scaling to geodesic distance matrix. Geodesic distances approximate true manifold distances; the k-NN graph reconstruction quality depends on k. Small k (too local) fragments graph; large k (too global) ignores manifold curvature. The number of components k is chosen via explained variance similar to PCA. Isomap provides stable dimension estimates (how many dimensions the manifold truly has) via spectral analysis. The algorithm scales as O(n³) (all-pairs shortest paths), limiting n to ~10k. For very large datasets, approximate Isomap or other methods are necessary.

### Locally Linear Embedding and Local Structure Preservation

LLE (Locally Linear Embedding) assumes each neighborhood lies near a linear patch of the manifold. Algorithm: (1) For each point, minimize reconstruction error (distance from point to linear combination of neighbors); (2) Weights capture local linear structure; (3) Apply these weights to low-dimensional embeddings, preserving local relationships. High-dimensional local linearity implies low-dimensional local linearity; this principle underlies LLE. Unlike Isomap (which requires computing all-pairs distances), LLE only requires k-NN neighbors, scaling better. Parameter k (neighborhood size) is crucial: too small (noisy), too large (destroys local structure). Number of components is chosen similarly to PCA. LLE often produces interpretable embeddings; nearby samples in input space remain nearby in embedding. However, unlike t-SNE, LLE doesn't emphasize cluster separation; it focuses on preserving local geometry.