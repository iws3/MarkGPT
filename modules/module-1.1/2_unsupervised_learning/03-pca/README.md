# Principal Component Analysis (PCA)

## Fundamentals

PCA is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. PCA finds orthogonal directions (principal components) that capture maximum variance in the data. Applications include visualization of high-dimensional data, noise reduction, and preprocessing for other algorithms. PCA is unsupervised and doesn't consider target labels, making it purely exploratory.

## Key Concepts

- **Principal Components**: Eigenvectors of covariance matrix
- **Explained Variance**: Proportion of variance retained
- **Scree Plot**: Variance by component
- **Cumulative Variance**: Total variance explained

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Dimensionality Reduction and Variance Maximization

Principal Component Analysis (PCA) is an unsupervised linear dimensionality reduction technique that identifies new features (principal components) that capture maximum variance in the data. The first principal component is the direction of maximum variance; the second principal component is orthogonal to the first and captures the second-most variance; and so on. These directions are eigenvectors of the data covariance matrix, ordered by descending eigenvalues (which represent variance along each direction). By selecting the top k principal components, we project high-dimensional data into a lower-dimensional subspace while retaining as much variance as possible. This variance-maximizing property ensures that important structure in data is preserved while removing noise and redundancy. PCA is unsupervised in that it does not use class labels; it discovers structure purely from feature correlations.

### Computational Implementation and Interpretation

Concretely, PCA involves computing the covariance matrix C = 1/n · X^T·X where X is the centered data matrix. The eigendecomposition of C yields eigenvectors (principal directions) and eigenvalues (variance along each direction). The top k eigenvectors form a projection matrix W, and data is projected as Y = X·W. Standard implementations use Singular Value Decomposition (SVD) rather than eigendecomposition directly, which is numerically more stable and efficient. The explained variance ratio for the k-th component is the percentage of total variance captured by that component, computed as λ_k / Σλ_i. Cumulative explained variance shows the proportion of total variance captured by the top k components. A scree plot shows eigenvalues (or explained variance) for each component; an elbow in this plot suggests the number of components to retain. Typically, components explaining 80-95% of total variance are retained, though application-specific requirements may differ.

### Feature Reconstruction and Loss Analysis

When projecting to lower dimensions, information is lost unless all variance is retained. The reconstruction error ||X - X_reconstructed||² measures information loss; perfectly retaining all variance means zero reconstruction error, while retaining only a fraction means non-zero error. The reconstruction error equals the sum of eigenvalues (variances) associated with discarded components. Therefore, minimizing retained components minimizes reconstruction error, directly reflecting our variance-maximization principle. For visualization purposes (k=2 or k=3), we accept significant reconstruction error in exchange for interpretable visualizations. For preprocessing steps in supervised learning, we select k to balance reconstruction error and computational efficiency; higher dimensionality better preserves information but increases downstream computation. Cross-validation can assess how dimensionality selection affects supervised learning performance.

### Limitations and Extensions

PCA identifies linear relationships and may miss nonlinear structure in data. Kernel PCA extends PCA to nonlinear dimensionality reduction by implicitly working in a high-dimensional kernel space. For data with highly non-Gaussian distributions, Independent Component Analysis (ICA) finds statistically independent components rather than maximum variance directions. PCA assumes continuous variables and can be affected by outliers due to the covariance matrix. Robust PCA variants use robust covariance estimators to handle outliers. For categorical variables, Multiple Correspondence Analysis (MCA) extends PCA logic. PCA whitening (normalizing by standard deviations) ensures all components have unit variance, which can be beneficial for downstream algorithms. Incremental PCA processes data in batches, enabling PCA on datasets too large to fit in memory. Despite limitations, PCA remains fundamental in unsupervised learning for visualization, denoising, and dimensionality reduction as a preprocessing step.

### Explained Variance and Choosing Number of Components

The explained variance ratio for each principal component shows what fraction of total variance it captures. Components are ordered by decreasing variance; the first component is the direction of maximum variance. A scree plot (explained variance for each component) reveals natural cutoffs: choosing components explaining ~90% of variance is standard, though 80% or 95% are common. The cumulative explained variance reaches 90% at component k suggesting k dimensions suffice. However, this assumes linear relationships; nonlinear structure might require more components. In practice, you compute PCA for many components, plot explained variance, choose k manually based on the chart, then recompute PCA with k components. For dimensionality reduction from 100 to 10 dimensions, examining whether 10 components explain 80-95% variance guides whether significant information is retained. The choice balances: reconstruction error (difference between original and reconstructed data) vs computational/interpretability benefits of lower dimensions.

### Centering, Scaling, and Standardization

PCA is sensitive to feature scaling. Features with large variances dominate principal directions. A feature with range [0, 10000] will dominate one with range [0, 1], even if the latter is more informative. StandardScaler (zero mean, unit variance) is essential preprocessing: PCA should be applied to scaled data (via fit-transform on scaled data). Failing to scale is a common mistake producing misleading results. Additionally, PCA centers data (subtracts mean), essential for the covariance matrix computation. Standardization and centering together ensure numerical stability. For features measured in different units (height in cm, weight in kg), scaling is not optional. In scikit-learn, PCA automatically centers data but not scales; always use StandardScaler beforehand.

### Interpretability of Principal Components

Principal components are linear combinations of original features; interpreting them reveals feature relationships. For images, PC1 might be 'brightness', PC2 'contrast'. For gene expression, early PCs capture major variation (often broad biological differences); later PCs capture subtle variation (noise or rare biology). Component loadings (coefficients in linear combination) show feature contributions: large positive loading means feature contributes positively; large negative, negatively. Visualizing heatmaps of loadings reveals which features drive each component. This interpretation aids understanding: in healthcare data, if PC1 is heavily loaded on immune markers, PC1 represents immune system state. However, orthogonal constraint (components are perpendicular) sometimes creates components mixing multiple interpretations. A visualization of data projected onto PC1-PC2 plane shows clustering and structure. This two-dimensional representation is invaluable for exploring high-dimensional data.

### PCA for Preprocessing and Denoising

Beyond visualization, PCA removes noise effectively. High-dimensional data's tail (components explaining <1% variance) is mostly noise. Projecting onto leading components, then reconstructing, removes noise without losing structure. This denoising improves downstream algorithms: classification on PCA-reduced features often outperforms original features (noise removed, fewer features reduce overfitting). Sparse PCA selects a subset of original features (rather than all) for components; interpretability improves at slight accuracy cost. Sparse components use fewer features, simpler to explain. Incremental PCA processes data in batches, enabling PCA on datasets too large for memory. This is valuable for streaming data or massive datasets. Mini-batch approach: process chunks, accumulate statistics, finalize PCA. Limited precision (data not seen simultaneously) is acceptable for denoising applications.

### Kernel PCA and Non-linear Structure

Standard PCA finds linear subspaces; if data's structure is non-linear (lie on curves or manifolds), linear subspaces miss structure. Kernel PCA addresses this: implicitly projects data to high-dimensional space via kernels, then applies PCA. RBF kernel enables discovering non-linear structure. A dataset with spiral structure (linear PCA produces tangled projections) is beautifully separated by kernel PCA. However, kernel PCA has computational cost O(n²) (kernel matrix size), and interpreting non-linear components is harder. For non-linear structure, manifold learning (t-SNE, Isomap, UMAP) sometimes outperforms kernel PCA. The tradeoff: kernel PCA provides mathematically principled non-linearity; manifold methods are sometimes more intuitive. When to use: linear structure suggesting standard PCA, non-linear suggesting manifold methods, and when theoretical grounding matters, kernel PCA is appropriate.