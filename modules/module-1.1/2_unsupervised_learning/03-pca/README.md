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