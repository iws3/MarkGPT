# Eigenvalues and Principal Component Analysis
## Comprehensive Learning Guide

## Eigenvalues and Eigenvectors

Eigenvectors are special vectors only stretched without changing direction.

Eigenvalues reveal how much stretching happens in each eigenvector direction.

The characteristic polynomial det(A - λI) has roots equal to the eigenvalues.

Symmetric matrices have special properties enabling diagonalization and decomposition.

## Singular Value Decomposition (SVD)

SVD extends eigendecomposition to rectangular matrices: A = U @ Σ @ V^T.

Singular values measure how much the matrix stretches vectors in directions.

SVD reveals the rank of a matrix through non-zero singular values.

SVD simplifies many problems: least-squares, approximation, compression.

## Principal Component Analysis (PCA)

PCA reduces dimensionality while preserving as much variance as possible.

Principal components are directions ordered by variance they capture.

Projecting onto principal components reduces dimensionality efficiently.

Practical PCA requires careful scaling and selection of component numbers.


## Advanced Dimensionality Reduction

Kernel PCA handles non-linear patterns unlike linear PCA.

Independent Component Analysis finds independent sources.

Factor Analysis models shared variance between variables.

Multidimensional Scaling preserves pairwise distances.

t-SNE visualizes high-dimensional data revealing clusters.

UMAP offers scalable alternative to t-SNE.


## Feature Selection Methods

Variance-based selection removes low-variance features.

