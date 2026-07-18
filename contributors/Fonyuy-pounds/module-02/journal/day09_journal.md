# Day 09: Linear Algebra Deep Dive
**Date:** 2026-04-23

## 📝 Learning Objectives
- Understanding matrix operations and their role in transformations.
- Deep dive into Eigenvalues and Eigenvectors.
- Implementing Principal Component Analysis (PCA) from scratch.
- Visualizing word embeddings and co-occurrence vectors.

## 🏋️‍♂️ Exercises: PCA from Scratch
**1. PCA Implementation Steps:**
*What are the five key steps in the PCA algorithm?*
1. Center the data by subtracting the mean.
2. Compute the covariance matrix of the centered data.
3. Perform eigen-decomposition on the covariance matrix to get eigenvalues and eigenvectors.
4. Sort eigenvalues in descending order and select the top $k$ eigenvectors as principal components.
5. Project the original centered data onto the selected principal components via matrix multiplication.

**2. Covariance and Eigen-decomposition:**
*What does the covariance matrix tell us about our features? Why do we use eigenvectors of this matrix for projection?*
> The covariance matrix quantifies how much pairs of features vary together. Diagonal elements are individual variances, and off-diagonals are covariances. Eigenvectors of this matrix represent the "principal directions" of variance in the data, with the corresponding eigenvalues indicating the magnitude of variance in those directions. By projecting onto the top eigenvectors, we preserve the most information (variance) possible in fewer dimensions.

## 🧠 Daily Reflection
**1. How does dimensionality reduction (like PCA) help in visualizing high-dimensional things like word embeddings?**
> Humans can only perceive data in 2D or 3D. Since word embeddings often have hundreds or thousands of dimensions, PCA (or t-SNE/UMAP) allows us to "squash" that space into 2D while keeping semantically similar words (which have similar vector directions) close to each other, making the relationships intuitive to inspect visually.

**2. In an LLM, dot products are used everywhere (especially in Attention). Conceptually, what does a dot product between two word vectors represent?**
> A dot product measures the alignment or similarity between two vectors. In the context of word vectors, a high dot product means the words often appear in similar contexts or share semantic features. In Attention, it specifically measures how much "focus" or "importance" one token (the Query) should assign to another (the Key).
