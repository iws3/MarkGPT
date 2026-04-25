# Day 09 Journal: Linear Algebra Deep Dive
**Date:** 2026-04-25
**Author:** Fonyuy-pounds

## Today's Learning
Today was about understanding the geometric intuition of Linear Algebra and how it applies to NLP. 

### Key Concepts
1. **Eigenvalues & Eigenvectors**: These are the "characteristic" directions of a transformation. In PCA, they help us find the directions of maximum variance.
2. **Principal Component Analysis (PCA)**: By projecting high-dimensional word co-occurrence data onto the top principal components, we can visualize semantic relationships.
3. **Word Co-occurrence**: Words that appear in similar contexts (within a small window) tend to have similar co-occurrence profiles.

## Hands-on Exercise
I implemented PCA from scratch using `numpy.linalg.eigh` and built a co-occurrence matrix from a combined corpus of KJV Bible verses and Banso proverbs.

### Observations
- Semantically related words like "God", "Spirit", and "Lord" often cluster together or align along similar axes.
- The visualization shows how the "geometry of language" starts to emerge even from simple statistical counts before we even touch neural networks.

## Reflections
It's fascinating how much "meaning" can be captured just by looking at which words sit next to each other. This is the bedrock of everything from Word2Vec to the latest Transformers.

## Output
![Word Clusters PCA](word_clusters_pca.png)
