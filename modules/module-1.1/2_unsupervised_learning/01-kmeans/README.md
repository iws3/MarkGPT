# K-Means Clustering

## Fundamentals

K-Means is a simple yet powerful unsupervised learning algorithm that partitions data into K clusters by minimizing within-cluster variance. The algorithm is widely used for customer segmentation, image compression, and anomaly detection. K-Means operates on a simple principle: iteratively assign points to nearest cluster centers and update centers. Despite its simplicity, K-Means is computationally efficient and scales well to large datasets. Understanding K-Means provides a foundation for more sophisticated clustering algorithms and is essential for practitioners working with unlabeled data.

## Key Concepts

- **Cluster Centers**: Centroids of clusters
- **Inertia**: Within-cluster sum of squares
- **Elbow Method**: Determining optimal K
- **Initialization**: K-means++, multiple restarts
- **Convergence**: Iterative optimization

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Clustering Objective and Algorithm Overview

K-means is an unsupervised learning algorithm that partitions data into k clusters by minimizing the within-cluster sum of squared distances. The algorithm begins by initializing k cluster centers (centroids) randomly or using smarter initialization strategies like k-means++. Each iteration comprises two steps: first, each data point is assigned to the nearest centroid (assignment step), and second, centroids are updated to the mean of all points assigned to each cluster (update step). This process repeats until convergence, when centroid positions no longer change significantly or a maximum iteration limit is reached. The algorithm optimizes the objective function J = Σ Σ ||x - μ_k||², where x are data points, μ_k are centroids, and the minimization is over all clusters. Despite its simplicity, k-means is widely used due to computational efficiency and reasonable performance in many applications.

### Clustering Objective and Algorithm Overview

K-means is an unsupervised learning algorithm that partitions data into k clusters by minimizing the within-cluster sum of squared distances. The algorithm begins by initializing k cluster centers (centroids) randomly or using smarter initialization strategies like k-means++. Each iteration comprises two steps: first, each data point is assigned to the nearest centroid (assignment step), and second, centroids are updated to the mean of all points assigned to each cluster (update step). This process repeats until convergence, when centroid positions no longer change significantly or a maximum iteration limit is reached. The algorithm optimizes the objective function J = Σ Σ ||x - μ_k||², where x are data points, μ_k are centroids, and the minimization is over all clusters. Despite its simplicity, k-means is widely used due to computational efficiency and reasonable performance in many applications.

### Initialization Strategies and Local Optima

A critical issue with k-means is its sensitivity to initialization; different random starting centroid positions can lead to different final clusters, some of which may be suboptimal. Random initialization often leads to poor local optima, where clusters are not well-separated. The k-means++ initialization algorithm addresses this by selecting the first centroid randomly, then iteratively choosing subsequent centroids with probability proportional to their squared distance from the nearest existing centroid. This biases initialization toward distant points, producing more separated starting centroids and better final solutions. Multiple random restarts followed by selecting the best result (lowest final cost) is another practical approach. Understanding that k-means is sensitive to initialization is crucial for practitioners; running the algorithm multiple times with different initializations and selecting the best result significantly improves solution quality without increasing algorithmic complexity.