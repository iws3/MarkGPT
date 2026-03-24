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

### Selecting k and Evaluation Metrics

Choosing the number of clusters k is a fundamental challenge in k-means; the problem provides no inherent ground truth regarding the appropriate number of clusters. The elbow method plots the within-cluster sum of squared distances (WCSS) against k and selects the k where the curve shows an elbow or significant change in slope. The silhouette score measures how similar each point is to its cluster compared to other clusters, ranging from -1 to 1 where higher is better. Gap statistics compare observed clustering quality to that of random data. Davies-Bouldin index measures the average similarity between each cluster and the most similar cluster, with lower values indicating better separation. These metrics provide heuristics but no definitive answer; domain knowledge frequently informs the choice of k. Cross-validation on downstream tasks (if clustering is a preprocessing step) or expert judgment often provides the best guidance.

### Scalability, Variants, and Limitations

K-means has linear time complexity O(nkd) per iteration for n points, k clusters, and d dimensions, making it scalable to large datasets. Mini-batch k-means processes data in batches, further improving scalability through reduced memory requirements and computations per iteration. K-means clusters are spherical and similar-sized; elongated clusters or clusters with different densities challenge the algorithm as it minimizes Euclidean distance without accounting for cluster shape or local density. Soft k-means (fuzzy c-means) assigns points probabilistically to clusters rather than hard assignments, providing a softer clustering that captures uncertainty. K-medoids selects actual data points as cluster centers instead of means, making it more robust to outliers. Despite limitations, k-means remains fundamental in unsupervised learning for its speed, simplicity, and often adequate performance; it frequently serves as a baseline or preprocessing step in more sophisticated clustering approaches.