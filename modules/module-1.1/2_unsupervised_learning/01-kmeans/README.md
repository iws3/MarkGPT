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

### K-Means Challenges: Convergence and Initialization Sensitivity

K-means has two critical challenges: convergence guarantees and initialization sensitivity. The algorithm is guaranteed to converge, but only to a local optimum. Different initializations produce different final clusters; some are clearly suboptimal. The k-means++ initialization addresses this probabilistically: instead of random initialization, it selects first centroid randomly, then subsequent centroids with probability proportional to squared distance from nearest existing centroid. This biases toward well-separated starting points, typically producing better solutions. Mathematically, k-means++ gives O(log k) competitive ratio (expected cost ≤ O(log k) × optimal cost), a strong theoretical guarantee. Running k-means++ multiple times with different random seeds and selecting best result is practical. Seeding strategies matter: proper initialization can reduce iterations needed by 50% and dramatically improve final cluster quality. For reproducibility, set random seed explicitly.

### Choosing k: Elbow Method and Silhouette Analysis

Selecting k (number of clusters) is fundamental but challenging; no ground truth exists. The elbow method plots within-cluster sum of squares (WCSS) against k; WCSS decreases monotonically. An elbow (significant slope change) suggests natural k. However, elbows are subjective; many datasets show gradual decreases without clear elbows. Silhouette score measures cluster quality: each sample's silhouette is (b-a)/max(a,b), where a=mean intra-cluster distance, b=mean inter-cluster distance. Silhouette ranges [-1,1]; 1=perfect clustering, 0=overlapping clusters, negative=misclassified. Average silhouette across samples guides k selection: higher average suggests better k. Silhouette is more objective than elbow. Davies-Bouldin index (average similarity between each cluster and most similar cluster) and Calinski-Harabasz index also guide k selection. Cross-validation on downstream tasks (if clustering is preprocessing) might be most principled: select k optimizing final task performance.

### K-Means for Large Datasets: Mini-Batch K-Means

Standard k-means loads all data in memory; for datasets > 1GB, this becomes impractical. Mini-batch k-means processes data in small batches, updating clusters incrementally. Each iteration: (1) Draw random batch; (2) Assign samples to nearest centroids; (3) Update centroid estimates. Memory is O(k*d + batch_size*d), manageable even for huge datasets. Processing is online: data can be streamed. Speed improves significantly: for n samples, k clusters, d dimensions, mini-batch is O(n*k*d*iter) vs standard O(n*k*d*iter) with potentially far fewer iterations. Mini-batch sacrifices some solution quality (slightly suboptimal clusters) for speed and memory efficiency. For billion-sample datasets, mini-batch k-means is necessary. Scikit-learn implements MiniBatchKMeans; batch size is tunable (larger = more accurate but slower).

### Applications: Customer Segmentation and Anomaly Detection

K-means is widely used for customer segmentation: each customer is a point in feature space (purchase history, demographics, behavior); clusters group similar customers. Marketing teams tailor strategies per cluster. Behavioral targeting, personalized recommendations, and pricing strategies differ per cluster. Feature engineering is crucial: raw transaction counts are less informative than derived metrics (average purchase value, purchase frequency, recency). Credit scoring and risk assessment use k-means to segment loan applicants. K-means can detect anomalies: clusters represent normal behavior; points far from all clusters are anomalies. This is simpler than dedicated anomaly detection but less sophisticated. Image compression uses k-means: pixels are points in color space; clustering to k colors and replacing each pixel with cluster center reduces color palette, compressing images. Time series clustering (e.g., stock price patterns) uses k-means with appropriate distance metrics or kernel transformations.

### Limitations and When to Use Alternatives

K-means struggles with non-spherical clusters (elongated, crescent-shaped clusters are poorly handled) and varying cluster sizes. It's also sensitive to outliers: a single outlier can shift centroids significantly. For data with these properties, alternatives are better. DBSCAN handles arbitrary shapes and varying densities. Hierarchical clustering shows structure at multiple scales. Gaussian mixture models provide probabilistic framework with soft assignments. Feature space structure and outlier presence should guide algorithm selection. K-means assumes continuous numerical features; categorical data requires preprocessing (encoding, gower distance). In high dimensions (d > 50), k-means efficiency degrades; dimensionality reduction beforehand helps. K-means is appropriate when: clusters are roughly spherical, sizes similar, no trust in soft assignments, and speed matters. For exploratory analysis, also try hierarchical clustering or DBSCAN to understand structure sensitivity to algorithm choice.