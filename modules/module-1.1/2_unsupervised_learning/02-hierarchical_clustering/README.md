# Hierarchical Clustering

## Fundamentals

Hierarchical clustering builds a dendrogram showing nested cluster structure at multiple levels of granularity. Unlike K-Means, hierarchical clustering doesn't require specifying K in advance. The dendrogram provides rich information about data structure and relationships between clusters. Hierarchical clustering is deterministic (unlike K-Means) and works well for exploratory analysis. Time complexity is higher than K-Means, limiting applicability to large datasets.

## Key Concepts

- **Dendrogram**: Hierarchical tree structure
- **Linkage Methods**: Single, complete, average, Ward
- **Distance Cutoff**: Determining final clusters
- **Agglomerative vs Divisive**: Bottom-up vs top-down

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Agglomerative and Divisive Hierarchical Clustering

Hierarchical clustering builds a tree-like structure (dendrogram) of nested clusters, allowing clusters at multiple granularities. Agglomerative clustering (bottom-up) starts with each point as its own cluster and recursively merges the two closest clusters until all points belong to one cluster. Divisive clustering (top-down) starts with all points in one cluster and recursively splits until each point is its own cluster. The agglomerative approach is more commonly used and computationally practical. The choice of linkage criterion determines how distances between clusters are computed. Single linkage uses the minimum distance between points from different clusters, tending to create long, chain-like clusters. Complete linkage uses the maximum distance, creating more compact clusters. Average linkage computes the average distance between all point pairs from different clusters, balancing extreme cases. Ward's method minimizes the increase in within-cluster variance, often producing well-separated, balanced clusters.

### Dendrogram Interpretation and Cluster Selection

The dendrogram visualization displays the hierarchical structure as a tree where leaves are individual points and internal nodes represent cluster merges. The height of each merge indicates the distance between merged clusters; larger heights represent larger distances and more disparate clusters. The dendrogram reveals cluster structure at different scales, showing which points are similar and how clusters naturally nest within each other. Selecting clusters from a dendrogram can be done by cutting the tree at a specific height; above the cut, cluster assignments are determined. This flexible approach allows adjustment of cluster granularity without re-running the algorithm. The choice of cutting height involves similar considerations as selecting k in k-means: elbow methods, domain knowledge, and downstream task performance provide guidance. Dendrograms also reveal interesting structure such as obvious cluster outliers or subclusters within larger clusters.

### Computational Complexity and Practical Considerations

Agglomerative hierarchical clustering has O(n² log n) time complexity and O(n²) space complexity, making it less scalable than k-means for large datasets. The distance matrix of size n×n must be stored and updated iteratively as clusters merge. For very large datasets, sub-sampling or approximate methods become necessary. However, hierarchical clustering provides unique advantages: the dendrogram offers rich structural information beyond simple cluster assignments, the method is deterministic (no random initialization), and no specification of cluster number is required a priori. Hierarchical clustering performs well when the true cluster structure has a natural hierarchical organization. It is particularly useful in exploratory data analysis, taxonomy construction, and customer segmentation where hierarchical organization provides insights.

### Hybrid Approaches and Integration with Other Methods

In practice, hierarchical clustering is often combined with other methods. Cutting a dendrogram at different heights and applying k-means to the resulting clusters (hierarchical-k-means) combines the strengths of both approaches. Consensus clustering applies multiple clustering algorithms and aggregates results, often using hierarchical clustering within the ensemble. Time-series data frequently benefits from hierarchical clustering combined with dynamic time warping distances rather than Euclidean distances. Document clustering uses hierarchical clustering with cosine similarity on TF-IDF vectors. Hierarchical clustering can serve as a preprocessing step to estimate k for k-means. Despite higher computational cost, the structural insights and flexibility of hierarchical clustering make it valuable in applications where interpretability and hierarchical organization are important.

### Linkage Criteria Comparison and Trade-offs

Linkage criteria define how cluster distances are computed. Single linkage (minimum distance between clusters) emphasizes the role of outliers: a single bridge point connects clusters. This creates long, chain-like clusters. Complete linkage (maximum distance) is conservative: clusters are merged only if all pairs are nearby. It produces compact clusters but may split natural structures. Average linkage computes average distance; it balances single and complete, often producing good results. Ward linkage minimizes within-cluster variance increase; historically popular and often produces well-separated clusters. Centroid linkage uses cluster centroid distances. Empirical comparison shows Ward and average often outperform; try multiple methods via cross-validation. The choice depends on desired cluster shapes: elongated clusters suit single linkage; spherical clusters suit complete/Ward. Domain knowledge and data visualization guide selection. Different linkages on the same data produce different dendrograms; dendrogram shape reveals algorithm preferences.

### Dendrogram Interpretation and Cutting Strategies

Dendrograms visualize nested cluster structures; height represents merging distance. Tall vertical lines indicate large jumps in similarity (natural cluster boundaries). The dendrogram visually reveals structure: how clusters subdivide, outliers (long branches from leaves), and meaningful scales. Cutting the dendrogram at different heights produces different numbers of clusters: cutting high (few clusters) produces coarse grouping; cutting low (many clusters) produces fine-grained grouping. The optimal cut height is subjective; visual inspection reveals natural gaps. Automated cutting: choose height maximizing gap between consecutive merges, or fixing number of clusters desired. Dendrograms are invaluable for exploratory analysis: showing which samples cluster together, revealing hierarchical organization, and identifying outliers. Visualization libraries (scipy, seaborn) enable interactive dendrograms: hover to highlight clusters, zoom for details. This interactivity aids exploration and communication with stakeholders unfamiliar with clustering.

### Computational Efficiency and Approximations

Hierarchical clustering's O(n² log n) or O(n²) complexity is prohibitive for large datasets. Approximations reduce cost: BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) builds a hierarchical tree incrementally, achieving O(n) time. CURE (Clustering Using Representatives) selects representative points from each cluster, reducing distance computations. For n = 1M, hierarchical clustering is impractical; alternatives are necessary. Another approach: subsample data (cluster 10% efficiently, then assign remaining points based on subsample clusters). This sacrifices some precision for speed. In practice, hierarchical clustering is used for datasets < 10k samples; larger datasets require alternatives. However, for exploratory analysis, hierarchical clustering on subsampled data often reveals structure.