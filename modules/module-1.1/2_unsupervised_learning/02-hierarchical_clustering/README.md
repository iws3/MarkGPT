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