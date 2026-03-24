# DBSCAN Clustering

## Fundamentals

DBSCAN (Density-Based Spatial Clustering) groups together points that are closely packed, marking points in sparse regions as outliers. DBSCAN is powerful for finding clusters of arbitrary shape and automatically detecting outliers. It doesn't require specifying K and is robust to noise. DBSCAN is widely used for spatial data, anomaly detection, and exploratory analysis where cluster shapes are unknown.

## Key Concepts

- **Density**: Points within epsilon distance
- **Core Points**: Sufficient neighboring points
- **Border Points**: Close to core points
- **Outliers**: Isolated low-density points
- **Epsilon and MinPts**: Critical parameters

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Density-Based Clustering and Eps-Neighbors

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) groups points that are densely packed together and identifies points in sparse regions as outliers. Unlike k-means or hierarchical clustering, DBSCAN does not require specifying the number of clusters; instead, it uses two parameters: eps (neighbor distance threshold) and min_pts (minimum points in eps-neighborhood). A point p is a core point if its eps-neighborhood contains at least min_pts points. Points within the eps-neighborhood of a core point are density-reachable from that core point. A cluster is formed by all density-connected points, where points are transitively connected through core points. Points not in any cluster are classified as noise or outliers. This density-based definition allows clusters of arbitrary shape and size, making DBSCAN more flexible than k-means or hierarchical clustering.

### Parameter Selection and Computational Aspects

Selecting eps and min_pts is crucial and often requires domain knowledge or data characteristics. The k-distance graph plots sorted distances to the k-th nearest neighbor for each point; a kink or elbow in this graph suggests an appropriate eps value corresponding to cluster density. min_pts is sometimes set to dimensionality + 1 or 2*dimensionality. Smaller eps values lead to more outliers and fragmented clusters; larger values merge distinct clusters. DBSCAN has O(n²) complexity in the worst case but O(n log n) with spatial indexing structures like KD-trees or R-trees. This makes DBSCAN practical for datasets where k-means' need for many iterations becomes expensive. DBSCAN is particularly suitable for spatial point clustering, geographic data, and applications where outlier detection is important. The noise classification is valuable when true outliers require special handling rather than forced assignment to clusters.