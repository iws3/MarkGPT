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