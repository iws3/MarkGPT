# K-Nearest Neighbors (KNN)

## Fundamentals

K-Nearest Neighbors (KNN) is a simple yet effective instance-based learning algorithm that makes predictions based on the proximity of training samples to the query point. KNN is a lazy learning algorithm that stores training data and makes predictions at query time, making it inherently flexible for both classification and regression. Despite its simplicity, KNN can capture complex non-linear patterns and handles multi-class problems naturally. However, it requires careful distance metric selection and feature scaling, and its computational cost grows with dataset size. KNN is commonly used as a baseline algorithm and in cases where interpretability through similar examples is valuable.

## Key Concepts

- **Distance Metrics**: Euclidean, Manhattan, Minkowski
- **Value of K**: Determines neighborhood size
- **Distance Weighting**: Weighted vs. uniform voting
- **Lazy Learning**: No training phase

## Applications

- Image recognition
- Recommendation systems
- Medical diagnosis
- Anomaly detection
- Document classification

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Distance Metrics Beyond Euclidean

While Euclidean distance is most common, other metrics suit different data types. Manhattan distance (L1) |x-y| sum uses coordinate differences; it's often more robust to outliers. Chebyshev distance (L-infinity) max(|x-y|) uses maximum coordinate difference; useful for bounded spaces. For high-dimensional data, Euclidean distances become less meaningful (curse of dimensionality); Manhattan or Chebyshev sometimes perform better. For text, cosine similarity (angle between vectors) is standard; it ignores magnitude, focusing on direction. For categorical data, Hamming distance (number of differing coordinates) applies. Domain-specific distances exist: edit distance (Levenshtein) for strings, dynamic time warping for time series, geodesic distance for data on manifolds. Choosing metric requires domain knowledge: geometric intuition in Euclidean space doesn't apply in all domains. k-NN performance is sensitive to metric choice; trying multiple metrics via cross-validation is worthwhile. Custom distances can be plugged into scikit-learn's k-NN.

### KD-Trees and Ball-Trees for Efficiency

Brute-force k-NN searches through all training points; O(n) per query. For large n (1M+), this is slow. KD-trees (k-dimensional trees) and ball-trees recursively partition space hierarchically. Search via these structures is O(log n) on average, massively faster. However, trees' efficiency degrades in high dimensions (curse of dimensionality); for d > 20, brute-force sometimes outperforms tree search. Scikit-learn automatically selects: if n < 30 or d > 16, it uses brute-force; otherwise, KD-tree or ball-tree depending on metrics. For production systems requiring fast queries, building trees once and querying many times is efficient. Approximate nearest neighbor methods (locality-sensitive hashing, learned indices) further improve speed, trading accuracy for extreme speed. Understanding these data structures helps practitioners: for thousands of queries on large data, preprocessing via tree construction is worthwhile.

### Feature Scaling and Dimensionality

k-NN's reliance on distances makes feature scaling critical. A feature with range [0, 1000] overshadows features with range [0, 1]; StandardScaler handles this. Without scaling, k-NN is biased toward high-variance features. Dimensionality reduction (PCA) before k-NN sometimes improves performance: noise in high dimensions is reduced, and computational cost decreases. However, be careful: dimensionality reduction loses information; if information is relevant to the task, performance drops. Feature selection (removing irrelevant features) is often better than reduction. Irrelevant features add noise; selecting relevant features improves k-NN significantly. In text classification with thousands of words, filtering to top 100 features via mutual information often helps. Domain knowledge guides selection: in healthcare, medical features matter; in text, stop words (the, a, is) don't. The number of features should be much smaller than the number of samples; d << n avoids the curse of dimensionality.

### Local Decision Boundaries and Confidence

k-NN creates local decision boundaries around each query point. Unlike global models, k-NN decisions depend only on nearby neighbors, useful for capturing local patterns and discontinuities. However, boundaries are discontinuous: moving a query point slightly can completely change the k nearest neighbors, causing prediction changes. This is problematic for confidence: slight perturbations cause different predictions. To quantify confidence, use soft predictions: proportion of neighbors in the positive class. If k=5 and 4 neighbors are positive, predicted probability is 0.8. Probabilities near 0.5 indicate low confidence (neighbors split). This enables ranking predictions by confidence. For critical applications (medical diagnosis), using only high-confidence predictions improves reliability. In risk-sensitive settings, abstaining on low-confidence predictions is better than guessing.

### k-NN Limitations and When to Use

k-NN's main limitation is computational cost: O(n) per prediction is prohibitive for real-time large-scale applications. It doesn't learn feature importance; all features contribute equally after scaling. It's prone to overfitting with small k (tight fit to training data); large k smooths predictions but loses detail. Memory requirements are huge (storing all training data); not feasible for billions of samples. It extrapolates poorly: predictions outside training data range are unprincipled. It assumes all neighbors are equally reliable; outliers influence predictions just like normal points. Despite limitations, k-NN excels in exploratory analysis: examining nearest neighbors reveals structure. It's useful for semi-supervised learning: using unlabeled neighbors. When to use: (1) exploratory analysis; (2) when instances themselves are important; (3) irregular decision boundaries; (4) datasets < 100k samples. When not: (1) real-time systems; (2) high-dimensional data (d > 50); (3) memory-constrained systems; (4) need for interpretability. k-NN's simplicity and flexibility make it valuable despite computational limitations.