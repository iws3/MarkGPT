# 🏠 Lesson 07 — K-Nearest Neighbours (KNN)

> **Core Idea**: To classify a new point, look at the K most similar training examples and take a majority vote. To predict a value, take the average of the K nearest neighbours. No model is trained — you simply remember all training data and consult it at prediction time.

---

## 📋 Table of Contents

1. [The "Lazy Learning" Philosophy](#1-lazy-learning)
2. [How Similarity Is Measured — Distance Metrics](#2-distance-metrics)
3. [Choosing K — The Central Hyperparameter](#3-choosing-k)
4. [Weighted KNN](#4-weighted-knn)
5. [The Curse of Dimensionality](#5-curse-of-dimensionality)
6. [Python Implementation](#6-python-implementation)
7. [Visual Summary](#7-visual-summary)
8. [When to Use](#8-when-to-use)

---

## 1. The "Lazy Learning" Philosophy

Most ML algorithms train a model upfront (compute weights, build a tree, etc.) and then use that compact model for predictions. KNN does the opposite: there is no training phase at all. Instead, all training data is stored in memory, and every prediction requires searching through the entire training set to find the nearest neighbours.

This is called **lazy learning** — the algorithm defers all computation to prediction time.

```
Training phase:  Just store all (X, y) pairs in memory.
                 Time: O(1)  ← instant!

Prediction for a new point x_new:
  1. Compute distance(x_new, xᵢ) for ALL training points xᵢ
  2. Sort by distance
  3. Take the K closest points
  4. Classification: majority vote among their labels
  5. Regression: average of their target values
  Time: O(n × d)  ← slow for large n (samples) or d (features)

n = 10 million training examples, d = 100 features:
  Each prediction = 10 million × 100 multiplications → very slow!
  (Data structures like KD-trees and ball trees speed this up significantly)
```

---

## 2. Distance Metrics

"Similar" means "close" in feature space. The most common distance measure is Euclidean distance — straight-line distance — but there are others:

```
Euclidean distance (default, works for continuous features):
  d(a, b) = √[ Σ (aᵢ - bᵢ)² ]
  = straight-line distance in n-dimensional space

Manhattan distance (L1, robust to outliers):
  d(a, b) = Σ |aᵢ - bᵢ|
  = "city block" distance — sum of absolute differences
  Imagine walking on a grid of city streets

Minkowski distance (generalisation):
  d(a, b) = (Σ |aᵢ - bᵢ|^p)^(1/p)
  p=1 → Manhattan,  p=2 → Euclidean,  p→∞ → Chebyshev (max absolute difference)

Hamming distance (for categorical/binary features):
  d(a, b) = proportion of positions where aᵢ ≠ bᵢ

⚠️ Critical: ALWAYS scale features before using KNN!
   Without scaling, a feature with range 0–10,000 (like salary)
   dominates a feature with range 0–1 (like probability score).
   The large-range feature completely controls the distance computation.
```

---

## 3. Choosing K — The Central Hyperparameter

K is the most important hyperparameter. It controls the bias-variance tradeoff directly:

```
K=1 (nearest neighbour only):
  Every training point becomes its own region.
  Perfect training accuracy (0 error) → overfit!
  Very sensitive to noise and outliers.
  HIGH VARIANCE.

K=N (all training points vote):
  Every query gets the same answer: the majority class.
  This is the simplest possible model — just predict the most common class.
  HIGH BIAS. (Unless the dataset is nicely balanced.)

K somewhere in the middle:
  Balance between memorising noise (small K) and ignoring local structure (large K).

Visualisation of decision boundaries:
  K=1: ██░░░░░██░   Jagged, complex boundaries
  K=5: ███░░░░███   Smoother boundaries
  K=15:████░░░████  Very smooth, almost linear boundaries

Rule of thumb: Start with K = √n (square root of training examples).
Always use odd K for binary classification to avoid ties.
Find optimal K using cross-validation.
```

---

## 4. Weighted KNN

A natural improvement: give closer neighbours more influence. Instead of each neighbour getting one equal vote, weight each vote by the inverse of its distance:

```
Uniform weights (standard KNN):  Each of K neighbours contributes equally.
Distance weights:                  Closer neighbours contribute more.
  weight(xᵢ) = 1 / distance(x_new, xᵢ)
  → A neighbour at distance 0.1 has 10× more influence than one at distance 1.0.

sklearn parameter: KNeighborsClassifier(weights='distance')
```

Weighted KNN is almost always better than uniform KNN, especially near class boundaries.

---

## 5. The Curse of Dimensionality

This is the most important theoretical limitation of KNN, and of many distance-based algorithms. As the number of features (dimensions) grows, the concept of "nearest neighbour" breaks down:

```
In 2D: K nearest neighbours are geometrically close → they're truly similar
In 100D: Even the "nearest" neighbour may be quite far away
In 1000D: All points become nearly equidistant from each other!

Proof by intuition:
  In 1D with 1,000 points uniformly distributed in [0,1]:
    Expected distance to nearest neighbour ≈ 0.001  (very close!)

  In 100D with 1,000 points:
    Expected distance to nearest neighbour ≈ 0.52  (over HALF the range!)

  → You'd need 2^100 ≈ 10^30 training points to maintain the same density.
    Completely infeasible.

Practical implication: KNN degrades badly as feature count grows.
Fix: Apply PCA or feature selection to reduce to < 20 dimensions before KNN.
```

---

## 6. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ─── Classification example ─────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

# ALWAYS scale for KNN!
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─── Find optimal K ──────────────────────────────────────────────────────
k_values = range(1, 31)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    scores = cross_val_score(knn, X_train_s, y_train, cv=5)
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal K: {optimal_k}")

plt.plot(k_values, cv_scores, 'b-o', markersize=4)
plt.axvline(optimal_k, color='red', linestyle='--', label=f'Best K={optimal_k}')
plt.xlabel('K (number of neighbours)')
plt.ylabel('5-fold CV Accuracy')
plt.title('K vs Cross-Validation Accuracy')
plt.legend()
plt.show()

# ─── Final model ─────────────────────────────────────────────────────────
final_knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
final_knn.fit(X_train_s, y_train)
y_pred = final_knn.predict(X_test_s)
print(f"\nTest accuracy: {final_knn.score(X_test_s, y_test):.4f}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ─── Regression example ──────────────────────────────────────────────────
diabetes = load_diabetes()
X_r, y_r = diabetes.data, diabetes.target
X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
    X_r, y_r, test_size=0.2, random_state=42)
scaler_r = StandardScaler()
X_r_train_s = scaler_r.fit_transform(X_r_train)
X_r_test_s  = scaler_r.transform(X_r_test)

from sklearn.metrics import mean_squared_error
for k in [1, 5, 10, 20]:
    knnr = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knnr.fit(X_r_train_s, y_r_train)
    rmse = np.sqrt(mean_squared_error(y_r_test, knnr.predict(X_r_test_s)))
    print(f"K={k:2d}: RMSE = {rmse:.2f}")
```

---

## 7. Visual Summary

```
╔══════════════════════════════════════════════════════════════════╗
║                     KNN — OVERVIEW                              ║
╠══════════════════════════════════════════════════════════════════╣
║  TRAINING:   Store all (X, y) pairs. Nothing else. Instant!     ║
║  PREDICTION: Find K closest training points → vote/average.     ║
║                                                                  ║
║  KEY HYPERPARAMETER: K                                           ║
║    Small K → complex, noisy boundary (high variance)            ║
║    Large K → smooth boundary (high bias)                        ║
║    Optimal K → find via cross-validation                        ║
║                                                                  ║
║  MUST DO: Scale features before training/predicting!            ║
║  WATCH FOR: Slow prediction on large datasets                   ║
║  AVOID: High-dimensional data (curse of dimensionality)         ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 8. When to Use

KNN is excellent as a non-parametric baseline when you don't know the shape of your data's decision boundary, for anomaly detection (an outlier will have no close neighbours), and for recommendation systems (find K users most similar to this user, recommend what they liked). Avoid it for large datasets (prediction is slow), high-dimensional data without prior dimensionality reduction, and real-time applications where latency matters.

> 📂 Next: [exercises.md](exercises.md)
