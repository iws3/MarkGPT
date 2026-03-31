# 🔍 Lesson 12 — Feature Selection

> **Core Idea**: Not all features are created equal. Some help the model learn, some are irrelevant noise, and some actively confuse the model. Feature selection is the art of identifying and keeping only the features that genuinely improve prediction.

---

## 📋 Table of Contents

1. [Why Feature Selection Matters](#1-why-it-matters)
2. [Filter Methods — Statistical Tests](#2-filter-methods)
3. [Wrapper Methods — Search Over Feature Subsets](#3-wrapper-methods)
4. [Embedded Methods — Learning and Selecting Together](#4-embedded-methods)
5. [Dimensionality Reduction vs Feature Selection](#5-dim-reduction)
6. [Python Implementation](#6-python-implementation)
7. [Visual Summary](#7-visual-summary)

---

## 1. Why Feature Selection Matters

More features is not always better. Adding irrelevant features harms models for several reasons. First, noise features introduce random variation that the model may accidentally overfit to. Second, many algorithms (especially KNN and SVM) are hurt by the curse of dimensionality — distance metrics become meaningless in high dimensions. Third, more features means more parameters to estimate, which requires more training data to get reliable estimates. Finally, irrelevant features slow down training and make models harder to interpret.

```
Dataset with 100 features, only 10 truly predictive:

Without feature selection:
  Model accuracy: 82%  (noise features confuse the model)
  Training time:  45 seconds
  Model parameters: 100 weights

With feature selection (keep top 10):
  Model accuracy: 89%  (cleaner signal)
  Training time:  5 seconds
  Model parameters: 10 weights
  Bonus: you can now visualise relationships in 2D/3D
```

Feature selection methods fall into three broad categories: filter, wrapper, and embedded methods.

---

## 2. Filter Methods — Statistical Tests

Filter methods evaluate each feature independently, using a statistical score to measure its relevance to the target variable. They are fast (no model training needed) but ignore interactions between features.

```
For Classification targets:

  Chi-square test: for categorical/count features vs categorical target.
    Tests whether feature values are independent of the class label.
    High chi-square = feature and class are NOT independent = useful feature.

  ANOVA F-test: for numerical features vs categorical target.
    Tests whether the mean of the feature differs significantly across classes.
    High F-statistic = feature means differ across classes = useful feature.

  Mutual Information: for any feature type vs any target.
    Measures how much knowing the feature reduces uncertainty about the target.
    MI = 0 → feature and target are independent (useless feature)
    MI > 0 → feature contains information about the target (useful!)

For Regression targets:
  Pearson correlation: measures linear relationship (r close to ±1 = useful)
  Spearman correlation: non-linear monotonic relationships
  Mutual Information: most powerful, detects any statistical dependence
```

Filter methods work best as a first pass: eliminate obviously irrelevant features before applying more expensive methods.

---

## 3. Wrapper Methods — Search Over Feature Subsets

Wrapper methods actually train a model on different feature subsets and pick the subset that maximises model performance. They are more accurate than filter methods but computationally expensive.

```
Forward Selection:
  Start with no features.
  1. Try adding each feature one at a time. Keep the one that helps most.
  2. Try adding each remaining feature. Keep the best addition.
  3. Repeat until adding any feature no longer improves performance.
  Result: features added in order of importance (greedy, not optimal).

Backward Elimination:
  Start with ALL features.
  1. Remove the least useful feature (smallest accuracy drop when removed).
  2. Repeat until removing any more features hurts performance too much.

Recursive Feature Elimination (RFE):
  Train a model. Rank features by importance (e.g., coefficient magnitude).
  Remove the least important feature. Retrain. Repeat.
  sklearn.feature_selection.RFE does this automatically.
  RFECV adds cross-validation to find the optimal number of features to keep.
```

---

## 4. Embedded Methods — Learning and Selecting Together

Embedded methods perform feature selection as part of the model training process. They are the best of both worlds: accurate and computationally efficient.

```
Lasso Regression:   Regularisation shrinks irrelevant feature weights to exactly 0.
                    Features with θⱼ = 0 are effectively selected out.
                    The regularisation parameter λ controls how many features survive.

Tree-based models:  Features used high in the tree (at the root) are most important.
                    feature_importances_ gives a score for each feature.
                    SelectFromModel keeps only the features above a threshold.

Regularised neural networks: L1 regularisation on input layer weights.
                              Some input connections go to zero → implicit feature selection.
```

---

## 5. Dimensionality Reduction vs Feature Selection

These are different concepts that are often confused. Feature selection picks a **subset of original features** (the others are discarded). Dimensionality reduction **creates new features** that are combinations of the originals (the original features are transformed, not selected):

```
Feature Selection:              Dimensionality Reduction (e.g., PCA):
  Original: [age, height,         Original: [age, height, weight, bmi]
             weight, bmi]          New 2D components:
  Selected: [age, weight]            PC1 = 0.6×age + 0.4×height + 0.5×weight + 0.5×bmi
                                     PC2 = 0.7×age − 0.3×height − ...

  ✅ Interpretable (original feats)  ✅ Captures maximum variance
  ✅ No information mixing           ✅ Handles collinearity
  ❌ May miss interaction feats      ❌ New features are hard to interpret
```

Use feature selection when interpretability matters. Use dimensionality reduction when you need maximum information compression and interpretability is not required.

---

## 6. Python Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import (SelectKBest, chi2, f_classif,
                                        mutual_info_classif, RFE, RFECV,
                                        SelectFromModel, VarianceThreshold)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# ─── 1. Filter: Variance Threshold ───────────────────────────────────────
# Remove features with near-zero variance (they carry almost no information)
var_selector = VarianceThreshold(threshold=0.01)
X_var = var_selector.fit_transform(X_s)
print(f"Variance filter: {X_s.shape[1]} → {X_var.shape[1]} features")

# ─── 2. Filter: SelectKBest with ANOVA F-test ─────────────────────────────
selector_anova = SelectKBest(score_func=f_classif, k=10)
X_anova = selector_anova.fit_transform(X_s, y)
scores = selector_anova.scores_
ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
print("\nTop 10 features by F-test score:")
for name, score in ranked[:10]:
    print(f"  {name:40s}: {score:.2f}")

# ─── 3. Filter: Mutual Information ────────────────────────────────────────
mi_scores = mutual_info_classif(X_s, y, random_state=42)
mi_ranked = sorted(zip(feature_names, mi_scores), key=lambda x: x[1], reverse=True)
print("\nTop 10 features by Mutual Information:")
for name, score in mi_ranked[:10]:
    print(f"  {name:40s}: {score:.4f}")

# ─── 4. Wrapper: Recursive Feature Elimination with CV ────────────────────
rfe_cv = RFECV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    step=1,       # remove 1 feature at a time
    cv=5,
    scoring='accuracy'
)
rfe_cv.fit(X_s, y)
print(f"\nRFECV optimal number of features: {rfe_cv.n_features_}")
selected_features = feature_names[rfe_cv.support_]
print(f"Selected features: {list(selected_features)}")

plt.plot(range(1, len(rfe_cv.cv_results_['mean_test_score']) + 1),
         rfe_cv.cv_results_['mean_test_score'])
plt.xlabel('Number of features kept')
plt.ylabel('5-fold CV accuracy')
plt.title('RFECV — Optimal Feature Count')
plt.axvline(rfe_cv.n_features_, color='red', linestyle='--', label=f'Best: {rfe_cv.n_features_}')
plt.legend()
plt.show()

# ─── 5. Embedded: SelectFromModel with RandomForest ──────────────────────
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_s, y)
sfm = SelectFromModel(rf, threshold='mean')  # keep features with > mean importance
sfm.fit(X_s, y)
X_rf_selected = sfm.transform(X_s)
print(f"\nRF SelectFromModel: {X_s.shape[1]} → {X_rf_selected.shape[1]} features")

# ─── 6. Compare models with different feature sets ──────────────────────
for name, X_subset in [('All 30 features',        X_s),
                         ('Top 10 (F-test)',        X_anova),
                         ('RFECV optimal',          rfe_cv.transform(X_s)),
                         ('RF selected',            X_rf_selected)]:
    score = cross_val_score(LogisticRegression(max_iter=1000), X_subset, y, cv=5).mean()
    print(f"{name:25s}: 5-fold CV accuracy = {score:.4f}")
```

---

## 7. Visual Summary

```
╔══════════════════════════════════════════════════════════════════╗
║               FEATURE SELECTION — OVERVIEW                      ║
╠══════════════════════════════════════════════════════════════════╣
║  METHOD         HOW IT WORKS             WHEN TO USE            ║
║  ──────────     ──────────────────────   ──────────────────────  ║
║  Filter         Statistical scores        First-pass screening   ║
║  (F-test, MI)   per feature, fast         before model training  ║
║                                                                  ║
║  Wrapper        Train model on subsets    When you can afford    ║
║  (RFE, RFE-CV)  of features, slower       more computation       ║
║                                                                  ║
║  Embedded       Selection during          Best default choice;   ║
║  (Lasso, RF)    model training            fast and effective     ║
╠══════════════════════════════════════════════════════════════════╣
║  GOLDEN RULE: Always evaluate on a held-out set. Feature        ║
║  selection is a form of model training — it can overfit!        ║
║  Use cross-validation throughout the selection process.         ║
╚══════════════════════════════════════════════════════════════════╝
```
