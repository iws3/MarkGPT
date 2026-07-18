# 🌲🌲🌲 Lesson 04 — Random Forests

> **Core Idea**: Instead of growing one deep, overfit decision tree, grow hundreds of diverse trees on random subsets of data and features — then let them vote. The wisdom of the crowd beats any single expert.

---

## 📋 Table of Contents

1. [The Problem with a Single Tree](#1-the-problem)
2. [The Ensemble Principle — Wisdom of the Crowd](#2-ensemble-principle)
3. [Two Kinds of Randomness — Bootstrap + Feature Subsampling](#3-two-kinds-of-randomness)
4. [How Prediction Works — Majority Voting & Averaging](#4-prediction)
5. [Out-of-Bag Error — Free Validation](#5-oob-error)
6. [Feature Importance in Random Forests](#6-feature-importance)
7. [Key Hyperparameters](#7-hyperparameters)
8. [Strengths and Weaknesses](#8-strengths-and-weaknesses)
9. [Python Implementation](#9-python-implementation)
10. [Visual Summary](#10-visual-summary)

---

## 1. The Problem with a Single Tree

A single decision tree has a fundamental flaw: it is a **high-variance** model. This means that if you change your training data slightly — say, swap out 10% of the examples — your tree can look completely different. The model is too sensitive to the specific quirks of the training data it happened to see.

```
Training set A        Training set B
(slightly different)  (slightly different)

Tree from A:          Tree from B:
  [Feat3 > 0.5]         [Feat1 > 2.1]
   /       \             /       \
[Feat1>2] [Class0]  [Feat3>0.4] [Class1]
  /    \                ...
                    ← Completely different tree!

Both trees have ~85% accuracy, but they disagree on many examples.
```

If only there were a way to reduce this sensitivity without sacrificing accuracy...

---

## 2. The Ensemble Principle

The central insight is simple: **individual experts make errors, but different experts tend to make different errors**. If you aggregate many imperfect but diverse opinions, the errors cancel out and you're left with something much more reliable.

```
Single tree:   ████████████░░░░░░░░  80% accuracy
                             ^^^^^ these 20% wrong predictions are consistent

100 diverse trees voting:
Tree 1:  ████████████████░░░░  correct on examples A,B,C... wrong on X,Y
Tree 2:  █████████████████░░░  correct on examples A,B,D... wrong on Y,Z
Tree 3:  ████████████████████  correct on examples A,C,D... wrong on X,Z
...
Majority vote:
  Example A: all trees agree → CORRECT
  Example X: most trees disagree → average votes out the noise → often CORRECT

Random Forest: ██████████████████░░  90% accuracy  ← better than any single tree!
```

The key is that the trees must be **diverse** — if they all make the same mistakes, averaging doesn't help. This is why random forests introduce two types of randomness.

---

## 3. Two Kinds of Randomness

### Randomness #1 — Bootstrap Sampling (Bagging)

Each tree is trained on a **different bootstrap sample** of the training data: sample N examples *with replacement* from N training examples. On average, about 63% of examples appear in each bootstrap sample (some appear multiple times, some not at all).

```
Original training set (N=6):  [A, B, C, D, E, F]

Bootstrap sample for Tree 1:  [A, A, C, D, D, F]  ← A and D duplicated; B, E missing
Bootstrap sample for Tree 2:  [B, B, C, C, E, F]  ← B and C duplicated; A, D missing
Bootstrap sample for Tree 3:  [A, C, D, E, E, F]  ← E duplicated; B missing
...
```

Each tree sees a slightly different version of the dataset → they make different mistakes.

### Randomness #2 — Random Feature Subsets

At each split in each tree, instead of considering **all** features, only a random subset of `max_features` features is considered. This forces each tree to learn different patterns, further increasing diversity.

```
Standard decision tree split:  considers ALL p features → always picks the same best one
Random forest split:           considers only √p features → forced to use different ones

With p = 16 features, √p ≈ 4 — so each split only sees 4 randomly chosen features.
```

These two mechanisms together ensure that no two trees are identical, which is exactly what makes the ensemble powerful.

---

## 4. How Prediction Works

### Classification — Majority Voting

Each of the N trees predicts a class. The final prediction is whichever class receives the most votes.

```
100 trees predict for a new house: "expensive" or "cheap"
  "expensive" votes: 73
  "cheap"    votes: 27
  Final prediction: "expensive" (73% of trees agree)

The probability estimate is simply 73/100 = 0.73 — more calibrated than a single tree's 100% confidence!
```

### Regression — Averaging

Each tree predicts a number. The final prediction is the average across all trees.

```
100 trees predict house price:
  Tree 1: $285,000
  Tree 2: $310,000
  Tree 3: $275,000
  ...
  Average: $291,000  ← much more stable than any single tree's prediction
```

---

## 5. Out-of-Bag Error — Free Validation

Remember that each tree's bootstrap sample leaves out about 37% of examples. Those "left-out" examples are called **out-of-bag (OOB)** samples. The clever insight is that we can use them as a free validation set — each training example can be evaluated by all the trees that didn't see it during training!

```
Example D is missing from Trees 1, 3, 7, 12, ... (those that didn't bootstrap-sample it)
→ Use those trees to predict for D
→ Compare to D's true label
→ OOB error for D

Average OOB error across all training examples ≈ test set error
→ No separate validation set needed!
→ Useful when data is limited

In sklearn: RandomForestClassifier(oob_score=True)
            then access: model.oob_score_
```

---

## 6. Feature Importance in Random Forests

Random forests provide a highly reliable measure of feature importance — arguably more reliable than single tree importance or linear regression coefficients.

**Mean Decrease in Impurity (MDI)**: Average how much each feature decreases the Gini/entropy across all trees and all splits. Features used near the top of trees (where they have the most impact) contribute most.

**Permutation Importance** (more reliable): For each feature, shuffle its values randomly in the test set and measure how much accuracy drops. A large drop = important feature. A small drop = unimportant feature.

```python
# Permutation importance (use this over MDI for more reliable results)
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
# result.importances_mean gives the average accuracy drop for each feature
```

---

## 7. Key Hyperparameters

```
┌──────────────────────┬─────────────────────────────────────────────────────┐
│ Hyperparameter       │ Effect and Guidance                                 │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ n_estimators         │ Number of trees. More = better (but slower).        │
│  default: 100        │ Start at 100; increase until accuracy stops growing.│
│  try: 100, 300, 500  │ Rarely hurts to use more trees — only costs time.   │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ max_features         │ Features considered at each split.                  │
│  default: 'sqrt'     │ 'sqrt' for classification; 1/3 for regression.      │
│  try: 'sqrt','log2'  │ Smaller = more diverse trees, lower variance.       │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ max_depth            │ Max depth of each individual tree.                  │
│  default: None       │ Unlike single trees, RF is less sensitive to this   │
│  try: None, 10, 20   │ because averaging handles overfitting.              │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ min_samples_leaf     │ Minimum samples per leaf — controls smoothness.     │
│  default: 1          │ Increasing gives smoother probability estimates.     │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ n_jobs               │ Parallel training jobs. n_jobs=-1 uses all cores.   │
│  default: 1          │ Always set to -1 for faster training!               │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ oob_score            │ Whether to compute OOB score. Set True for free CV. │
└──────────────────────┴─────────────────────────────────────────────────────┘
```

---

## 8. Strengths and Weaknesses

```
✅ STRENGTHS                          ❌ WEAKNESSES
──────────────────────────────────    ──────────────────────────────────
Dramatically lower variance than      Slower to train than a single tree
  a single tree                       (especially with 500+ trees)
Robust to outliers and noise          Not as interpretable — you can't
Works well out-of-the-box              draw and explain 500 trees!
  (less hyperparameter tuning)        Memory-intensive (stores all trees)
Free OOB validation estimate          Slower at prediction (must query
Handles missing values (some              all trees)
  implementations)                    Still struggles with extrapolation
Provides reliable feature             Biased toward high-cardinality
  importance measures                   categorical features (MDI)
Parallelisable — scales well          For very structured tabular data,
  with multi-core CPUs                  gradient boosting often wins
```

---

## 9. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report

# ─── Data ─────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)
# Note: Random forests do NOT require feature scaling!
# They are invariant to monotonic feature transformations.

# ─── Train ────────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,    # 200 trees
    max_features='sqrt', # consider √(30) ≈ 5 features at each split
    oob_score=True,      # free out-of-bag validation
    n_jobs=-1,           # use all CPU cores for parallel training
    random_state=42
)
rf.fit(X_train, y_train)

print(f"OOB accuracy (free validation):    {rf.oob_score_:.4f}")
print(f"Test accuracy:                     {rf.score(X_test, y_test):.4f}")
print(f"5-fold CV accuracy:                {cross_val_score(rf, X, y, cv=5).mean():.4f}")

# ─── Feature importance ───────────────────────────────────────────────────
# Method 1: Mean Decrease in Impurity (fast but biased)
mdi_importance = rf.feature_importances_

# Method 2: Permutation Importance (slower but unbiased)
perm_result = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42)
perm_importance = perm_result.importances_mean

# Compare both methods
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, imp, title in zip(axes, [mdi_importance, perm_importance],
                           ['MDI Feature Importance', 'Permutation Importance']):
    sorted_idx = np.argsort(imp)
    ax.barh(np.array(data.feature_names)[sorted_idx], imp[sorted_idx])
    ax.set_title(title)
    ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.show()

# ─── n_estimators sweep — when do more trees stop helping? ────────────────
n_trees = [10, 25, 50, 100, 200, 500]
train_scores, oob_scores = [], []
for n in n_trees:
    rf_n = RandomForestClassifier(n_estimators=n, oob_score=True,
                                   n_jobs=-1, random_state=42)
    rf_n.fit(X_train, y_train)
    train_scores.append(rf_n.score(X_train, y_train))
    oob_scores.append(rf_n.oob_score_)

plt.plot(n_trees, train_scores, 'b-o', label='Train accuracy')
plt.plot(n_trees, oob_scores,   'r-o', label='OOB accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Effect of n_estimators on Accuracy')
plt.show()
```

---

## 10. Visual Summary

```
╔══════════════════════════════════════════════════════════════════╗
║                  RANDOM FORESTS — OVERVIEW                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Training data                                                   ║
║       │                                                          ║
║  ┌────┴────┐  Bootstrap   ┌────────┐ ┌────────┐ ┌────────┐     ║
║  │ Dataset │──sampling──► │ Tree 1 │ │ Tree 2 │ │Tree N  │     ║
║  └─────────┘  + random    └────────┘ └────────┘ └────────┘     ║
║               features       │           │          │           ║
║                               └─────────┬┘──────────┘           ║
║  New example ─────────────────────────► │                        ║
║                                    AGGREGATE                     ║
║                              (vote / average)                    ║
║                                         │                        ║
║                                  Final Prediction                ║
╠══════════════════════════════════════════════════════════════════╣
║  KEY INSIGHT: Randomness creates diversity. Diversity + averaging║
║  cancels out errors. Result: low variance, high accuracy.        ║
╚══════════════════════════════════════════════════════════════════╝
```
