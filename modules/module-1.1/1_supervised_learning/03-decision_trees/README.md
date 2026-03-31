# 🌳 Lesson 03 — Decision Trees

> **Core Idea**: A decision tree asks a series of yes/no questions about your data, branching left or right at each step, until it reaches a leaf node that gives a prediction. It is the most human-readable machine learning algorithm — you can literally draw it on a whiteboard and explain every decision.

---

## 📋 Table of Contents

1. [The Big Picture — What Problem Are We Solving?](#1-the-big-picture)
2. [Anatomy of a Decision Tree](#2-anatomy)
3. [How the Tree Learns — Splitting Criteria](#3-splitting-criteria)
4. [Information Gain and Entropy](#4-information-gain-and-entropy)
5. [Gini Impurity — The Alternative](#5-gini-impurity)
6. [Regression Trees](#6-regression-trees)
7. [Overfitting and Pruning](#7-overfitting-and-pruning)
8. [Key Hyperparameters](#8-hyperparameters)
9. [Strengths and Weaknesses](#9-strengths-and-weaknesses)
10. [Python Implementation](#10-python-implementation)
11. [Visual Summary](#11-visual-summary)

---

## 1. The Big Picture

Imagine you're a doctor trying to decide if a patient has a disease. You might think: *"Is their age over 50? If yes, do they have high blood pressure? If yes, is their cholesterol elevated? Then they are high risk."* You're doing decision tree reasoning — asking a series of questions that progressively narrow down the answer.

Decision trees formalise this intuition. The algorithm automatically learns *which* questions to ask and *in what order*, based purely on the training data.

```
                    [Is age > 50?]
                    /            \
                 YES              NO
                /                  \
    [Cholesterol > 200?]        [Predict: Low Risk]
         /         \
       YES           NO
      /               \
[High Risk]       [Medium Risk]
```

This entire structure is learned from data. No human hand-crafted the questions — the algorithm found them.

---

## 2. Anatomy of a Decision Tree

Every decision tree is made of three types of nodes, connected by branches:

```
                        ┌──────────────────┐
                        │   ROOT NODE      │  ← The first question (top of tree)
                        │  "Age > 50?"     │    Uses the most informative feature
                        └────────┬─────────┘
                                 │
               ┌─────────────────┴──────────────────┐
               │ YES                                 │ NO
               ▼                                     ▼
    ┌──────────────────┐                  ┌──────────────────┐
    │  INTERNAL NODE   │                  │   LEAF NODE      │
    │ "Cholesterol>200?"│                  │  "Low Risk"      │  ← A prediction!
    └────────┬─────────┘                  │  No more splits  │
             │                            └──────────────────┘
    ┌────────┴───────┐
    │ YES            │ NO
    ▼                ▼
┌──────────┐    ┌──────────┐
│  LEAF    │    │  LEAF    │
│"High Risk"│    │"Med Risk"│
└──────────┘    └──────────┘

ROOT NODE    = The very first split — the most important feature goes here
INTERNAL NODE = An intermediate split — further refines the prediction
LEAF NODE    = A final prediction — no more questions are asked
BRANCH       = A YES or NO answer to the question at the node above
DEPTH        = The number of edges from root to a leaf
```

---

## 3. How the Tree Learns — Splitting Criteria

The key question during training is: **at each node, which feature and which threshold should we split on?**

The algorithm evaluates every possible split and picks the one that makes the resulting groups *most pure* — meaning each group contains mostly examples of the same class, not a jumbled mix.

```
BEFORE SPLIT (very impure):
  Group: [🔴, 🔵, 🔴, 🔵, 🔴, 🔵]   ← 50% red, 50% blue — useless!

AFTER SPLIT on feature X > 3:
  Left group:  [🔴, 🔴, 🔴, 🔴]   ← almost all red  → pure! ✅
  Right group: [🔵, 🔵, 🔵]       ← all blue         → pure! ✅

A good split creates pure subgroups where we can make confident predictions.
```

The mathematical tools for measuring impurity are **Entropy** and **Gini Impurity**.

---

## 4. Information Gain and Entropy

Entropy is borrowed from information theory. It measures the "disorder" or "uncertainty" in a group:

```
Entropy H(S) = −Σ pᵢ × log₂(pᵢ)

Where pᵢ = proportion of class i in the group

Examples:
  Pure group [all class A]:   H = −(1.0 × log₂(1.0)) = 0 bits   (no uncertainty)
  Perfectly mixed [50/50]:    H = −(0.5 × log₂(0.5)) − (0.5 × log₂(0.5))
                                = 0.5 + 0.5 = 1.0 bit            (maximum uncertainty)
  Slightly skewed [75/25]:    H = −(0.75 × log₂(0.75)) − (0.25 × log₂(0.25))
                                ≈ 0.81 bits                        (some uncertainty)

Entropy ranges from 0 (perfectly pure) to log₂(K) (K = number of classes)
```

**Information Gain** measures how much a particular split *reduces* the entropy:

```
IG(S, feature) = H(S) − [ (|S_left|/|S|) × H(S_left) + (|S_right|/|S|) × H(S_right) ]
                          └─────────────────────────────────────────────────────────┘
                                        weighted average entropy after split

The algorithm picks the feature+threshold that MAXIMISES Information Gain.
```

---

## 5. Gini Impurity — The Alternative

Gini impurity is a slightly simpler alternative to entropy that often gives similar results but is faster to compute (no logarithm):

```
Gini(S) = 1 − Σ pᵢ²

Examples:
  Pure group [all class A]:  Gini = 1 − 1² = 0        (minimum — perfectly pure)
  Perfectly mixed [50/50]:   Gini = 1 − (0.5² + 0.5²) = 1 − 0.5 = 0.5  (maximum)
  75/25 split:               Gini = 1 − (0.75² + 0.25²) = 1 − 0.625 = 0.375

Gini ranges from 0 (pure) to 1−(1/K) for K classes (max impurity)

sklearn's DecisionTreeClassifier uses Gini by default.
Both Gini and Entropy usually produce very similar trees — the choice rarely matters much.
```

---

## 6. Regression Trees

Decision trees can also predict continuous values (numbers), not just classes. The only change is what goes in the leaf nodes and how we measure split quality:

```
Classification Tree leaf:  most common CLASS in that group   → "predict: spam"
Regression Tree leaf:      MEAN of target values in group    → "predict: $285,000"

Splitting criterion:       instead of entropy/gini, use
                           REDUCTION IN MSE (variance reduction)

IG(S, feature) = MSE(S) − [(|S_left|/|S|)×MSE(S_left) + (|S_right|/|S|)×MSE(S_right)]
```

---

## 7. Overfitting and Pruning

A decision tree grown without limits will perfectly memorise training data — creating a leaf node for every single training example. This achieves 100% training accuracy but terrible generalisation:

```
Fully grown tree (overfit):                 Pruned tree (generalises well):
  [Feature A > 0.123?]                        [Feature A > 0.5?]
    /          \                               /          \
[Feat B > 0.456?]  [Feat C > 0.789?]     [Class 1]    [Feature B > 0.3?]
  /    \           /     \                              /          \
[Feat D] [Feat E] [Leaf] [Feat F...]              [Class 0]    [Class 1]
  ...                     ...
One leaf per training example!          Simple, generalisable rules

Training accuracy: 100%                 Training accuracy: 90%
Test accuracy:     55%                  Test accuracy:     87%  ✅
```

The main hyperparameters for controlling overfitting are explained in the next section.

---

## 8. Key Hyperparameters

Understanding these gives you precise control over the bias-variance tradeoff:

```
┌──────────────────────┬─────────────────────────────────────────────────────┐
│ Hyperparameter       │ Effect                                              │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ max_depth            │ Maximum depth of the tree.                         │
│  default: None       │ None = grow until pure leaves (overfit risk).       │
│  try: 3, 5, 10       │ Start with 3 for interpretability. Tune via CV.     │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ min_samples_split    │ Minimum samples needed to split a node.             │
│  default: 2          │ Larger = simpler tree (fewer splits allowed).        │
│  try: 5, 10, 20      │ Increasing this prunes the tree effectively.         │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ min_samples_leaf     │ Minimum samples required in each leaf node.         │
│  default: 1          │ Larger = each leaf represents more examples.         │
│  try: 1, 5, 10       │ Prevents tiny, noisy leaves.                         │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ max_features         │ How many features to consider at each split.        │
│  default: None (all) │ sqrt(n) and log2(n) add randomness → used in RF.    │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ criterion            │ Split quality measure.                              │
│  'gini' or 'entropy' │ Usually similar results. Entropy slightly slower.   │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ ccp_alpha            │ Cost-complexity pruning parameter.                  │
│  default: 0.0        │ Larger = more pruning. Tune via cross-validation.   │
└──────────────────────┴─────────────────────────────────────────────────────┘
```

**Golden rule**: always use cross-validation (not a single train-test split) to tune these, as decision trees are highly sensitive to their hyperparameters and can vary a lot across different data splits.

---

## 9. Strengths and Weaknesses

```
✅ STRENGTHS                          ❌ WEAKNESSES
──────────────────────────────────    ──────────────────────────────────
Extremely interpretable               High variance — small data changes
Works with no feature scaling             create very different trees
Handles mixed feature types           Tends to overfit without pruning
Captures non-linear relationships     Struggles with diagonal decision
Fast to train and predict                 boundaries (e.g., XOR problem)
Built-in feature selection            Biased toward features with many
  (important features appear high)        possible split values
Can visualise with graphviz           Not great at extrapolation
```

The weakness of high variance is what motivates **Random Forests** (next lesson): by averaging many different trees trained on random subsets, we dramatically reduce variance while keeping the non-linear expressiveness.

---

## 10. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# ─── Load data ──────────────────────────────────────────────────────────────
data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── Train with default settings (will overfit!) ────────────────────────────
tree_default = DecisionTreeClassifier(random_state=42)
tree_default.fit(X_train, y_train)
print(f"Default tree depth: {tree_default.get_depth()}")
print(f"Default train acc:  {tree_default.score(X_train, y_train):.3f}")
print(f"Default test acc:   {tree_default.score(X_test, y_test):.3f}")

# ─── Tune max_depth via grid search ─────────────────────────────────────────
param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, None],
              'min_samples_leaf': [1, 2, 5, 10]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                           param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.3f}")

best_tree = grid_search.best_estimator_
print(f"Test accuracy (best): {best_tree.score(X_test, y_test):.3f}")

# ─── Visualise the tree ──────────────────────────────────────────────────────
plt.figure(figsize=(16, 8))
plot_tree(best_tree,
          feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True,          # colour nodes by class
          rounded=True,         # rounded boxes look nicer
          fontsize=10)
plt.title("Decision Tree — Iris Dataset (Best Depth)")
plt.show()

# ─── Text representation (printable, shareable) ──────────────────────────────
print(export_text(best_tree, feature_names=list(data.feature_names)))

# ─── Feature importance ──────────────────────────────────────────────────────
importances = best_tree.feature_importances_
feat_imp = sorted(zip(data.feature_names, importances), key=lambda x: x[1], reverse=True)
print("\nFeature Importances:")
for feat, imp in feat_imp:
    bar = '█' * int(imp * 40)
    print(f"  {feat:30s} {bar} ({imp:.3f})")
```

---

## 11. Visual Summary

```
╔══════════════════════════════════════════════════════════════════╗
║                 DECISION TREES — OVERVIEW                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  TRAINING:  At each node, find the feature + threshold that      ║
║             maximises Information Gain (or minimises Gini).      ║
║             Repeat recursively until stopping criteria hit.      ║
║                                                                  ║
║  PREDICTION: Start at root, follow YES/NO branches, return       ║
║              the label at the leaf you reach.                    ║
║              Time complexity: O(depth) — extremely fast!         ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  CONTROL OVERFITTING WITH:                                       ║
║    max_depth        ← limits tree depth directly                 ║
║    min_samples_leaf ← prevents tiny noisy leaves                 ║
║    ccp_alpha        ← post-training cost-complexity pruning      ║
╠══════════════════════════════════════════════════════════════════╣
║  NEXT STEP → Random Forests: average 100s of trees               ║
║              to dramatically reduce variance                     ║
╚══════════════════════════════════════════════════════════════════╝
```

> 📂 **Next**: [exercises.md](exercises.md) — 10 hands-on exercises to master decision trees.
> 📌 **Coming up**: [04 — Random Forests](../04-random_forest/README.md) — the upgrade that fixes decision trees' biggest weakness.
