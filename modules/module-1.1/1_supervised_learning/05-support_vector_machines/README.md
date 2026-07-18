# ⚔️ Lesson 05 — Support Vector Machines (SVM)

> **Core Idea**: Find the hyperplane that separates two classes with the **maximum possible margin** — the widest "road" between the two groups. Points closest to the boundary (support vectors) define this margin and are the only ones that matter.

---

## 📋 Table of Contents

1. [Intuition — The Widest Street](#1-intuition)
2. [The Margin and Support Vectors](#2-margin-and-support-vectors)
3. [The Optimisation Problem](#3-optimisation-problem)
4. [The Kernel Trick — Non-Linear Boundaries](#4-kernel-trick)
5. [Soft Margin — Handling Noise](#5-soft-margin)
6. [SVM for Regression (SVR)](#6-svr)
7. [Key Hyperparameters](#7-hyperparameters)
8. [Python Implementation](#8-python-implementation)
9. [Visual Summary](#9-visual-summary)
10. [When to Use](#10-when-to-use)

---

## 1. Intuition — The Widest Street

Imagine you're drawing a boundary between two groups of points. You could draw infinitely many lines that separate them correctly. But which is the *best* line? 

SVM says: pick the line that has the largest "buffer zone" on each side — the widest possible road between the two groups. This makes the classifier most robust to slight variations in new data.

```
Bad boundary (no margin):   SVM boundary (maximum margin):

  ✗ ✗ |● ●                  ✗ ✗ ‖     ‖ ● ●
  ✗   |  ●                  ✗   ‖     ‖   ●
    ✗ |●                      ✗ ‖     ‖ ●
                                  ^^^^^
  Barely works             Margin: the wider the better
                           Support vectors are on the ‖ lines
```

The boundary line is called the **decision hyperplane**. The gap on each side is the **margin**. SVM's job is to find the hyperplane with the maximum margin.

---

## 2. The Margin and Support Vectors

The margin is determined by the training examples that are *closest* to the decision boundary. These critical points are called **support vectors** — they literally "support" (define) the position and orientation of the boundary.

```
Feature space:
                  ●   ●
              ● →[SV]←          Support vector of class +1
   ───────── margin ─────────   ← Decision boundary (hyperplane)
             →[SV]← ✗          Support vector of class -1
               ✗   ✗

Margin width = 2 / ||w||  where w is the weight vector (normal to boundary)
Maximising margin = minimising ||w||²

Key insight: move a non-support-vector point and the boundary doesn't change.
Move a support vector even slightly and the whole boundary shifts.
```

---

## 3. The Optimisation Problem

SVM finds the optimal boundary by solving a constrained optimisation problem:

```
Minimise:   (1/2) ||w||²                     ← maximise margin (= minimise ||w||)

Subject to: yᵢ(w·xᵢ + b) ≥ 1  for all i     ← all points correctly classified
                                                 and outside the margin

Where:
  w = weight vector (direction of boundary)
  b = bias (offset of boundary)
  yᵢ = +1 or −1 (class label)
  xᵢ = feature vector

This is a quadratic programming problem with a unique global optimum.
```

---

## 4. The Kernel Trick — Non-Linear Boundaries

What if the classes are not linearly separable? The **kernel trick** maps data into a higher-dimensional space where it *is* linearly separable — without ever explicitly computing the transformation.

```
Original 2D space (not linearly separable):

  ●●●●●           ← class 1 (surrounding ring)
●         ●
●  ✗✗✗✗  ●        ← class 0 (inside)
●  ✗✗✗✗  ●
  ●●●●●

Impossible to draw a straight line between them.

Map to 3D using kernel φ(x): z = x₁² + x₂²
  ← class 1 projects to high z values (large radius)
  ← class 0 projects to low z values (small radius)

Now they're linearly separable by a horizontal plane in 3D!
The kernel computes this inner product WITHOUT the expensive mapping.
```

**Common Kernels:**

```
┌───────────────┬──────────────────────────────────────┬───────────────┐
│ Kernel        │ Formula K(xᵢ, xⱼ)                    │ When to use   │
├───────────────┼──────────────────────────────────────┼───────────────┤
│ Linear        │ xᵢ · xⱼ                               │ Linearly sep. │
│ Polynomial    │ (γ xᵢ·xⱼ + r)^d                      │ Polynomial    │
│ RBF/Gaussian  │ exp(−γ ||xᵢ − xⱼ||²)                 │ Most common   │
│ Sigmoid       │ tanh(γ xᵢ·xⱼ + r)                    │ Neural-like   │
└───────────────┴──────────────────────────────────────┴───────────────┘
The RBF kernel is usually a great default choice.
```

---

## 5. Soft Margin — Handling Noise

Real data has noise and outliers. A **hard margin** SVM requires perfect separation, which often doesn't exist. The **soft margin** SVM (C-SVM) allows some misclassification, controlled by the C parameter:

```
Objective: Minimise (1/2)||w||² + C × Σ ξᵢ

Where ξᵢ ≥ 0 are slack variables allowing points inside the margin or misclassified.

C large:  penalise margin violations heavily → narrow margin, fits training closely
          Risk: overfitting
C small:  allow more violations → wider margin, more generalised
          Risk: underfitting

Intuition:
  C = ∞  →  hard margin (no violations tolerated)
  C = 0.1  →  lots of margin violations allowed, very wide margin
```

---

## 6. SVM for Regression (SVR)

SVR flips the margin idea: instead of keeping most points *outside* the margin (classification), SVR tries to fit as many points *inside* a tube of width ε around the predicted line:

```
SVR prediction tube:

  ●        ●
    ────────────── upper bound (y_pred + ε)
  ○ ○ ○ ○ ○ ○      ← points INSIDE tube: no loss, counted as correct
    ────────────── lower bound (y_pred − ε)
            ●  ●
Points outside the tube are penalised (support vectors for regression).
```

---

## 7. Key Hyperparameters

```
┌───────────┬────────────────────────────────────────────────────────┐
│ Parameter │ Effect                                                  │
├───────────┼────────────────────────────────────────────────────────┤
│ C         │ Regularisation strength (inverse).                     │
│           │ Large C: harder margin, may overfit.                   │
│           │ Small C: wider margin, may underfit.                   │
│           │ Default: 1.0. Try: 0.001, 0.01, 0.1, 1, 10, 100.     │
├───────────┼────────────────────────────────────────────────────────┤
│ kernel    │ 'rbf' (default), 'linear', 'poly', 'sigmoid'.         │
│           │ RBF works for most problems. Linear if n_features>>n.  │
├───────────┼────────────────────────────────────────────────────────┤
│ gamma     │ Kernel width for RBF. Controls radius of influence.    │
│           │ 'scale' (default) = 1/(n_features × X.var()).         │
│           │ Large γ: each point has local influence → overfit.    │
│           │ Small γ: each point has global influence → underfit.  │
├───────────┼────────────────────────────────────────────────────────┤
│ degree    │ For polynomial kernel only. Degree of polynomial.      │
│           │ Default: 3. Larger = more complex boundary.            │
└───────────┴────────────────────────────────────────────────────────┘
Always scale features before SVM — the algorithm is very sensitive to feature magnitudes!
```

---

## 8. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ─── Example 1: Linear SVM on breast cancer ─────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

# CRITICAL: always scale before SVM!
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm.fit(X_train_s, y_train)
print(f"Test accuracy: {svm.score(X_test_s, y_test):.4f}")
print(f"Support vectors: {svm.n_support_}")  # one count per class
print(classification_report(y_test, svm.predict(X_test_s)))

# ─── Example 2: Non-linear SVM on moons dataset ─────────────────────────
X_m, y_m = make_moons(n_samples=300, noise=0.2, random_state=42)
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_m, y_m, test_size=0.2)
scaler_m = StandardScaler()
X_m_s = scaler_m.fit_transform(X_m_train)

# Compare linear vs RBF kernel
for kernel in ['linear', 'rbf', 'poly']:
    clf = SVC(kernel=kernel, C=1.0, random_state=42)
    clf.fit(X_m_s, y_m_train)
    acc = clf.score(scaler_m.transform(X_m_test), y_m_test)
    print(f"Kernel={kernel}: test accuracy = {acc:.3f}")

# ─── Hyperparameter tuning via GridSearchCV ──────────────────────────────
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.001, 0.01, 0.1]}
grid = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_s, y_train)
print(f"\nBest params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")
print(f"Test accuracy (best SVM): {grid.score(X_test_s, y_test):.4f}")
```

---

## 9. Visual Summary

```
╔══════════════════════════════════════════════════════════════════╗
║                    SVM — OVERVIEW                               ║
╠══════════════════════════════════════════════════════════════════╣
║  GOAL: Find the hyperplane with MAXIMUM MARGIN                   ║
║        Only support vectors define the boundary                  ║
║                                                                  ║
║  FOR NON-LINEAR DATA: Use kernel trick (usually RBF)             ║
║        Maps to higher dimensions implicitly                      ║
║                                                                  ║
║  TRADE-OFF:  Large C → narrow margin, fits data tightly         ║
║              Small C → wide margin, more generalised            ║
║              Large γ → local decisions, complex boundary        ║
║              Small γ → global decisions, smooth boundary        ║
║                                                                  ║
║  ALWAYS scale features before training SVM!                      ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 10. When to Use

SVM excels when the number of features is large relative to the number of training examples (such as text classification with TF-IDF features), when the classes are nearly linearly separable or become so in a higher-dimensional kernel space, and when you need good generalisation with a theoretically principled method. It struggles when you have more than about 100,000 training examples (training time is O(n²) to O(n³)), when you need fast online learning (training is a batch process), or when interpretability is critical (the decision boundary is hard to explain).

> 📂 Next: [exercises.md](exercises.md) | Then: [06 — Naive Bayes](../06-naive_bayes/README.md)
