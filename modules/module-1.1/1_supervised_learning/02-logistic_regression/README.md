# 🔀 Lesson 02 — Logistic Regression

> **Core Idea**: Use a linear model to produce a *probability* — the chance that an input belongs to a particular class — then make a classification decision based on that probability.

---

## 📋 Table of Contents

1. [Why Not Just Use Linear Regression for Classification?](#1-why-not-linear-regression)
2. [The Sigmoid Function — Squishing to Probabilities](#2-the-sigmoid-function)
3. [The Decision Boundary](#3-the-decision-boundary)
4. [The Cost Function — Binary Cross-Entropy](#4-the-cost-function)
5. [Training with Gradient Descent](#5-gradient-descent)
6. [Multi-class Classification — Softmax](#6-multi-class-softmax)
7. [Evaluating a Classifier](#7-evaluation)
8. [Regularisation in Logistic Regression](#8-regularisation)
9. [Python Implementation](#9-python-implementation)
10. [Visual Summary](#10-visual-summary)
11. [When to Use](#11-when-to-use)

---

## 1. Why Not Linear Regression for Classification?

Suppose you want to predict whether an email is spam (1) or not spam (0). You could try linear regression:

```
Predicted value from linear regression:
  −0.3   0.1   0.8   1.4   2.1 ...

  Problem 1: Values can go below 0 and above 1
             But probabilities must be in [0, 1]!
  Problem 2: What threshold do you use?
  Problem 3: A very extreme outlier shifts the entire line
```

We need a model whose output is *always* between 0 and 1 — so it can represent a probability. That's exactly what logistic regression provides.

---

## 2. The Sigmoid Function

The "trick" in logistic regression is to wrap the linear model inside a special S-shaped function called the **sigmoid** (or logistic) function:

```
σ(z) = 1 / (1 + e^(−z))

Where z = θ₀ + θ₁x₁ + θ₂x₂ + ... (the linear combination, same as before)

Key properties:
  σ(0)    = 0.5  (50% probability)
  σ(+∞)   → 1.0  (very certain it's class 1)
  σ(−∞)   → 0.0  (very certain it's class 0)
  Always between 0 and 1 ✅

Shape:
  P(class=1)
  1.0 │                              ┌──────────
      │                          ╱──╯
  0.5 │──────────────────────────╱
      │                  ╱──────╯
  0.0 │─────────────────
      └────────────────────────────────────────► z (linear score)
           negative z        positive z
           → likely class 0  → likely class 1
```

The output `ŷ = σ(θᵀx)` is interpreted as: **"the probability that this example belongs to class 1"**.

---

## 3. The Decision Boundary

We classify an example as class 1 if `ŷ ≥ 0.5`, and class 0 otherwise. Since σ(z) ≥ 0.5 when z ≥ 0, this means we predict class 1 when:

```
θᵀx ≥ 0   →   class 1
θᵀx < 0   →   class 0

The DECISION BOUNDARY is the line (or surface) where θᵀx = 0
```

Visually for two features:

```
x₂
│            ✗  ✗  ✗  (class 0)
│         ✗  ✗
│                        Decision boundary line: θ₀ + θ₁x₁ + θ₂x₂ = 0
│──────────────────────────────────
│  ○   ○                          (class 1)
│     ○  ○  ○
└──────────────────────────────────────────────► x₁

For non-linear boundaries: add polynomial features
  e.g., a circle: θ₀ + θ₁x₁² + θ₂x₂² = 0  → circular boundary
```

---

## 4. The Cost Function — Binary Cross-Entropy

We cannot use MSE for logistic regression — it produces a non-convex cost function with many local minima. Instead, we use **binary cross-entropy** (also called log loss):

```
For a single example:
  If y = 1:  Cost = −log(ŷ)        ← if we predict ŷ=1, cost=0; if ŷ≈0, cost→∞
  If y = 0:  Cost = −log(1 − ŷ)   ← if we predict ŷ=0, cost=0; if ŷ≈1, cost→∞

Combined (elegant single formula):
  Cost(ŷ, y) = −y log(ŷ) − (1−y) log(1−ŷ)

For the entire training set:
  J(θ) = −(1/m) × Σ [ yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ) ]

Intuition: The cost is 0 when our prediction is correct and confident.
           The cost → ∞ when we are confidently WRONG.
           The logarithm penalises confident wrong predictions extremely harshly.
```

This function is convex — gradient descent will always find the global minimum.

---

## 5. Training with Gradient Descent

The update rule looks almost identical to linear regression — only the hypothesis function changes:

```
Repeat until convergence:
  θⱼ := θⱼ − α × (1/m) × Σ (ŷᵢ − yᵢ) × xᵢⱼ

Where ŷᵢ = σ(θᵀxᵢ)   ← the sigmoid, not just θᵀxᵢ

Same formula, different ŷ.
```

---

## 6. Multi-class Classification — Softmax

Logistic regression extends naturally to more than 2 classes using the **Softmax** function:

```
For K classes, the model outputs K scores: z₁, z₂, ..., zK

P(class k) = e^(zk) / (e^(z₁) + e^(z₂) + ... + e^(zK))
           = softmax(zk)

Properties:
  - All outputs between 0 and 1
  - All outputs sum to 1  →  a valid probability distribution!
  - The class with highest probability is predicted

Example (3-class flower classification):
  Scores z = [2.1, 0.5, −0.3]
  Softmax   = [0.78, 0.17, 0.05]
  Prediction: class 0 (78% confidence)
```

Scikit-learn handles this automatically with `multi_class='multinomial'`.

---

## 7. Evaluating a Classifier

Accuracy alone is dangerous — especially with imbalanced classes. Always check the confusion matrix:

```
                  Predicted
                  0        1
Actual  0   [   TN    |   FP   ]
        1   [   FN    |   TP   ]

TN = True Negative  (correctly said "no")
TP = True Positive  (correctly said "yes")
FP = False Positive (incorrectly said "yes" — "false alarm")
FN = False Negative (missed a real positive — "missed alarm")

Precision = TP / (TP + FP)  ← "Of everything I flagged, how much was really positive?"
Recall    = TP / (TP + FN)  ← "Of all real positives, how many did I catch?"
F1 Score  = 2 × (P × R) / (P + R)  ← harmonic mean, good single-number summary

ROC-AUC: Plot TPR vs FPR at every possible threshold.
         AUC = 1.0 → perfect classifier
         AUC = 0.5 → no better than random guessing
```

The **classification threshold** (default 0.5) can be adjusted depending on your problem. For cancer screening, you'd lower it to 0.3 to catch more true positives (better recall), accepting more false alarms (lower precision).

---

## 8. Regularisation

Logistic regression also suffers from overfitting, especially with many features. The `C` parameter in scikit-learn is the *inverse* of regularisation strength:

```
C large  →  weak regularisation  →  may overfit
C small  →  strong regularisation  →  may underfit

LogisticRegression(C=1.0)            ← default
LogisticRegression(C=100, penalty='l2')   ← strong model, less regularisation
LogisticRegression(C=0.01, penalty='l1')  ← heavy L1, sparse coefficients
```

---

## 9. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)

# ─── Load data ──────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target   # y: 1=malignant, 0=benign

# ─── Split and scale ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42,
                                                     stratify=y)  # preserve class ratio!
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─── Train ──────────────────────────────────────────────────────────────────
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)

# ─── Evaluate ───────────────────────────────────────────────────────────────
y_pred      = model.predict(X_test_s)
y_pred_prob = model.predict_proba(X_test_s)[:, 1]  # P(class=1)

print(classification_report(y_test, y_pred, target_names=data.target_names))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")

# ─── Confusion matrix ───────────────────────────────────────────────────────
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.show()

# ─── ROC Curve ──────────────────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_prob):.3f}')
plt.plot([0,1], [0,1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

## 10. Visual Summary

```
╔══════════════════════════════════════════════════════════════════╗
║               LOGISTIC REGRESSION — OVERVIEW                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  INPUT ──► LINEAR SCORE ──► SIGMOID ──► PROBABILITY ──► CLASS   ║
║    X     θ₀+θ₁x₁+θ₂x₂      σ(z)        0.0 to 1.0    0 or 1   ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  COST FUNCTION: Binary Cross-Entropy (log loss)                  ║
║  J(θ) = −(1/m) Σ [y log(ŷ) + (1−y) log(1−ŷ)]                  ║
║  → Convex! Gradient descent guaranteed to find global minimum    ║
╠══════════════════════════════════════════════════════════════════╣
║  KEY METRICS: Accuracy, Precision, Recall, F1, ROC-AUC           ║
║  THRESHOLD: Default 0.5, adjust based on cost of errors          ║
║  REGULARISATION: L2 (default), L1 for sparsity                   ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 11. When to Use

Logistic regression works best when the classes are roughly linearly separable, when you need probability outputs (not just class labels), when interpretability matters (e.g., in healthcare or finance), and when you want a fast, reliable baseline for any binary classification problem. Avoid it when the decision boundary is highly non-linear — in those cases, try decision trees, SVM with RBF kernel, or neural networks.

---

> 📂 Next: [exercises.md](exercises.md) | After that: [03-decision_trees](../03-decision_trees/README.md)
