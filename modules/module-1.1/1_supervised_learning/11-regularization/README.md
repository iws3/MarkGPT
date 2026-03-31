# 🔧 Lesson 11 — Regularisation

> **Core Idea**: Regularisation is the collection of techniques we use to prevent a model from memorising training data (overfitting) so it generalises well to new, unseen data. Think of it as adding a "complexity penalty" that forces the model to stay simple.

---

## 📋 Table of Contents

1. [The Overfitting Problem — Why We Need Regularisation](#1-overfitting)
2. [L1 Regularisation — Lasso](#2-l1)
3. [L2 Regularisation — Ridge](#3-l2)
4. [Elastic Net — Combining Both](#4-elastic-net)
5. [Dropout for Neural Networks](#5-dropout)
6. [Early Stopping](#6-early-stopping)
7. [Data Augmentation as Regularisation](#7-data-augmentation)
8. [Python Implementation](#8-python-implementation)
9. [Visual Summary](#9-visual-summary)

---

## 1. The Overfitting Problem

Overfitting happens when a model is too flexible and learns the specific noise in the training data rather than the underlying pattern. The result is a model that performs brilliantly on training data but poorly on new data.

```
True pattern:  y = 2x + noise

Overfitting (degree-15 polynomial):
  y = x + 3x² − 2x³ + 5x⁴ − ... + noise × coefficients
  Fits EVERY training point perfectly.
  Goes completely off-rails between training points.
  Training R² = 1.00  but  Test R² = −2.47

Underfitting (flat line):
  y = 5 (always predicts the mean)
  Training R² = 0.00  and  Test R² ≈ 0.00

Just right (degree-1 polynomial with regularisation):
  y = 2.1x + 0.3
  Training R² = 0.91  and  Test R² = 0.89  ← generalises!
```

Regularisation adds a penalty term to the loss function that discourages complexity:

```
Regularised loss = Original loss + λ × Complexity penalty

λ (lambda) = regularisation strength:
  λ = 0     → no regularisation (plain model, can overfit)
  λ = small → slight penalty (mild regularisation)
  λ = large → heavy penalty (strong simplification, may underfit)
```

---

## 2. L1 Regularisation — Lasso

L1 adds the **sum of absolute values** of coefficients as the penalty:

```
J(θ) = MSE + λ × Σ|θⱼ|

Geometric interpretation:
  Unconstrained optimum:    Constrained to L1 ball:
       ●  (best θ without   The L1 ball (diamond shape) forces
          regularisation)    some coefficients exactly to zero.
                             The solution tends to lie at a CORNER
                             of the diamond where some θ = 0.

Effect: SPARSE solutions — many coefficients become exactly 0.
        Lasso performs automatic feature selection!

When θⱼ = 0, feature j is completely ignored by the model.
This makes Lasso invaluable when you have many irrelevant features.
```

---

## 3. L2 Regularisation — Ridge

L2 adds the **sum of squared values** of coefficients as the penalty:

```
J(θ) = MSE + λ × Σθⱼ²

Effect: Shrinks ALL coefficients toward zero, but rarely makes them exactly zero.
        All features stay in the model but with smaller influence.

Why does squaring give different behaviour than absolute value?
  L1 (diamond): the solution lies at a corner → some θ = 0 exactly
  L2 (circle):  the solution rarely lies at a corner → all θ slightly shrunk

When to use Ridge vs Lasso:
  Ridge: All features are probably somewhat relevant. Handles multicollinearity.
  Lasso: Many features are probably irrelevant. Want automatic feature selection.
  Elastic Net: Unsure which situation applies → use both!
```

---

## 4. Elastic Net — Combining Both

Elastic Net linearly combines L1 and L2 penalties:

```
J(θ) = MSE + λ₁ × Σ|θⱼ| + λ₂ × Σθⱼ²

sklearn: ElasticNet(alpha=0.5, l1_ratio=0.5)
  alpha    = overall regularisation strength (λ)
  l1_ratio = what fraction is L1 (l1_ratio=0 → pure Ridge, l1_ratio=1 → pure Lasso)

Elastic Net is especially useful when:
  - There are many correlated features (Lasso would pick one and drop the rest; Elastic Net keeps all)
  - You want some sparsity but also handle collinearity
```

---

## 5. Dropout for Neural Networks

Dropout is the neural network analogue of regularisation. During training, each neuron is randomly "turned off" with probability p (the dropout rate):

```
Without dropout:          With Dropout(rate=0.5):
  Layer has 4 neurons       Randomly half the neurons are disabled each step
  All 4 always active       This step: neurons 1 and 3 are active
                             Next step: neurons 2 and 4 are active

Why does this work as regularisation?
  1. Neurons cannot co-adapt — each must learn to function without relying on others.
  2. It's equivalent to training an exponentially large ensemble of networks
     (each mask pattern = a different sub-network).
  3. At test time: all neurons active, weights scaled by (1 − rate).

Typical dropout rates: 0.2 to 0.5 for hidden layers. Never on the output layer.
```

---

## 6. Early Stopping

The simplest regularisation technique: stop training as soon as the validation loss stops improving:

```
Training loss:  ████████████████░░░░░░░  (keeps decreasing)
Validation loss: ████████████░░░░░░░░░░  (starts INCREASING at the "sweet spot")
                              ↑
                         Stop here! Save weights from this point.
                         This is called "restore_best_weights=True" in Keras.

Early stopping is essentially free regularisation — it requires no extra computation
and has no extra hyperparameter (other than the patience, i.e., how many epochs to wait
after the validation loss stops improving before giving up).
```

---

## 7. Data Augmentation as Regularisation

For image and audio data, you can artificially enlarge the training set by applying transformations that don't change the label:

```
Original image:  [cat photo]
Augmented:  [horizontally flipped cat] → still a cat, so label unchanged
            [slightly rotated cat]    → still a cat
            [zoomed-in cat]           → still a cat
            [colour-jittered cat]     → still a cat

Effect: The model sees many variations of each training example.
        It cannot memorise individual pixels — it must learn the actual concept.
        This is regularisation by effectively increasing training set diversity.

For tabular data: add Gaussian noise to features, or use SMOTE for class balancing.
```

---

## 8. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate an overfit-prone dataset
np.random.seed(42)
X = np.sort(np.random.uniform(0, 1, 30)).reshape(-1, 1)
y = np.sin(2 * np.pi * X.ravel()) + np.random.normal(0, 0.3, 30)
X_test = np.linspace(0, 1, 100).reshape(-1, 1)

# Overfit: degree-10 polynomial, no regularisation
for degree, reg, alpha in [(10, None, 0),
                            (10, 'ridge', 0.001),
                            (10, 'ridge', 10.0),
                            (10, 'lasso', 0.001),
                            (10, 'lasso', 0.1)]:
    if reg is None:
        from sklearn.linear_model import LinearRegression
        model = Pipeline([('poly', PolynomialFeatures(degree)),
                          ('scaler', StandardScaler()),
                          ('reg', LinearRegression())])
    elif reg == 'ridge':
        model = Pipeline([('poly', PolynomialFeatures(degree)),
                          ('scaler', StandardScaler()),
                          ('reg', Ridge(alpha=alpha))])
    else:
        model = Pipeline([('poly', PolynomialFeatures(degree)),
                          ('scaler', StandardScaler()),
                          ('reg', Lasso(alpha=alpha, max_iter=10000))])

    model.fit(X, y)
    train_mse = np.mean((y - model.predict(X))**2)
    # Use a fresh test set
    X_test_eval = np.random.uniform(0, 1, 50).reshape(-1, 1)
    y_test_eval = np.sin(2 * np.pi * X_test_eval.ravel()) + np.random.normal(0, 0.3, 50)
    test_mse = np.mean((y_test_eval - model.predict(X_test_eval))**2)
    label = f"deg={degree}, {reg or 'none'}, α={alpha}"
    print(f"{label:40s}: train MSE={train_mse:.4f}, test MSE={test_mse:.4f}")

# ─── Lasso path — which features survive as lambda increases? ───────────
from sklearn.linear_model import lasso_path
X_r, y_r = make_regression(n_samples=100, n_features=20, n_informative=5, noise=10)
alphas, coefs, _ = lasso_path(X_r, y_r, alphas=np.logspace(-1, 2, 100))
plt.figure(figsize=(10, 5))
plt.semilogx(alphas[::-1], coefs.T[::-1])  # plot coefficient vs alpha
plt.xlabel('alpha (regularisation strength)')
plt.ylabel('Coefficient value')
plt.title('Lasso Regularisation Path — Features Disappear as Lambda Grows')
plt.axvline(x=1.0, color='red', linestyle='--', label='λ=1 (reasonable default)')
plt.legend()
plt.show()
```

---

## 9. Visual Summary

```
╔══════════════════════════════════════════════════════════════════╗
║               REGULARISATION — OVERVIEW                         ║
╠══════════════════════════════════════════════════════════════════╣
║  METHOD          PENALTY            EFFECT                       ║
║  ─────────────── ─────────────────  ─────────────────────────── ║
║  Ridge (L2)      λΣθⱼ²             Shrinks all weights          ║
║  Lasso (L1)      λΣ|θⱼ|            Zeros out irrelevant weights  ║
║  Elastic Net     λ₁Σ|θⱼ|+λ₂Σθⱼ²   Combination of both           ║
║  Dropout         Random node disab. Neural network regulariser  ║
║  Early Stopping  Stop at best val.  Universal technique          ║
║  Data Augment.   More diverse data  Image/audio tasks           ║
╠══════════════════════════════════════════════════════════════════╣
║  λ too small → overfitting. λ too large → underfitting.         ║
║  Always tune λ via cross-validation, not test set!              ║
╚══════════════════════════════════════════════════════════════════╝
```
