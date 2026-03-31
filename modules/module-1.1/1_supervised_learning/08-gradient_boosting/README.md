# 🚀 Lesson 08 — Gradient Boosting

> **Core Idea**: Build an ensemble of weak learners (shallow decision trees) sequentially. Each new tree focuses on correcting the errors made by all previous trees. The ensemble grows smarter with every addition, turning many weak learners into one powerful predictor.

---

## 📋 Table of Contents

1. [Boosting vs Bagging — The Key Difference](#1-boosting-vs-bagging)
2. [Gradient Boosting Step by Step](#2-gradient-boosting)
3. [Residuals as Pseudo-Gradients](#3-residuals)
4. [The Learning Rate — Shrinkage](#4-learning-rate)
5. [XGBoost, LightGBM, CatBoost](#5-modern-implementations)
6. [Key Hyperparameters](#6-hyperparameters)
7. [Python Implementation](#7-python-implementation)
8. [Visual Summary](#8-visual-summary)

---

## 1. Boosting vs Bagging — The Key Difference

Both create ensembles of trees, but with opposite philosophies:

```
BAGGING (Random Forest):              BOOSTING (Gradient Boosting):
  Train trees IN PARALLEL               Train trees SEQUENTIALLY
  On RANDOM subsets of data             Each tree focuses on ERRORS of previous
  Reduces VARIANCE                      Reduces both BIAS and VARIANCE
  Trees are INDEPENDENT                 Trees are DEPENDENT on each other
  Ensemble votes / averages             Ensemble adds weighted predictions

  Good default choice.                  Often achieves higher accuracy
  Hard to overfit.                        but easier to overfit.
                                        Slower to train.
```

---

## 2. Gradient Boosting Step by Step

Here is the complete training algorithm, laid out step by step:

```
Step 1: Start with a naive initial prediction.
        For regression: ŷ₀ = mean(y)  (just predict the average)
        For classification: ŷ₀ = log(p/(1-p))  (log-odds of class proportion)

Step 2: Compute the RESIDUALS (errors of the current model):
        rᵢ = yᵢ − ŷᵢ   (how wrong are we for each training example?)

Step 3: Train a SHALLOW DECISION TREE to predict the residuals r.
        (Not the original y — we're learning to correct the current mistakes)
        Typical max_depth = 3 to 5.

Step 4: Add this tree's predictions to the ensemble (with a small learning rate α):
        ŷᵢ ← ŷᵢ + α × (tree₁'s prediction for xᵢ)

Step 5: Compute new residuals from the UPDATED predictions.

Step 6: Repeat steps 3–5 for N iterations (N trees total).

Final prediction: ŷ = ŷ₀ + α×T₁(x) + α×T₂(x) + ... + α×Tₙ(x)
```

**Intuition**: Each tree "corrects" the previous ensemble. The first tree corrects the naive prediction, the second corrects the first tree's corrections, and so on. After many iterations, the model becomes highly accurate.

---

## 3. Residuals as Pseudo-Gradients

The "gradient" in gradient boosting refers to gradient descent in **function space**. Instead of adjusting parameters, we're adding functions (trees) to reduce the loss:

```
Loss function: L(y, ŷ) — measures prediction error

The residual  rᵢ = yᵢ − ŷᵢ  is the NEGATIVE GRADIENT of L with respect to ŷ:
  −∂L/∂ŷ = y − ŷ  (for MSE loss)

This is why we train the next tree to predict residuals — we're following the gradient
downhill in the space of all possible prediction functions.

For different loss functions, the "pseudo-residuals" are different:
  MSE loss:              rᵢ = yᵢ − ŷᵢ
  Log loss (classifier): rᵢ = yᵢ − σ(ŷᵢ)   (sigmoid of current score)
  Huber loss:            rᵢ = clip(yᵢ − ŷᵢ, −δ, δ)  (robust to outliers)
```

---

## 4. The Learning Rate — Shrinkage

The learning rate α (called "shrinkage") scales the contribution of each tree:

```
ŷ ← ŷ + α × (new tree prediction)

Small α (e.g., 0.01):  Each tree contributes very little.
  Needs many more trees to converge.
  But final model generalises much better — regularisation effect!
  Rule: Smaller α + More trees = Usually better accuracy.

Large α (e.g., 0.5):  Each tree contributes a lot.
  Converges faster (fewer trees needed).
  Risk: overshoots the minimum, oscillates, overfits.

Classic trade-off:
  α = 0.1 with n_estimators = 100  ≈  α = 0.01 with n_estimators = 1000
  (Similar performance, very different training time)
```

---

## 5. Modern Implementations

Original gradient boosting (sklearn's `GradientBoostingClassifier`) is accurate but slow. Three modern implementations are dramatically faster and more powerful:

```
┌────────────┬──────────────────────────────────────────────────────────┐
│ Library    │ Key Innovations                                          │
├────────────┼──────────────────────────────────────────────────────────┤
│ XGBoost    │ Regularised objective, efficient handling of missing     │
│            │ values, column/row subsampling (like random forests),    │
│            │ GPU support. The "original" modern boosting library.     │
├────────────┼──────────────────────────────────────────────────────────┤
│ LightGBM   │ Leaf-wise (not level-wise) tree growth — much faster    │
│ (Microsoft)│ for large datasets. Histogram-based binning. Uses less  │
│            │ memory. Best choice for large datasets (>100K rows).    │
├────────────┼──────────────────────────────────────────────────────────┤
│ CatBoost   │ Native handling of categorical features (no need for    │
│ (Yandex)   │ one-hot encoding). Ordered boosting reduces overfitting.│
│            │ Often the best out-of-the-box without tuning.           │
└────────────┴──────────────────────────────────────────────────────────┘
For Kaggle competitions and production tabular data: try all three, use the best.
```

---

## 6. Key Hyperparameters

```
┌──────────────────────┬──────────────────────────────────────────────────┐
│ Hyperparameter       │ Effect and Guidance                              │
├──────────────────────┼──────────────────────────────────────────────────┤
│ n_estimators         │ Number of trees. More = better up to a point.   │
│                      │ Use early stopping to find the right amount.     │
├──────────────────────┼──────────────────────────────────────────────────┤
│ learning_rate        │ Shrinkage. Start at 0.1, reduce with more trees. │
│                      │ Try: 0.001, 0.01, 0.05, 0.1, 0.3.              │
├──────────────────────┼──────────────────────────────────────────────────┤
│ max_depth            │ Depth of each tree. Keep shallow! Try 3, 4, 5.  │
│                      │ Deeper trees capture more interactions.          │
├──────────────────────┼──────────────────────────────────────────────────┤
│ subsample            │ Fraction of training data used per tree (0–1).   │
│                      │ Adds stochasticity, reduces overfitting.         │
│                      │ Try: 0.5, 0.75, 1.0.                           │
├──────────────────────┼──────────────────────────────────────────────────┤
│ min_samples_leaf     │ Minimum samples per leaf. Larger = more pruning. │
│ (or min_child_weight)│ Increases regularisation.                       │
├──────────────────────┼──────────────────────────────────────────────────┤
│ early_stopping_rounds│ Stop if no improvement after N rounds.          │
│ (XGBoost/LightGBM)   │ Use a validation set. Prevents overfitting.     │
└──────────────────────┴──────────────────────────────────────────────────┘
```

---

## 7. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb  # pip install xgboost

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ─── Sklearn GradientBoosting ────────────────────────────────────────────
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                 max_depth=3, random_state=42)
gb.fit(X_train, y_train)
print(f"GradientBoosting test accuracy: {gb.score(X_test, y_test):.4f}")

# Plot training deviance (loss) vs n_estimators
train_score = np.zeros(200)
test_score  = np.zeros(200)
for i, y_pred in enumerate(gb.staged_predict_proba(X_train)):
    train_score[i] = gb.loss_(y_train, y_pred[:, 1])
for i, y_pred in enumerate(gb.staged_predict_proba(X_test)):
    test_score[i]  = gb.loss_(y_test, y_pred[:, 1])
plt.plot(train_score, 'b-', label='Train loss')
plt.plot(test_score,  'r-', label='Test loss')
plt.xlabel('Number of trees'); plt.ylabel('Log loss')
plt.legend(); plt.title('Gradient Boosting: Loss vs n_estimators'); plt.show()

# ─── XGBoost with early stopping ─────────────────────────────────────────
xgb_model = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', early_stopping_rounds=20,
    random_state=42, use_label_encoder=False
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"XGBoost best iteration: {xgb_model.best_iteration}")
print(f"XGBoost test accuracy:  {xgb_model.score(X_test, y_test):.4f}")

# Feature importance
xgb.plot_importance(xgb_model, max_num_features=10)
plt.show()
```

---

## 8. Visual Summary

```
╔══════════════════════════════════════════════════════════════════╗
║                GRADIENT BOOSTING — OVERVIEW                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Initial prediction (mean/log-odds)                             ║
║       + α × Tree₁ (learns residuals of initial prediction)      ║
║       + α × Tree₂ (learns residuals of prediction after Tree₁)  ║
║       + α × Tree₃ ...                                           ║
║       + α × Treeₙ                                               ║
║  ──────────────────────────────────────────────────────────      ║
║  Final prediction = weighted sum of all tree predictions        ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  KEY TUNING RECIPE:                                              ║
║  1. Start: learning_rate=0.1, max_depth=3, n_estimators=100     ║
║  2. Use early stopping to find optimal n_estimators             ║
║  3. Lower learning_rate (e.g., 0.01), increase n_estimators     ║
║  4. Add subsample and colsample_bytree for regularisation       ║
║  5. Use XGBoost/LightGBM for speed on large datasets            ║
╚══════════════════════════════════════════════════════════════════╝
```
