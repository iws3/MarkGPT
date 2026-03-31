# 🎯 Lesson 10 — Ensemble Methods

> **Core Idea**: Combine multiple models so their strengths amplify and their weaknesses cancel out. Ensembles almost always outperform any single model — this is the secret weapon of Kaggle champions and production ML engineers alike.

---

## 📋 Table of Contents

1. [Why Ensembles Work — The Diversity Principle](#1-diversity)
2. [Voting — Simple Majority Rule](#2-voting)
3. [Averaging — Soft Combination](#3-averaging)
4. [Bagging — Bootstrap Aggregation](#4-bagging)
5. [Boosting — Sequential Correction](#5-boosting)
6. [Stacking — Meta-Learning](#6-stacking)
7. [Blending — A Simpler Stacking](#7-blending)
8. [When to Ensemble](#8-when-to-ensemble)
9. [Python Implementation](#9-python-implementation)

---

## 1. Why Ensembles Work — The Diversity Principle

The key insight behind ensembles is that diverse models make different errors. When you combine them, errors cancel out:

```
Model A: 80% accurate  →  wrong on examples {3, 7, 12, 18, 25}
Model B: 80% accurate  →  wrong on examples {2, 7, 14, 21, 25}
Model C: 80% accurate  →  wrong on examples {5, 7, 16, 22, 25}

Only example 7 and 25 are wrong by ALL THREE models.
For all others, at least two models are correct → majority vote wins!

Ensemble accuracy: much higher than 80%, even though each base model is 80%.

CRITICAL: If all models make the SAME errors, combining them does nothing!
Models must be DIVERSE (trained differently, on different data, or using different algorithms).
```

---

## 2. Voting — Simple Majority Rule

The simplest ensemble: train multiple classifiers, let each vote, pick the majority:

```
Hard voting (use predicted classes):
  Classifier 1: spam
  Classifier 2: spam
  Classifier 3: not spam
  Result:       spam  (2 votes vs 1)

Soft voting (use predicted probabilities — usually better):
  Classifier 1: P(spam) = 0.80
  Classifier 2: P(spam) = 0.70
  Classifier 3: P(spam) = 0.30
  Average:      P(spam) = (0.80 + 0.70 + 0.30) / 3 = 0.60
  Result:       spam  (> 0.5)

Soft voting is better because it uses confidence information, not just the winner.
```

---

## 3. Averaging — Soft Combination

For regression, simply average the predictions of multiple models:

```
Model 1 prediction: $280,000
Model 2 prediction: $310,000
Model 3 prediction: $295,000
Ensemble:           $295,000  ← simple average

Weighted average (if you trust some models more):
  $280k × 0.5 + $310k × 0.3 + $295k × 0.2 = $290,500
```

---

## 4. Bagging — Bootstrap Aggregation

Train multiple versions of the same model on different bootstrap samples of the training data. We covered this in detail with Random Forests (the most famous bagging ensemble):

```
Bootstrap sample 1 → Train Model 1
Bootstrap sample 2 → Train Model 2
Bootstrap sample 3 → Train Model 3
...
All models predict → Average / Vote → Final prediction

Key effect: Reduces VARIANCE (fixes overfitting) without increasing bias.
sklearn: BaggingClassifier(DecisionTreeClassifier(), n_estimators=100)
```

---

## 5. Boosting — Sequential Correction

Train models sequentially, each one focusing on the mistakes of the previous. This reduces both bias and variance. Covered in full in Lesson 08. The key ensemble variants are AdaBoost (re-weights misclassified examples), Gradient Boosting (fits residuals), and XGBoost/LightGBM/CatBoost (optimised implementations).

---

## 6. Stacking — Meta-Learning

Stacking is the most powerful but most complex ensemble method. The idea is to use the predictions of base models as inputs to a "meta-learner" (also called a blender) that learns how to best combine them:

```
Layer 0 — Base Models (trained on training data):
  Model A: LogisticRegression
  Model B: RandomForest
  Model C: GradientBoosting
  Model D: SVM

Layer 1 — Meta-Learner (trained on base model PREDICTIONS):
  Input:  [pred_A, pred_B, pred_C, pred_D]  ← 4 features (one per base model)
  Output: Final prediction
  Typically a simple model: LogisticRegression or Ridge

Why not just pick the best base model?
  The meta-learner learns WHEN each base model is right.
  "When Model A says 'spam' and Model B says 'not spam', trust Model A on this type of email."
```

**Critical**: The meta-learner must be trained on predictions the base models made on data they haven't seen. Use cross-validation to generate out-of-fold predictions, otherwise the meta-learner just learns to trust whichever base model overfits most.

```
Correct stacking procedure:
  1. Split training data into K folds.
  2. For each fold k:
     a. Train each base model on the other K-1 folds.
     b. Generate predictions on fold k.
  3. After K iterations, each training example has predictions from all base models
     (generated on data they didn't train on — no data leakage!).
  4. Train meta-learner on these out-of-fold predictions.
  5. To predict on test data: average each base model's predictions across all K versions,
     then pass through meta-learner.
```

---

## 7. Blending — A Simpler Stacking

Blending is a simpler alternative: hold out a fixed "blend set" (e.g., 20% of training data), train base models on the rest, generate predictions on the blend set, and train the meta-learner on those predictions. Less rigorous than proper stacking but faster and still effective:

```python
from sklearn.model_selection import train_test_split

X_train_base, X_blend, y_train_base, y_blend = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train base models on X_train_base
# Generate predictions on X_blend (blending set)
# Train meta-learner on X_blend predictions
# Final test prediction: base models predict, meta-learner combines
```

---

## 8. When to Ensemble

Ensembles are worth the extra complexity when you need every last bit of accuracy (production models, competitions) and when you have diverse base models available. The diversity requirement is key: averaging two versions of the same algorithm on the same data gives little benefit. The best ensembles combine fundamentally different algorithms (linear models + trees + neural networks).

---

## 9. Python Implementation

```python
from sklearn.ensemble import (VotingClassifier, VotingRegressor,
                               BaggingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─── Voting ensemble ─────────────────────────────────────────────────────
estimators = [
    ('lr',  LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

for voting in ['hard', 'soft']:
    vc = VotingClassifier(estimators=estimators, voting=voting)
    score = cross_val_score(vc, X_train_s, y_train, cv=5).mean()
    print(f"VotingClassifier ({voting} voting): {score:.4f}")

# ─── Stacking ────────────────────────────────────────────────────────────
stacker = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=1.0, max_iter=1000),
    cv=5,            # use 5-fold cross-val to generate out-of-fold predictions
    passthrough=False  # only pass base model predictions to meta-learner
)
stacking_score = cross_val_score(stacker, X_train_s, y_train, cv=5).mean()
print(f"StackingClassifier: {stacking_score:.4f}")

# ─── Compare all approaches ───────────────────────────────────────────────
results = {}
for name, clf in [('Logistic Regression', estimators[0][1]),
                   ('Random Forest',       estimators[1][1]),
                   ('SVM',                 estimators[2][1]),
                   ('Soft Voting',         VotingClassifier(estimators=estimators, voting='soft')),
                   ('Stacking',            stacker)]:
    scores = cross_val_score(clf, X_train_s, y_train, cv=5)
    results[name] = (scores.mean(), scores.std())
    print(f"{name:25s}: {scores.mean():.4f} ± {scores.std():.4f}")
```
