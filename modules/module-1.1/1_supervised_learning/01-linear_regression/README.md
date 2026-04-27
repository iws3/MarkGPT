# 📈 Lesson 01 — Linear Regression

> **Core Idea**: Find the best-fit straight line through your data so you can predict a number for any new input.

---

## 📋 Table of Contents

1. [The Big Picture — What Problem Are We Solving?](#1-the-big-picture)
2. [Intuition Before Equations](#2-intuition-before-equations)
3. [The Hypothesis Function](#3-the-hypothesis-function)
4. [The Cost Function (How Wrong Are We?)](#4-the-cost-function)
5. [Gradient Descent — How We Learn](#5-gradient-descent)
6. [The Closed-Form Solution](#6-the-closed-form-solution)
7. [Multiple Linear Regression](#7-multiple-linear-regression)
8. [Feature Scaling — Why It Matters](#8-feature-scaling)
9. [Evaluating Your Model](#9-evaluating-your-model)
10. [Regularisation — Ridge & Lasso](#10-regularisation)
11. [Assumptions of Linear Regression](#11-assumptions)
12. [Python Implementation](#12-python-implementation)
13. [Visual Summary](#13-visual-summary)
14. [Common Mistakes](#14-common-mistakes)
15. [When to Use (and When Not To)](#15-when-to-use)

---

## 1. The Big Picture

**The problem**: You have data where you know the input (e.g., house size in m²) and the output (e.g., house price in dollars). You want to predict the price of a *new* house you haven't seen before.

**The solution**: Draw the best possible straight line through your data. Once you have that line, predicting for a new house is as simple as plugging its size into the line equation.

```
House Price ($)
     │
 500k│         ●
     │       ●   ←── actual data points
 400k│     ●       ●
     │   ●  ●
 300k│ ●       ←── best-fit line goes through here
     │
 200k│────────────────────────────────────────► House Size (m²)
     50   100   150   200   250   300
```

This is the essence of linear regression: **find the line, use the line**.

---

## 2. Intuition Before Equations

Imagine you're trying to balance a ruler so it sits as close as possible to all the dots on a scatter plot. You can tilt it (change slope) or slide it up and down (change intercept). The goal is to minimise the total "distance" between the ruler and all the dots.

That's it. That's linear regression.

The two questions we need to answer are:
1. **What does "best" mean?** (How do we measure how good our line is?)
2. **How do we find the best line?** (How do we actually search for it?)

We'll answer both below.

---

## 3. The Hypothesis Function

The straight line we're fitting is described mathematically by:

```
ŷ = θ₀ + θ₁ × x

Where:
  ŷ      = predicted value (what our model says the output should be)
  x      = input feature (e.g., house size)
  θ₀     = intercept (the y-value when x = 0)
  θ₁     = slope (how much ŷ changes when x increases by 1)
  θ      = "theta" — parameters, or "weights" — what we're trying to learn
```

For example, if we learn `θ₀ = 50,000` and `θ₁ = 200`, our model says:

```
House price = 50,000 + 200 × (house size)

So for a 150 m² house:
  price = 50,000 + 200 × 150 = 50,000 + 30,000 = $80,000
```

For **multiple features** (more than one input), we simply extend this:

```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ

→ Instead of a line, we now have a HYPERPLANE in n-dimensional space
→ Same idea, just more variables
```

--- 

## 4. The Cost Function

### Why do we need a cost function?

We need a way to measure "how bad" any particular line is. Then we can search for the line that minimises this "badness".

### Mean Squared Error (MSE)

For each data point, we calculate the **residual**: the difference between what the model predicted and what the actual answer was.

```
Residual for point i:  eᵢ = yᵢ - ŷᵢ   (actual minus predicted)

  Actual price:   $300k
  Predicted price: $270k
  Residual:        $30k  ← we were off by $30k
```

We **square** each residual (to make all errors positive and penalise large errors more heavily) and take the average:

```
MSE  = (1/m) × Σ(yᵢ - ŷᵢ)²
J(θ) = (1/2m) × Σ(h_θ(xᵢ) - yᵢ)²

Where:
  m        = number of training examples
  yᵢ       = actual output for example i
  h_θ(xᵢ) = model's prediction for example i
  The ½ is added for mathematical convenience (cancels when we differentiate)
```

**Visual: What we're minimising**

```
Price error
    │
    │  ●              residual = vertical gap
    │  ↕ ←── we want to minimise these gaps
    ├──●─────────── predicted line
    │        ↕
    │        ●
    │              ●
    │────────────────────────────────► Size
```

### Why square the errors?

There are three reasons: (1) Negative and positive errors cancel if you just sum them — squaring prevents that. (2) Larger errors are penalised much more than small errors — a prediction that is off by 10 gets penalised 100x more than one off by 1. (3) The MSE function is smooth and convex, which makes it easy to optimise with gradient descent.

---

## 5. Gradient Descent

### The Idea

Imagine you're blindfolded on a hilly landscape and you need to find the lowest valley. Your strategy: feel which way is downhill, take a small step in that direction, repeat.

That's gradient descent. The "landscape" is our cost function, and we're looking for the lowest point (minimum MSE).

```
Cost J(θ)
    │
    │   ╲              ╱
    │    ╲            ╱
    │     ╲          ╱
    │      ╲        ╱
    │       ╲      ╱
    │        ╲    ╱
    │         ╲  ╱
    │          ╲╱  ←── minimum! This is where our best θ lives
    │
    └────────────────────────────────────────────► θ₁ (slope)
       The algorithm rolls "downhill" by following the gradient
```

### The Update Rule

```
Repeat until convergence:
  θⱼ := θⱼ - α × (∂/∂θⱼ) J(θ)

For our linear regression:
  θ₀ := θ₀ - α × (1/m) × Σ(h_θ(xᵢ) - yᵢ)
  θ₁ := θ₁ - α × (1/m) × Σ(h_θ(xᵢ) - yᵢ) × xᵢ

Where:
  α (alpha) = learning rate — how big each step is
  ∂J/∂θⱼ   = the gradient (which direction is "uphill"?)
  We subtract because we want to go DOWNHILL
```

### The Learning Rate α — Critical Hyperparameter

```
Learning rate too SMALL:          Learning rate too LARGE:
  ●                                   ●         ●
    ●                               ↗   ↘     ↗   (bounces, never converges)
      ●                                   ●
        ●   (converges, but slowly)
          ●
            ●                       Just right α:
              ●─── minimum              ●
                                          ●
                                            ●
                                              ●── minimum
```

**Rule of thumb**: Try α = 0.001, 0.01, 0.1 and plot the cost vs iteration curve. If the cost goes down smoothly and levels off, your α is good. If it oscillates or increases, your α is too large.

---

## 6. The Closed-Form Solution

For linear regression specifically, there's a mathematical shortcut — you don't have to iterate at all. The optimal θ can be computed directly:

```
θ = (XᵀX)⁻¹ Xᵀy

Where:
  X = design matrix of shape (m × n+1)  [features + bias column of 1s]
  y = target vector of shape (m × 1)
  Xᵀ = transpose of X
  (XᵀX)⁻¹ = matrix inverse

In Python:
  theta = np.linalg.inv(X.T @ X) @ X.T @ y
```

**When to use each approach:**

```
┌─────────────────────┬────────────────────┬─────────────────────┐
│                     │ Gradient Descent   │ Normal Equation      │
├─────────────────────┼────────────────────┼─────────────────────┤
│ Need to tune α?     │ Yes                │ No                  │
│ Iterations needed?  │ Yes (many)         │ No (one shot)       │
│ Scales to big data? │ ✅ Yes (use SGD)   │ ❌ No (slow for n>10k)│
│ Works when n large? │ ✅ Yes             │ ❌ (XᵀX becomes huge)│
│ Always works?       │ Yes (with right α) │ Needs XᵀX invertible│
└─────────────────────┴────────────────────┴─────────────────────┘
```

**Use the Normal Equation** when you have fewer than ~10,000 features and a dataset that fits in memory. **Use Gradient Descent** for large datasets, neural networks, and any other model where a closed form doesn't exist.

---

## 7. Multiple Linear Regression

When you have more than one input feature, we simply add more terms. Each feature gets its own weight θⱼ.

```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + θ₃x₃ + ... + θₙxₙ

Example: predicting house price from multiple features
  ŷ = 10,000 + 200×(size) + 5000×(bedrooms) - 1000×(age)
              ↑              ↑                   ↑
       θ₁ for size    θ₂ for bedrooms    θ₃ for age
```

The mathematics is exactly the same as simple linear regression — just more dimensions. Gradient descent and the Normal Equation both extend naturally.

---

## 8. Feature Scaling

### The Problem Without Scaling

Imagine your features are house size (range: 50–500 m²) and number of bedrooms (range: 1–10). The cost function landscape becomes very elongated:

```
                   θ (bedrooms)
Without scaling:      │
                      │        ╱─── Very elongated "bowl"
         ╱────────────│───────╱     Gradient descent zigzags inefficiently
        │             │      │
         ╲────────────│───────╲
                      └──────────────────────────► θ (size)
                         Takes MANY iterations

With scaling (both features 0–1):
         ╭─────────╮
        ╱           ╲       Nice circular bowl
       │      ★      │      Gradient descent goes straight to minimum
        ╲           ╱
         ╰─────────╯
                            Takes FAR FEWER iterations
```

### How to Scale

```python
# Method 1: Min-Max Scaling (rescales to [0, 1])
x_scaled = (x - x.min()) / (x.max() - x.min())

# Method 2: Standardisation / Z-score (mean=0, std=1) ← most common
x_scaled = (x - x.mean()) / x.std()

# ⚠️ CRITICAL RULE: Fit the scaler ONLY on training data!
#   Then use THOSE parameters to scale validation and test data.
#   Never compute mean/std from validation or test data.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit AND transform
X_test_scaled  = scaler.transform(X_test)        # transform ONLY (no fit!)
```

---

## 9. Evaluating Your Model

You need more than one number to understand how well your model is working. Always look at at least these three:

```
MSE  (Mean Squared Error)   = (1/m) × Σ(y - ŷ)²
RMSE (Root MSE)             = √MSE         ← same units as y, more interpretable
MAE  (Mean Absolute Error)  = (1/m) × Σ|y - ŷ|  ← less sensitive to outliers
R²   (R-squared)            = 1 - SS_res/SS_tot   ← between 0 and 1 (1 = perfect)

R² interpretation:
  R² = 0.0  → Model does no better than predicting the mean every time
  R² = 0.7  → Model explains 70% of the variance in y
  R² = 1.0  → Perfect fit (usually means overfitting on training data!)
```

**Always plot residuals** — a residual plot reveals patterns your metrics can't:

```
Good residuals:             Bad residuals (pattern remains):
    ●  ●                        ●                      ←── curving up
  ●      ●                   ●     ●                       means linear
──────────────── 0 line    ──────────────── 0 line         assumption
  ●      ●                ●          ●                    violated!
    ●  ●                           ●
                                              ●
Random scatter ✅           Systematic curve ❌
```

---

## 10. Regularisation — Ridge & Lasso

### The Overfitting Problem

A high-degree polynomial can fit any training data perfectly — but it will be terrible on new data because it memorised the noise:

```
Degree 1 (underfit):     Degree 10 (overfit):      Degree 3 (just right):
   /                         ╱╲    ╱╲   ╱              /
  /    ●  ●                 ╱  ╲  ╱  ╲ ╱           ╱─╱  ●
 / ●                       ╱    ╲╱    ╱       ●──╱   ●
/ ●                       ╱           ╱       ●
Too simple               Memorises noise      Generalises well
```

### Ridge Regression (L2)

Add a penalty to the cost function that discourages large coefficients:

```
J(θ) = MSE + λ × Σθⱼ²
               ↑
        Ridge penalty: sum of squared weights
        λ controls how strong the penalty is:
          λ = 0  → regular linear regression (no penalty)
          λ large → all θ shrink toward 0 (simpler model)
```

### Lasso Regression (L1)

Similar idea, but uses absolute values instead of squares:

```
J(θ) = MSE + λ × Σ|θⱼ|
               ↑
        Lasso penalty: sum of absolute weights
        Key difference: Lasso can shrink weights ALL THE WAY to 0
        → Automatic feature selection! Irrelevant features get θ = 0
```

```
┌──────────────────┬──────────────────────┬──────────────────────┐
│                  │ Ridge (L2)           │ Lasso (L1)           │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Penalty          │ λ × Σθⱼ²             │ λ × Σ|θⱼ|           │
│ Effect           │ Shrinks all θ small  │ Sets some θ = 0 exactly│
│ Feature select?  │ No (keeps all feats) │ Yes (sparse model)   │
│ Best when        │ All features relevant│ Many irrelevant feats │
│ Multicollinearity│ Handles well         │ Picks one, drops rest│
└──────────────────┴──────────────────────┴──────────────────────┘

Elastic Net = Ridge + Lasso (controlled by α parameter between 0 and 1)
```

---

## 11. Assumptions

Linear regression makes 4 key assumptions. When violated, the model can still work but becomes less reliable:

```
Assumption 1 — LINEARITY
  The relationship between X and y is actually linear.
  Check: scatter plot of y vs each feature; residual vs fitted plot

Assumption 2 — INDEPENDENCE
  Each data point is independent of the others.
  Violated by: time series data without accounting for autocorrelation

Assumption 3 — HOMOSCEDASTICITY (constant variance)
  The spread of residuals is the same across all predicted values.
  Violated when: errors are larger for large predictions (fan shape)
  Check: scale-location plot

Assumption 4 — NORMALITY OF ERRORS
  Residuals are roughly normally distributed.
  Check: Q-Q plot (should be close to diagonal line)
  (Less important for large datasets due to Central Limit Theorem)
```

---

## 12. Python Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ─── 1. Load data ──────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # house price (in $100,000s)

# ─── 2. Split ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # always set random_state for reproducibility!
)

# ─── 3. Scale features ─────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # fit on train only!
X_test_s  = scaler.transform(X_test)       # apply same transformation to test

# ─── 4. Train ──────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_s, y_train)

# ─── 5. Predict & evaluate ─────────────────────────────────────────────────
y_pred = model.predict(X_test_s)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")        # e.g. RMSE: 0.726
print(f"R²:   {r2:.3f}")          # e.g. R²:   0.612

# ─── 6. Inspect coefficients ───────────────────────────────────────────────
coef_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)
print(coef_df)  # positive coef = feature increases price; negative = decreases

# ─── 7. Plot residuals ─────────────────────────────────────────────────────
residuals = y_test - y_pred
plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot — should be random scatter around 0')
plt.show()

# ─── 8. With regularisation ────────────────────────────────────────────────
ridge = Ridge(alpha=1.0)   # alpha here is λ in our formula
ridge.fit(X_train_s, y_train)
print(f"Ridge R²: {r2_score(y_test, ridge.predict(X_test_s)):.3f}")

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_s, y_train)
print(f"Lasso R²: {r2_score(y_test, lasso.predict(X_test_s)):.3f}")
print(f"Lasso kept {np.sum(lasso.coef_ != 0)} features")  # feature selection!
```

---

## 13. Visual Summary

```
╔══════════════════════════════════════════════════════════════════╗
║                   LINEAR REGRESSION — OVERVIEW                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  INPUT            MODEL              OUTPUT                      ║
║  Features X  ──►  ŷ = θ₀ + θ₁x₁  ──►  Predicted number ŷ       ║
║  (numbers)         + θ₂x₂ + ...        (house price, temp...)   ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  TRAINING:  Find θ that minimises  J(θ) = (1/2m)Σ(ŷ - y)²      ║
║                                                                  ║
║  Method 1 — Gradient Descent (iterative):                        ║
║    θ ← θ - α∇J(θ)    ← repeat until convergence                ║
║                                                                  ║
║  Method 2 — Normal Equation (direct):                            ║
║    θ = (XᵀX)⁻¹Xᵀy    ← one-shot, works for small datasets      ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  REGULARISATION (to prevent overfitting):                        ║
║    Ridge:  J(θ) + λΣθⱼ²    ← shrinks all weights               ║
║    Lasso:  J(θ) + λΣ|θⱼ|   ← zeros out irrelevant weights      ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  METRICS:  RMSE (interpretable error) + R² (explained variance) ║
║  ALWAYS:   Plot residuals to check assumptions                   ║
║  ALWAYS:   Scale features before training                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 14. Common Mistakes

**Mistake 1**: Not scaling features before gradient descent. Result: slow convergence or divergence. Fix: always use `StandardScaler` before distance- or gradient-based training.

**Mistake 2**: Scaling the test set with its own statistics. Result: data leakage — your test set evaluation is now unrealistic. Fix: fit the scaler on training data only, then `transform()` (not `fit_transform()`) on test data.

**Mistake 3**: Reporting only R² without plotting residuals. A high R² can hide systematic errors that residual plots make obvious.

**Mistake 4**: Interpreting coefficients without scaling. Raw coefficients are not comparable across features if those features have very different units (e.g., temperature in °C vs population in millions). Scale first.

**Mistake 5**: Extrapolating far beyond your training data range. Linear regression assumes linearity holds everywhere — but a relationship that is linear for house sizes 50–300 m² may not be linear for a 2,000 m² mansion.

---

## 15. When to Use (and When Not To)

**Use linear regression when:**
- Your target variable is continuous (a number)
- You suspect a roughly linear relationship between features and target
- Interpretability matters — stakeholders want to understand the model
- You want a fast baseline to compare other models against
- Computational resources are limited

**Don't use it when:**
- The relationship is clearly non-linear (try polynomial features, or a tree model)
- Your target is a category, not a number (use logistic regression instead)
- Features are images, audio, or raw text (use deep learning)
- You have very few data points relative to the number of features (use regularisation at minimum)

---

> 📂 **Next step**: Work through the [exercises](exercises.md) to practise what you've learned. Start with Exercise 1 (simple linear regression) and work your way up to Exercise 10 (real-world project).
>
> 📌 **Coming up next**: [02 — Logistic Regression](../02-logistic_regression/README.md) — the same ideas, adapted for classification problems.
