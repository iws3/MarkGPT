# 📐 0_maths — Mathematics for Machine Learning

> **Why maths?** You can use ML libraries without understanding the maths, just as you can drive a car without understanding the engine. But when things break — and they will — knowing the maths tells you *exactly* what went wrong and *how* to fix it.

---

## Table of Contents

1. [Linear Algebra](#linear-algebra)
2. [Statistics & Probability](#statistics--probability)
3. [Calculus & Optimisation](#calculus--optimisation)
4. [How These Connect to ML](#how-these-connect-to-ml)

---

## 🔢 Linear Algebra

Linear algebra is the language of data. Every dataset is a matrix. Every transformation is a matrix operation.

```
A single data point:  x = [age, salary, years_experience]
                          = [25,  50000,  3]
                          → a VECTOR (1D array of numbers)

A whole dataset:      X = [[25, 50000,  3],
                            [32, 75000,  8],
                            [28, 60000,  5],
                            ...]
                          → a MATRIX (2D array)
                          Shape: (n_samples × n_features)
```

**Key operations you must know:**

```
Matrix Multiplication:  C = A × B
  If A is (m×k) and B is (k×n), then C is (m×n)
  C[i][j] = dot product of row i of A with column j of B

Dot Product:  a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
  → This is how a linear model makes a prediction:
     ŷ = θ₀ + θ₁x₁ + θ₂x₂ = [1, x₁, x₂] · [θ₀, θ₁, θ₂]

Transpose:  Aᵀ  (rows become columns)
  Used constantly in the closed-form solution for linear regression:
     θ = (XᵀX)⁻¹ Xᵀy

Matrix Inverse:  A⁻¹  (like dividing by a matrix)
  Only square matrices can be inverted
  AA⁻¹ = I  (identity matrix — like the number 1 for matrices)
```

**Sub-folders:** See [`1_linear algebra/`](1_linear%20algebra/) for full lessons.

---

## 📊 Statistics & Probability

Statistics lets us understand our data before we model it. Probability lets us reason about uncertainty in predictions.

```
Mean (μ):         Average value
                  μ = (x₁ + x₂ + ... + xₙ) / n

Variance (σ²):    How spread out values are around the mean
                  σ² = (1/n) Σ(xᵢ - μ)²

Std Dev (σ):      Square root of variance — same units as data
                  Useful: "68% of data falls within μ ± σ" (bell curve)

Correlation (r):  Linear relationship between two variables
                  r = 1  → perfect positive correlation
                  r = 0  → no linear relationship
                  r = -1 → perfect negative correlation
```

**Probability:**
```
P(A)        → probability of event A  (between 0 and 1)
P(A|B)      → probability of A GIVEN B has occurred
Bayes:      P(A|B) = P(B|A) × P(A) / P(B)
              ↑ This is the foundation of Naive Bayes and Bayesian ML!
```

**Sub-folders:** See [`2_statistics/`](2_statistics/) for full lessons.

---

## 📈 Calculus & Optimisation

Calculus tells us how to minimise the error of our model — which is the entire point of training.

```
Derivative f'(x):  Rate of change of f at point x
                   "slope" of the function at that point
                   If f'(x) > 0  → function is increasing
                   If f'(x) < 0  → function is decreasing
                   If f'(x) = 0  → local minimum or maximum (!)

Gradient ∇f:       The derivative for functions with multiple inputs
                   Points in the direction of steepest INCREASE
                   → Gradient Descent moves in the OPPOSITE direction

Chain Rule:        d/dx[f(g(x))] = f'(g(x)) × g'(x)
                   Essential for backpropagation in neural networks!
```

**Gradient Descent — the engine of ML training:**
```
Repeat until convergence:
  θ ← θ - α × ∇J(θ)

Where:
  θ     = model parameters (what we're learning)
  α     = learning rate (how big each step is)
  ∇J(θ) = gradient of the cost function (which direction is "uphill"?)
  → We subtract because we want to go DOWNHILL (minimise cost)
```

**Sub-folders:** See [`3_calculus/`](3_calculus/) for full lessons.

---

## 🔗 How These Connect to ML

```
┌────────────────┬─────────────────────────────────────────────────┐
│ Maths Topic    │ Where It Shows Up in ML                         │
├────────────────┼─────────────────────────────────────────────────┤
│ Vectors        │ Data points, weight vectors, embeddings          │
│ Matrices       │ Datasets, weight matrices in neural nets         │
│ Dot Product    │ Every linear model's prediction step             │
│ Eigenvalues    │ PCA (dimensionality reduction)                   │
├────────────────┼─────────────────────────────────────────────────┤
│ Mean/Variance  │ Feature scaling, Gaussian distributions          │
│ Bayes Theorem  │ Naive Bayes, Bayesian ML, probabilistic models   │
│ Distributions  │ Output probabilities, data assumptions           │
│ Hypothesis test│ A/B testing model performance                    │
├────────────────┼─────────────────────────────────────────────────┤
│ Derivatives    │ Gradient descent, finding optimal parameters     │
│ Chain Rule     │ Backpropagation in neural networks               │
│ Partial deriv. │ Multi-variable optimisation                      │
│ Convexity      │ Why gradient descent finds the global minimum    │
└────────────────┴─────────────────────────────────────────────────┘
```

> 💡 **Tip for beginners**: Don't try to master all the maths before touching ML code. Learn the maths *alongside* the algorithms — you'll understand *why* you need it immediately, and that makes it stick much faster.
