# 📝 Linear Regression — Exercises

> **How to use these exercises**: Read each prompt carefully, write the code yourself (don't copy-paste from the README), run it, and look at the output critically. The goal isn't just to get the code running — it's to understand *what* the numbers mean and *why* the model behaves the way it does. After each exercise, write 2–3 sentences in a comment block explaining what you observed.

---

## Exercise 1 — Simple Linear Regression from Scratch

**Goal**: Predict house price from square footage using only NumPy (no scikit-learn). This forces you to understand what's happening under the hood.

**Instructions:**
1. Generate a synthetic dataset: `X = np.linspace(50, 300, 100)` (sizes in m²) and `y = 500*X + 50000 + np.random.normal(0, 20000, 100)` (prices with noise).
2. Implement the Normal Equation: `theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y` where `X_b` is X with a column of 1s prepended.
3. Plot the data points as a scatter plot and overlay your fitted line.
4. Print the learned intercept (θ₀) and slope (θ₁) and interpret them in plain English — what does each value mean in the context of house prices?
5. Calculate and print RMSE for your model.

**Expected observation**: Your slope should be close to 500 (the true value) and your intercept close to 50,000, with some noise because the data has random variation.

**Starter code structure:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate data
np.random.seed(42)
X = np.linspace(50, 300, 100)
y = 500 * X + 50000 + np.random.normal(0, 20000, 100)

# Step 2: Add bias column (column of 1s) to X
X_b = np.c_[np.ones(len(X)), X]   # shape: (100, 2)

# Step 3: Normal equation — fill this in!
theta = ...  # (X_bᵀ X_b)⁻¹ X_bᵀ y

# Step 4: Print parameters
print(f"Intercept (θ₀): {theta[0]:.2f}")
print(f"Slope     (θ₁): {theta[1]:.2f}")

# Step 5: Plot
x_line = np.array([50, 300])
y_line = theta[0] + theta[1] * x_line
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(x_line, y_line, 'r-', linewidth=2, label='Fitted line')
plt.xlabel('House Size (m²)')
plt.ylabel('Price ($)')
plt.legend()
plt.title('Exercise 1: Simple Linear Regression from Scratch')
plt.show()

# Step 6: RMSE
y_pred = X_b @ theta
rmse = np.sqrt(np.mean((y - y_pred)**2))
print(f"RMSE: ${rmse:,.0f}")
```

---

## Exercise 2 — Gradient Descent from Scratch

**Goal**: Implement gradient descent yourself to understand how the algorithm actually works. Compare the result to the Normal Equation from Exercise 1.

**Instructions:**
1. Use the same dataset from Exercise 1.
2. Initialise θ = [0, 0] (both parameters start at zero).
3. Implement the gradient descent loop: for 1,000 iterations, compute the gradient and update θ.
4. Track the cost (MSE) at each iteration and plot it vs the number of iterations.
5. Experiment with three different learning rates: α = 0.0001, α = 0.001, α = 0.01. Plot all three cost curves on the same graph. What do you observe?
6. Compare your final θ to the Normal Equation result. They should be very close.

**Key questions to answer in comments:**
- What happens to the cost curve when α is too large?
- How many iterations does it take for each α to approximately converge?
- Why do both methods (gradient descent and Normal Equation) give the same θ?

**Starter code structure:**
```python
def gradient_descent(X_b, y, alpha, n_iterations):
    m = len(y)
    theta = np.zeros(2)   # initialise at zero
    cost_history = []

    for iteration in range(n_iterations):
        # 1. Compute predictions
        y_pred = X_b @ theta

        # 2. Compute error
        error = y_pred - y

        # 3. Compute gradient (partial derivatives)
        gradients = (1/m) * X_b.T @ error   # shape: (2,)

        # 4. Update parameters
        theta = theta - alpha * gradients

        # 5. Record cost (MSE)
        cost = (1/(2*m)) * np.sum(error**2)
        cost_history.append(cost)

    return theta, cost_history

# Run with different learning rates
for alpha in [0.0001, 0.001, 0.01]:
    theta_gd, costs = gradient_descent(X_b, y, alpha, n_iterations=1000)
    plt.plot(costs, label=f'α = {alpha}')
    print(f"α={alpha}: θ = {theta_gd}")

plt.xlabel('Iteration')
plt.ylabel('Cost (MSE/2)')
plt.yscale('log')   # log scale makes differences clearer
plt.legend()
plt.title('Exercise 2: Cost vs Iterations for Different Learning Rates')
plt.show()
```

---

## Exercise 3 — Multiple Linear Regression with Scikit-learn

**Goal**: Build a model using multiple features from the California Housing dataset and interpret the coefficients.

**Instructions:**
1. Load the California Housing dataset (`from sklearn.datasets import fetch_california_housing`).
2. Split into train (80%) and test (20%) sets with `random_state=42`.
3. Scale all features using `StandardScaler`.
4. Train a `LinearRegression` model.
5. Print a sorted table of feature names and their coefficients — which feature has the highest positive influence on price? Which the highest negative?
6. Calculate RMSE and R² on the test set.
7. Create a scatter plot of actual vs predicted values. A perfect model would be a diagonal line from bottom-left to top-right — how close is yours?

**Expected interpretation**: After scaling, you can directly compare coefficients across features. A coefficient of +0.5 means "a 1 standard deviation increase in this feature is associated with a $50,000 increase in price" (since house prices are in $100k units).

---

## Exercise 4 — Feature Scaling Impact Experiment

**Goal**: Prove empirically that feature scaling matters for gradient descent by running a controlled experiment.

**Instructions:**
1. Create a dataset with two features on very different scales, e.g., `x1 = age (range 18–65)` and `x2 = salary (range 20,000–200,000)`.
2. Train gradient descent (your implementation from Exercise 2) on the **unscaled** features. Log how many iterations it takes to converge (cost stops decreasing meaningfully).
3. Train the same gradient descent on **scaled** features (apply StandardScaler first).
4. Plot both cost curves on the same graph.
5. Print: the final cost, number of iterations to convergence, and final θ for both versions.

**The "aha" moment**: With unscaled data, the salary feature dominates the gradient because its values are ~1000× larger. This makes the cost landscape very elongated, and gradient descent must zigzag to find the minimum. With scaled data, both features are on equal footing and gradient descent goes straight downhill.

---

## Exercise 5 — Polynomial Regression & Overfitting

**Goal**: Visualise the bias-variance tradeoff by fitting polynomials of increasing degree to a noisy dataset.

**Instructions:**
1. Generate: `x = np.linspace(0, 1, 30)` and `y = np.sin(2*np.pi*x) + np.random.normal(0, 0.3, 30)`.
2. Fit polynomial regression for degrees 1, 3, 5, 9, and 15 using `sklearn.preprocessing.PolynomialFeatures`.
3. Plot each fitted curve over the data.
4. For each degree, record training MSE and test MSE (use 70/30 split).
5. Plot training MSE and test MSE vs degree on the same graph.

**Expected pattern**: Training MSE always decreases as degree increases. Test MSE decreases initially (underfitting → good fit) then increases sharply (overfitting). The ideal degree is where test MSE is minimised.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

degrees = [1, 3, 5, 9, 15]
train_mses, test_mses = [], []

for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ])
    model.fit(x_train.reshape(-1,1), y_train)
    train_mses.append(mean_squared_error(y_train, model.predict(x_train.reshape(-1,1))))
    test_mses.append(mean_squared_error(y_test, model.predict(x_test.reshape(-1,1))))
```

---

## Exercise 6 — Residual Analysis

**Goal**: Learn to diagnose whether your linear regression assumptions are satisfied by analysing the residuals.

**Instructions:** After fitting a linear regression model on California Housing data:
1. **Residual vs Fitted plot**: Plot `(y_pred, residuals)`. Look for patterns — a flat cloud around 0 is ideal; any curve means a linear assumption is violated.
2. **Q-Q Plot**: Use `scipy.stats.probplot`. Points should lie roughly on the diagonal line — deviations indicate non-normal errors.
3. **Scale-Location plot**: Plot `(y_pred, sqrt(|standardised residuals|))`. A horizontal line indicates homoscedasticity (constant variance). A slope indicates heteroscedasticity.
4. **Residuals vs each feature**: Sometimes a relationship with a specific feature reveals which assumption is most violated.

Write a short paragraph (in comments) diagnosing any violations you find and suggesting what to do about them (e.g., log-transform the target, add polynomial features, remove outliers).

---

## Exercise 7 — Ridge and Lasso Regularisation

**Goal**: Understand how regularisation strength (λ) affects coefficients and model performance.

**Instructions:**
1. Generate a high-dimensional dataset with 100 features but only 80 samples: `X, y = make_regression(n_samples=80, n_features=100, noise=10, random_state=42)`. With more features than samples, unregularised linear regression will overfit badly.
2. Train: unregularised `LinearRegression`, `Ridge` with α=0.1, 1.0, 10.0, and `Lasso` with α=0.1, 1.0, 10.0.
3. For each, record training R² and cross-validation R² (5-fold).
4. Plot how the Ridge and Lasso coefficients change as λ increases (the "regularisation path").
5. For Lasso, count how many coefficients become exactly zero at each λ value.

**Question to answer**: Which value of λ gives the best cross-validated R²? What is the interpretation of a coefficient being exactly zero in a Lasso model?

---

## Exercise 8 — Cross-Validation Deep Dive

**Goal**: Understand why cross-validation gives a more reliable performance estimate than a single train-test split.

**Instructions:**
1. Use the California Housing dataset.
2. Run 50 different 80/20 random train-test splits and record the R² on the test set each time.
3. Plot a histogram of these 50 R² values — notice the variation! Which split was "lucky" and which was "unlucky"?
4. Now run 5-fold and 10-fold cross-validation using `cross_val_score`.
5. Compare: single-split R² range vs 5-fold mean±std vs 10-fold mean±std.

**The key insight**: A single train-test split is just one "sample" from the space of possible splits. It can be misleading. Cross-validation averages over many splits, giving a much more reliable estimate.

```python
from sklearn.model_selection import cross_val_score

# 50 random splits
single_split_scores = []
for i in range(50):
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=i)
    model = LinearRegression().fit(X_tr, y_tr)
    single_split_scores.append(r2_score(y_te, model.predict(X_te)))

# Cross-validation
cv5_scores  = cross_val_score(LinearRegression(), X_scaled, y, cv=5,  scoring='r2')
cv10_scores = cross_val_score(LinearRegression(), X_scaled, y, cv=10, scoring='r2')

print(f"Single split: {np.mean(single_split_scores):.3f} ± {np.std(single_split_scores):.3f}")
print(f"5-fold CV:    {cv5_scores.mean():.3f} ± {cv5_scores.std():.3f}")
print(f"10-fold CV:   {cv10_scores.mean():.3f} ± {cv10_scores.std():.3f}")
```

---

## Exercise 9 — Multicollinearity Detection and Treatment

**Goal**: Diagnose multicollinearity (highly correlated features) and apply the appropriate fix.

**Instructions:**
1. Create a dataset where features are artificially correlated: take California Housing features and add a new feature that is `age + noise` — almost identical to an existing feature.
2. Train a linear regression model and look at the coefficients. You'll notice they are unstable — very large positive and negative values that don't make physical sense.
3. Calculate the Variance Inflation Factor (VIF) for each feature. VIF > 10 is a red flag.
4. Try three remedies: (a) drop one of the correlated features, (b) Ridge regression with α=1.0, (c) PCA to combine correlated features.
5. For each remedy, report the new coefficients and the cross-validated R².

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_s, i) 
                   for i in range(X_train_s.shape[1])]
print(vif_data.sort_values('VIF', ascending=False))
# VIF > 10 → problematic multicollinearity
```

---

## Exercise 10 — End-to-End Real World Project

**Goal**: Build a complete, production-quality linear regression pipeline on a real dataset. This exercise ties together everything from the module.

**Dataset**: Download the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) from Kaggle, or use the built-in California Housing dataset as a simpler alternative.

**Deliverables** (write each as a clearly commented section of a Jupyter notebook):

**Section A — Exploratory Data Analysis**
- Distribution of the target variable (house price) — is it skewed? Consider log-transforming it.
- Correlation heatmap of all numerical features with the target.
- Identify the top 5 features most correlated with price.
- Check for missing values and decide: drop the rows, drop the column, or impute?

**Section B — Preprocessing Pipeline**
- Build a `sklearn.pipeline.Pipeline` with `SimpleImputer` → `StandardScaler` → `LinearRegression`.
- Using pipelines ensures you never accidentally fit the scaler on test data.

**Section C — Model Training and Selection**
- Train: plain `LinearRegression`, `Ridge` with grid-searched α, `Lasso` with grid-searched α.
- Use `GridSearchCV` with 5-fold cross-validation to find the best α for Ridge and Lasso.
- Report the best α value and explain why that value makes sense given your dataset.

**Section D — Evaluation and Interpretation**
- Report final test set RMSE and R² for the best model.
- Plot actual vs predicted prices — annotate a few large outliers.
- Plot the residuals. Are there any patterns? What might cause them?
- List the top 5 most influential features and explain their coefficients in plain English.

**Section E — Reflection**
Write a short paragraph (5–8 sentences) answering: What would you try next to improve this model? What are the main limitations of linear regression for this dataset? Would a non-linear model help, and why?

> ⭐ **Bonus challenge**: Apply the model to predict prices for 5 houses you make up yourself. Are the predictions sensible? What happens when you extrapolate to a 10-bedroom mansion (out of training data range)?
