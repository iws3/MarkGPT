# Linear Regression

## Fundamentals

Linear Regression is one of the most fundamental and widely used algorithms in machine learning. It models the relationship between one or more input features (independent variables) and a continuous target variable (dependent variable) by fitting a linear equation to the observed data. The algorithm assumes that there exists a linear relationship between inputs and outputs, and it aims to find the best-fit line (or hyperplane in multiple dimensions) that minimizes the prediction error. Linear regression serves as the foundation for understanding more complex regression techniques and is extensively used in economics, physics, engineering, and business analytics for forecasting and trend analysis.

## Key Concepts

- **Hypothesis**: $h(x) = \theta_0 + \theta_1 x$
- **Cost Function**: Mean Squared Error (MSE)
- **Optimization**: Closed-form solution or Gradient Descent
- **Assumptions**: Linearity, Independence, Homoscedasticity, Normality of errors

## Applications

- Stock price prediction
- Real estate valuation
- Sales forecasting
- Temperature prediction
- Risk assessment

## When to Use

Use linear regression when:
- You need a simple, interpretable model
- Relationship appears linear
- Speed of prediction is critical
- You have limited computational resources

## When NOT to Use

- Clear non-linear patterns in data
- Categorical relationships
- When interpretability is not important and accuracy is paramount

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Mathematical Foundations

Linear regression is fundamentally based on the principle of minimizing the sum of squared errors between predicted and actual values. The cost function, also known as the Mean Squared Error (MSE), is defined as J(θ) = 1/(2m) * Σ(h(x) - y)², where h(x) represents the hypothesis function and m is the number of training examples. This quadratic cost function creates a convex optimization landscape, ensuring that any local minimum found is also the global minimum. Understanding this mathematical foundation is crucial for implementing optimization algorithms like gradient descent effectively.

### Gradient Descent Optimization

Gradient descent is the iterative optimization algorithm used to find the parameters that minimize the cost function. The algorithm works by computing the partial derivatives of the cost function with respect to each parameter and moving in the direction of steepest descent. The learning rate (alpha) is a critical hyperparameter that controls the size of each step taken during optimization. A learning rate that is too small will result in slow convergence, while a learning rate that is too large may cause the algorithm to overshoot the optimal solution. Choosing an appropriate learning rate requires careful consideration and often involves experimentation with different values.

### Feature Scaling and Normalization

When features have vastly different scales, gradient descent can converge slowly and may struggle to find optimal parameters. Feature scaling techniques such as standardization (z-score normalization) and mean normalization help ensure that all features contribute equally to the optimization process. Standardization transforms features to have zero mean and unit variance, computed as (x - mean) / standard_deviation. Without proper feature scaling, features with larger numerical ranges can dominate the learningprocess, leading to a model that does not properly utilize all available information in the training data.

### Regularization and Model Complexity

Overfitting occurs when a linear regression model learns the noise in the training data rather than the underlying pattern, resulting in poor performance on new, unseen data. Regularization techniques add a penalty term to the cost function to discourage overly complex models. Ridge regression adds an L2 penalty proportional to the square of the coefficients, while Lasso regression adds an L1 penalty proportional to the absolute value of the coefficients. The regularization parameter (lambda) controls the strength of the penalty; higher values lead to smaller coefficients and simpler models, while lower values allow for more complex models. Cross-validation can be used to find the optimal regularization parameter that balances model complexity and generalization performance.

### Python Implementation and Scikit-learn

Python's scikit-learn library provides LinearRegression: `from sklearn.linear_model import LinearRegression; model = LinearRegression().fit(X_train, y_train); predictions = model.predict(X_test)`. The model object provides coefficients (β parameters) as `model.coef_` and intercept as `model.intercept_`. For large datasets, stochastic gradient descent via SGDRegressor provides memory-efficient training. The fit method internally handles centering and scaling. Key consideration: features should be scaled using StandardScaler before fitting ridge/lasso regression, as these regularization techniques are sensitive to feature magnitudes. Cross-validation with cross_val_score assesses model generalization. Visualization of predictions vs. actuals reveals systematic biases.

### Feature Engineering for Linear Regression

Linear relationships between raw features and targets are rare; feature engineering improves linear models significantly. Polynomial features expand input dimension: if x=[x1, x2], polynomial features of degree 2 create [1, x1, x2, x1^2, x1*x2, x2^2]. Logarithmic transformations handle exponential relationships. Interaction terms capture multiplicative effects: price might depend on size*location. Binning continuous variables creates categorical features, useful for capturing non-linear thresholds. Domain knowledge guides feature engineering: in real estate, features like square footage, number of bedrooms, and proximity to transit are meaningful. Feature scaling via standardization ensures coefficients are comparable in magnitude. One-hot encoding converts categorical variables to binary features, with care taken to avoid multicollinearity through dropping one category.

### Multicollinearity and Coefficient Interpretation

Multicollinearity occurs when predictor variables are highly correlated, making individual coefficients unstable and hard to interpret. High correlation between features inflates variance of coefficient estimates; small data changes cause large coefficient changes. Detection uses variance inflation factors (VIF); VIF > 10 indicates problematic multicollinearity. Solutions include removing correlated features, combining them via PCA, or using regularization (ridge/lasso). Ridge regression is particularly effective: it handles multicollinearity by distributing weight across correlated features. When multicollinearity exists, coefficient magnitudes are no longer reliable indicators of feature importance. This is critical in interpretability: a feature with small coefficient might be important but overshadowed by highly correlated features. Always check correlations between features via correlation matrices.

### Ridge vs Lasso Trade-offs

Ridge and lasso regularization differ in how they constrain coefficients. Ridge penalizes sum of squares: ||β||^2, keeping all features with smaller coefficients. Lasso penalizes sum of absolute values: ||β||_1, forcing some coefficients exactly to zero, performing automatic feature selection. Ridge is preferred when all features are likely relevant; lasso when many features are irrelevant. Elastic Net combines both, controlled by parameter α ∈ [0,1]. In practice, lasso often provides more interpretable models through feature selection, while ridge handles multicollinearity better. For categorical variables with many categories, lasso's sparsity is beneficial. The regularization strength λ is crucial: use cross-validation to find optimal λ. Very large λ shrinks all coefficients to zero; very small λ behaves like unregularized regression.

### Common Pitfalls and Debugging

Common issues in linear regression include: (1) Forgetting to scale features before regularized regression, leading to incorrect solutions where large-magnitude features are penalized more; (2) Not centering the data, causing intercept interpretation issues; (3) Ignoring regression assumptions (linearity, constant variance, normality of errors), leading to inefficient or biased estimates; (4) Using just R² for model evaluation, ignoring residual patterns; (5) Extrapolating beyond training data range, where linear relationships often break. Residual analysis via qq-plots and residual plots reveals assumption violations. Heteroscedasticity (non-constant variance) appears as fan-shaped residual patterns; log-transformations often help. Outliers visible in residual plots warrant investigation. Cross-validation helps detect overfitting: training error much lower than validation error signals regularization is needed. Always plot data before fitting to verify linearity assumptions.