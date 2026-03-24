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