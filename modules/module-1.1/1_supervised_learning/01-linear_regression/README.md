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