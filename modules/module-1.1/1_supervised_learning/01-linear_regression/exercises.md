# Linear Regression - Exercises

## Exercise 1: Simple Linear Regression with Scikit-learn
Create a simple linear regression model to predict house prices based on square footage. Load the Boston Housing dataset, train the model, and evaluate it using R² score and RMSE.

## Exercise 2: Multiple Linear Regression
Build a multiple linear regression model using at least 5 features from the Boston Housing dataset. Compare the performance with Exercise 1 and discuss the impact of additional features.

## Exercise 3: Gradient Descent Implementation
Implement linear regression from scratch using gradient descent (not scikit-learn). Visualize the cost function over iterations and compare the results with scikit-learn's implementation.

## Exercise 4: Feature Scaling Impact
Train two linear regression models on the same data: one with scaled features and one without. Measure the difference in convergence speed when using gradient descent and explain why scaling matters.

## Exercise 5: Polynomial Regression
Extend linear regression to polynomial regression (degree 2 and 3) for a non-linear dataset. Plot the fitted curves and discuss overfitting vs underfitting trade-offs.

## Exercise 6: Residual Analysis
Fit a linear regression model and analyze the residuals. Create visualizations for:
- Residual plot (residuals vs predicted values)
- Q-Q plot (normality check)
- Scale-location plot

## Exercise 7: Cross-Validation
Implement k-fold cross-validation (k=5) on a linear regression model. Compare mean CV score with a single train-test split and explain the benefits of cross-validation.

## Exercise 8: Feature Importance via Coefficients
Train a linear regression model on a dataset with multiple features (normalize the features first). Rank the features by their coefficients and visualize them in a bar plot.

## Exercise 9: Handling Multicollinearity
Create a dataset with highly correlated features. Train a linear regression model, identify multicollinearity using VIF (Variance Inflation Factor), and apply solutions (dropping features or regularization).

## Exercise 10: Real-World Project
Choose a real dataset from Kaggle (e.g., Used Car Prices, Real Estate Prices). Build a linear regression model with data preprocessing, feature engineering, evaluation metrics, and a brief report on model interpretation.

