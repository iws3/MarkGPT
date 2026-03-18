# Random Forest - Exercises

## Exercise 1: Basic Random Forest
Train a Random Forest classifier on the Iris dataset with default parameters. Compare accuracy with a single decision tree to demonstrate ensemble benefits.

## Exercise 2: Number of Trees Impact
Train Random Forest models with varying n_estimators (10, 50, 100, 200, 500) on a classification dataset. Plot OOB error vs. number of trees and identify diminishing returns.

## Exercise 3: Feature Importance Ranking
Train a Random Forest on a multi-feature dataset. Extract and visualize feature importances. Compare with importances from a single decision tree.

## Exercise 4: Feature Subsampling
Train Random Forest models with different max_features values (sqrt, log2, all) on the same data. Discuss the trade-off between bias and variance.

## Exercise 5: Parallel Processing
Implement a Random Forest with n_jobs parameter to enable parallel tree building. Measure training time with single vs. multiple cores.

## Exercise 6: Out-of-Bag (OOB) Error Estimation
Train a Random Forest and use OOB error as a validation metric without explicit train-test split. Compare OOB error with cross-validation results.

## Exercise 7: Regression with Random Forest
Build a Random Forest regressor and compare with linear regression and individual decision tree regressor on a real dataset.

## Exercise 8: Hyperparameter Optimization
Use GridSearchCV to tune Random Forest hyperparameters (n_estimators, max_depth, min_samples_split, max_features). Compare results with default parameters.

## Exercise 9: Class Imbalance Handling
Train Random Forest on imbalanced classification data. Experiment with class_weight='balanced_subsample' and compare with other approaches.

## Exercise 10: End-to-End Project with Feature Engineering
Use a real dataset, perform feature engineering, train a tuned Random Forest, perform nested cross-validation, and create a comprehensive report with visualizations.

