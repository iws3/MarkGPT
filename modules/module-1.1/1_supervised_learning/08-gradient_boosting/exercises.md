# Gradient Boosting - Exercises

## Exercise 1: Basic Gradient Boosting
Implement Gradient Boosting classifier on a binary classification dataset using scikit-learn's GradientBoostingClassifier. Evaluate performance.

## Exercise 2: Number of Estimators Impact
Train GB models with varying n_estimators (10, 50, 100, 200, 500). Plot train/test error curves and identify overfitting point.

## Exercise 3: Learning Rate Tuning
Train GB models with different learning rates (0.001, 0.01, 0.1, 0.5). Analyze convergence speed and final performance. Discuss learning rate trade-offs.

## Exercise 4: Feature Importance from GB
Train a gradient boosting model and extract feature importances. Visualize and compare with Random Forest importances.

## Exercise 5: XGBoost vs. GradientBoosting
Implement both scikit-learn's GradientBoostingClassifier and XGBoost. Compare performance, training time, and ease of use.

## Exercise 6: Regression with Gradient Boosting
Build a GB regressor for continuous prediction. Compare MSE with linear regression and random forest regressors.

## Exercise 7: Handling Class Imbalance
Train GB on imbalanced classification data. Use scale_pos_weight or sample_weight to address class imbalance and compare results.

## Exercise 8: Hyperparameter Optimization
Use GridSearchCV or RandomizedSearchCV to tune GB hyperparameters (n_estimators, learning_rate, max_depth, subsample). Report optimal parameters and CV scores.

## Exercise 9: Early Stopping
Implement early stopping in XGBoost to prevent overfitting. Monitor validation metrics and stop training when performance plateaus.

## Exercise 10: End-to-End Kaggle-Style Project
Build a complete GB pipeline on a real competition-style dataset: data preprocessing, feature engineering, model tuning, ensemble stacking, and final evaluation.

