# 📝 Gradient Boosting — Exercises

## Exercise 1 — Boosting vs Bagging Comparison
On the breast cancer dataset, train a RandomForestClassifier (n_estimators=200) and a GradientBoostingClassifier (n_estimators=200, learning_rate=0.1, max_depth=3). Compare test accuracy, training time, and 5-fold cross-validation score. Then deliberately add 10% label noise to the training data (randomly flip some y_train values) and repeat. Which model is more robust to noisy labels? Write a paragraph explaining why.

## Exercise 2 — Staged Predictions — Watching the Model Learn
GradientBoostingClassifier has a `staged_predict` method that gives predictions after each tree is added. Use it on breast cancer data to plot how training accuracy and test accuracy evolve as trees are added (x-axis: number of trees, y-axis: accuracy). Identify the point where test accuracy peaks and then starts to decline (overfitting). Annotate this point on the graph.

## Exercise 3 — Learning Rate vs n_estimators Trade-off
Train gradient boosting models with these (learning_rate, n_estimators) pairs: (0.1, 100), (0.05, 200), (0.01, 1000), (0.001, 5000). For each, report test accuracy and total training time. Create a scatter plot of training time vs accuracy. Conclude: is a lower learning rate always worth the extra training time?

## Exercise 4 — Early Stopping with XGBoost
Install xgboost (`pip install xgboost`). Train an XGBClassifier on breast cancer with n_estimators=2000, learning_rate=0.01, and early_stopping_rounds=50. Use 20% of the training data as a validation set. How many trees were actually used (check `model.best_iteration`)? Compare the test accuracy of the early-stopped model to one trained for exactly 2000 trees without early stopping. This demonstrates that early stopping is both more accurate AND more efficient.

## Exercise 5 — Feature Importance Comparison
On the California Housing dataset (regression, using GradientBoostingRegressor), compare feature importance from: (1) `model.feature_importances_` (MDI), and (2) permutation importance on the test set. Plot both as horizontal bar charts side by side. Do they agree on the top features? If they disagree, trust the permutation importance — explain in a comment why MDI can be misleading for tree-based models.

## Exercise 6 — Gradient Boosting for Ranking (LTR)
Gradient boosting is widely used for learning-to-rank in search engines. Simulate a ranking problem: create a dataset where 10 queries each have 20 documents with relevance scores (0, 1, 2). Use LightGBM's `lgb.Dataset` with group sizes and objective='lambdarank'. Evaluate using NDCG@10 (normalised discounted cumulative gain). Compare LightGBM's ranking performance to a simple regression baseline that predicts the relevance score directly.

## Exercise 7 — CatBoost with Categorical Features
Download the Mushroom dataset (UCI ML Repository) which has all categorical features. Compare: (1) encode all categories with one-hot encoding, then use GradientBoostingClassifier; (2) use CatBoostClassifier natively with cat_features parameter (no manual encoding needed). Report accuracy and training time for each. Note how CatBoost handles categoricals more efficiently and often more accurately.

## Exercise 8 — Hyperparameter Tuning with Optuna
Instead of grid search, use Optuna (`pip install optuna`) for Bayesian hyperparameter optimisation of an XGBClassifier on the breast cancer dataset. Define a study that searches over: n_estimators (50-1000), max_depth (2-8), learning_rate (0.001-0.5, log scale), subsample (0.5-1.0), colsample_bytree (0.5-1.0). Run 50 trials. Plot the optimization history (trial number vs best CV accuracy so far). Compare the best Optuna result to the best GridSearchCV result — which finds a better solution in fewer function evaluations?

## Exercise 9 — Gradient Boosting from Scratch (Simplified)
Implement a simplified gradient boosting regressor using only NumPy and sklearn's DecisionTreeRegressor. The algorithm: (1) initialise ŷ = mean(y), (2) for each iteration, compute residuals r = y - ŷ, (3) fit a DecisionTreeRegressor(max_depth=3) to r, (4) add the tree's predictions × learning_rate to ŷ, (5) repeat. Test on a sine wave with noise. Plot the prediction after 1, 5, 20, and 100 trees. See how the model progressively refines its prediction.

## Exercise 10 — Kaggle Tabular Competition Simulation
Use the "House Prices: Advanced Regression Techniques" dataset from Kaggle. Build a full ML pipeline: (1) EDA to understand missing values and distributions, (2) feature engineering (log-transform skewed features, create interaction features), (3) train LightGBM with cross-validation and early stopping, (4) tune with Optuna, (5) generate predictions for the test set and create a submission file. Record your public leaderboard score. Then try blending (averaging) LightGBM predictions with XGBoost predictions — does the blend score higher than either individual model?
