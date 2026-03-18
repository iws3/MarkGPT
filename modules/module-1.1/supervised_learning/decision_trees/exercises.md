# Decision Trees - Exercises

## Exercise 1: Basic Tree Building
Build a decision tree classifier on the Iris dataset. Visualize the tree structure using graphviz and explain the splits and decision rules.

## Exercise 2: Tree Depth and Overfitting
Train decision trees with varying max_depth (2, 5, 10, 15) on the same dataset. Plot training vs. test accuracy to identify overfitting and find the optimal depth.

## Exercise 3: Splitting Criteria Comparison
Train two decision trees on a classification dataset: one using Gini impurity and another using information gain. Compare the resulting trees and their performance metrics.

## Exercise 4: Feature Importance
Train a decision tree on a dataset with multiple features. Extract and visualize feature importances. Discuss which features are most discriminative.

## Exercise 5: Regression with Decision Trees
Build a decision tree regressor for continuous target prediction. Compare MSE and R² with linear regression on the same dataset.

## Exercise 6: Tree Pruning
Train a full decision tree and then apply cost-complexity pruning. Plot alpha vs. accuracy and select the best pruned tree using cross-validation.

## Exercise 7: Handling Imbalanced Data
Train decision trees on imbalanced classification data. Experiment with class_weight='balanced' and compare results. Discuss interpretability benefits.

## Exercise 8: Cross-Validation and Hyperparameter Tuning
Use GridSearchCV to find optimal hyperparameters (max_depth, min_samples_split, min_samples_leaf). Report the best parameters and improvement over baseline.

## Exercise 9: Decision Rules Extraction
Train a decision tree and manually extract decision rules from the tree structure. Write these rules as human-readable if-then statements for business use.

## Exercise 10: Real-World Project with Tree Interpretation
Use a real dataset to build a decision tree classifier. Focus on interpretability: visualize the tree, extract decision rules, and explain business implications to a non-technical audience.

