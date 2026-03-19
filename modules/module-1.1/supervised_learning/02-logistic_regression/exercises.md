# Logistic Regression - Exercises

## Exercise 1: Binary Classification with Logistic Regression
Load the Iris dataset (binary: Setosa vs. non-Setosa) and build a logistic regression classifier. Evaluate using accuracy, precision, recall, and F1-score.

## Exercise 2: Multi-class Classification
Extend logistic regression to multi-class classification using the entire Iris dataset (3 classes). Compare One-vs-Rest vs. Softmax approaches and discuss their differences.

## Exercise 3: Feature Scaling and Normalization
Train two logistic regression models: one with raw features and one with scaled features. Show how feature scaling affects convergence speed and model performance.

## Exercise 4: Probability Calibration
Train a logistic regression model and plot the predicted probabilities. Analyze the calibration curve and discuss whether the predicted probabilities are well-calibrated.

## Exercise 5: Decision Boundary Visualization
For a 2-feature dataset, train a logistic regression model and visualize the decision boundary. Show how changing the probability threshold (0.3, 0.5, 0.7) affects the boundary.

## Exercise 6: Class Imbalance Handling
Create a binary classification dataset with imbalanced classes. Train logistic regression models using:
- Standard approach
- Class weights
- Oversampling/Undersampling
Compare results and discuss trade-offs.

## Exercise 7: Cross-Validation and Hyperparameter Tuning
Apply k-fold cross-validation (k=5) and grid search to find optimal hyperparameters (C, penalty type) for logistic regression on a real dataset.

## Exercise 8: ROC and AUC Analysis
Train a logistic regression model and plot the ROC curve. Calculate AUC score and explain its interpretation. Compare with other metrics like accuracy and F1-score.

## Exercise 9: Regularization Comparison
Train three logistic regression models with L1, L2, and no regularization on the same dataset. Compare their coefficients and discuss bias-variance trade-off.

## Exercise 10: Real-World Classification Project
Use a real dataset (e.g., Titanic, Bank Marketing) to build a complete logistic regression pipeline including:
- Data preprocessing and EDA
- Feature engineering
- Model training with cross-validation
- Evaluation metrics and ROC curve
- Interpretation of coefficients

