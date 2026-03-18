# Ensemble Methods - Exercises

## Exercise 1: Basic Voting Ensemble
Create an ensemble with 3 different base classifiers (e.g., SVM, Random Forest, Logistic Regression). Implement majority voting and evaluate.

## Exercise 2: Soft Voting Comparison
Compare hard voting (majority) with soft voting (probability averaging) on a multi-class classification task. Analyze which performs better.

## Exercise 3: Averaging Predictions
Train multiple models and combine predictions through averaging (uniform and weighted). Analyze how prediction averaging reduces variance.

## Exercise 4: Stacking Implementation
Implement a stacking ensemble with level-0 base learners (RF, SVM, KNN) and a level-1 meta-learner (Logistic Regression). Compare with individual models.

## Exercise 5: Blending Approach
Implement blending ensembles where a portion of training data trains base learners and remaining data trains the meta-learner. Compare with stacking.

## Exercise 6: Negative Correlation Learning
Create an ensemble where base models are intentionally diverse. Measure the trade-off between individual model accuracy and ensemble diversity.

## Exercise 7: Boosting vs. Bagging Ensemble
Compare a boosting ensemble (Gradient Boosting), bagging ensemble (Random Forest), and stacking ensemble on the same dataset.

## Exercise 8: Hyperparameter Search for Ensemble
Use GridSearchCV to optimize base learner hyperparameters and meta-learner parameters. Analyze improvement vs. single model.

## Exercise 9: Uncertainty Estimation
Use ensemble methods to estimate prediction confidence and uncertainty. Analyze confidence calibration and prediction intervals.

## Exercise 10: Kaggle-Style Ensemble Winning Solution
Build a complete ensemble system combining multiple algorithms (RF, GB, SVM, NN), implement stacking/blending, and achieve competitive performance on a competition dataset.

