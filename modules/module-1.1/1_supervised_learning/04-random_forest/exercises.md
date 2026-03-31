# 📝 Random Forests — Exercises

> Each exercise here is designed to deepen your intuition for *why* random forests work, not just *how* to use them. The goal is to see the ensemble effect in action.

---

## Exercise 1 — Bootstrap Sampling Simulation

Before touching sklearn, understand bootstrap sampling. Write a function `bootstrap_sample(X, y)` that draws N samples with replacement from a dataset of size N. Call it 10 times and for each bootstrap sample, print: (1) which training examples were included (their indices), (2) which were left out (the OOB examples), and (3) the proportion of unique examples in each bootstrap (should average around 63.2%). Compute the theoretical OOB proportion mathematically: the probability that any given example is NOT selected in a single draw is `(1 - 1/N)`. For a sample of size N, the probability of never being selected is `(1 - 1/N)^N`, which approaches `1/e ≈ 0.368` as N grows. Verify this empirically.

---

## Exercise 2 — Single Tree vs Forest: Variance Experiment

This exercise makes the variance reduction visible and quantitative. On the breast cancer dataset, repeat the following 50 times with different random seeds: train one single decision tree (max_depth=None) and one random forest (n_estimators=100) on the same 80/20 split. Record the test accuracy each time. Plot two histograms — one for the single tree accuracies and one for the forest accuracies — on the same axes. Compute and compare the mean and standard deviation of each. The forest's distribution should be narrower (lower standard deviation) and shifted right (higher mean), making the variance reduction concrete.

---

## Exercise 3 — The Effect of n_estimators

Train random forests with n_estimators from 1 to 200 (step by 5) on the Iris dataset. For each, record OOB accuracy. Plot OOB accuracy vs n_estimators. You will see the characteristic curve: accuracy starts low with very few trees, rises sharply, then flattens out. Identify the "elbow point" where adding more trees gives negligible benefit. Annotate it on the graph. This is your answer to "how many trees do I need?"

---

## Exercise 4 — Feature Importance: MDI vs Permutation

On the California Housing dataset (regression), compare the two feature importance methods. Train a RandomForestRegressor (n_estimators=200). Plot both MDI importance and permutation importance side by side as horizontal bar charts. Rank features by each method and note any differences. Research why MDI can be biased toward high-cardinality features and why permutation importance avoids this bias. Write your conclusion as a comment: which method would you trust more, and when would you use each?

---

## Exercise 5 — OOB Score as Model Selection

When you have limited data and cannot afford a separate validation set, OOB score is invaluable. On the wine quality dataset (`sklearn.datasets.load_wine()`), tune `max_features` (try 'sqrt', 'log2', and values 2, 4, 6, 8) using OOB score as your metric. For each setting, report the OOB score. Plot OOB score vs max_features. Which value wins? Now verify by checking test accuracy on a held-out set — does the OOB-optimal setting also win on the test set?

---

## Exercise 6 — Random Forest for Regression

Load the diabetes dataset. Train a RandomForestRegressor and compare it to the LinearRegression from an earlier lesson. Report RMSE and R² for both. Then plot the actual vs predicted values for the random forest — scatter them and draw a diagonal "perfect prediction" line. Are there systematic regions where the forest is wrong? When would you prefer the forest over linear regression for a regression problem?

---

## Exercise 7 — Partial Dependence Plots

Partial dependence plots show the marginal effect of one or two features on the model's prediction, averaging over all other features. Using your trained RandomForestClassifier on the breast cancer dataset, generate partial dependence plots for the top 3 most important features (by permutation importance). Use `sklearn.inspection.PartialDependenceDisplay`. Interpret each plot: does the predicted probability of malignancy increase or decrease as each feature increases? Does the relationship look monotonic, or does it have interesting non-linearities?

---

## Exercise 8 — Handling Imbalanced Classes

On an imbalanced dataset (use `make_classification(weights=[0.9, 0.1])` to create one), compare three approaches: (1) a plain random forest, (2) a random forest with `class_weight='balanced'`, and (3) a random forest trained on SMOTE-oversampled data. For each, report F1 score for the minority class and AUC-ROC. Plot three ROC curves on the same axes. Which approach gives the best minority-class recall? Write a recommendation for when you'd use each approach in a production setting.

---

## Exercise 9 — Extremely Randomised Trees (Extra-Trees)

Scikit-learn also has `ExtraTreesClassifier` — the key difference from random forests is that it picks split thresholds *randomly* rather than finding the optimal threshold. Train both a RandomForestClassifier and ExtraTreesClassifier on the breast cancer dataset with n_estimators=200. Compare: training time (use `time.time()`), training accuracy, test accuracy, and OOB score. Extra-Trees trains faster because it skips the expensive threshold-optimisation step. Does it pay off in accuracy? Write when you'd choose Extra-Trees over Random Forest.

---

## Exercise 10 — Full Pipeline: Telecom Customer Churn

Download or simulate a customer churn dataset (you can find one on Kaggle: "Telco Customer Churn"). Build a full ML pipeline: (1) Explore the data — what percentage of customers churned? What features correlate with churn? (2) Preprocess — handle categoricals with one-hot encoding, handle any missing values. (3) Train a RandomForestClassifier with OOB scoring and tune n_estimators and max_features. (4) Evaluate on a test set with a full classification report and ROC-AUC. (5) Plot and interpret feature importances — which customer behaviours most predict churn? (6) Write a 5-sentence business recommendation: given what the model learned, what actions should the telecom company take to reduce churn?
