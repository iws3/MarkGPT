# 📝 Decision Trees — Exercises

> These exercises progress from understanding the fundamentals to diagnosing overfitting and applying trees to real datasets. Write your observations as code comments — the "why" matters as much as the code.

---

## Exercise 1 — Computing Entropy and Gini by Hand

Before trusting a library to do it for you, implement both measures yourself. Write functions `entropy(labels)` and `gini(labels)` that accept a list of class labels and return the impurity score. Then verify: for `[0,0,0,0]` both should return 0 (pure), for `[0,1,0,1]` entropy should return 1.0 and gini should return 0.5, and for `[0,0,0,1]` entropy should be about 0.811 and gini should be 0.375. Plot both measures side-by-side as a function of the proportion p of class 1, for p from 0 to 1. Notice how similar their shapes are — this explains why the choice rarely matters in practice.

---

## Exercise 2 — Information Gain from Scratch

Using your entropy function from Exercise 1, implement `information_gain(parent_labels, left_labels, right_labels)`. Test it on a simple example: a parent node with `[0,0,0,0,1,1,1,1]` (50/50 split). Compare two possible splits: one that perfectly separates the classes and one that leaves the distribution unchanged. Confirm that a perfect split gives IG = 1.0 and a useless split gives IG = 0.0. This is exactly what the decision tree algorithm does internally at every node for every possible feature and threshold.

---

## Exercise 3 — Build a Decision Stump (Depth-1 Tree)

Manually implement a decision stump — a tree with exactly one split. Write a function that takes X and y, loops over every feature and every unique threshold value, computes the information gain for each split, and returns the best feature index and threshold. Test it on the Iris dataset. The algorithm should find that petal length (or petal width) is the best first split — check whether it matches what `DecisionTreeClassifier(max_depth=1)` learns.

---

## Exercise 4 — Visualising Decision Boundaries

Decision trees create rectangular decision regions (because each split is a straight line parallel to one axis). Visualise this by: loading the Iris dataset and using only two features (petal length and petal width), training trees of depth 1, 2, 3, and 5, and plotting the decision boundary for each on a 2D scatter plot using a mesh grid. You'll see how the boundaries become more rectangular and complex as depth increases — and how the depth-5 tree creates very small regions that perfectly fit individual training points (overfitting).

```python
def plot_decision_boundary(model, X, y, feature_names, ax, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=30)
    ax.set_title(title)
```

---

## Exercise 5 — Depth vs Accuracy Experiment

This is the bias-variance tradeoff made visible. Train decision trees of depth 1 through 20 on the breast cancer dataset. For each depth, record training accuracy and 5-fold cross-validation accuracy. Plot both curves on the same graph with depth on the x-axis. Identify: the depth where underfitting is most obvious (both train and val accuracy are low), the depth where overfitting begins (train accuracy still rises but val accuracy plateaus or drops), and the optimal depth (highest validation accuracy). Mark these three regions on your graph with annotations.

---

## Exercise 6 — Feature Importance Analysis

Train a decision tree on the California Housing dataset to predict whether a house is expensive (above median price → binary classification). After training, extract `feature_importances_` and visualise them as a horizontal bar chart, sorted from most to least important. Then answer: which feature is most important for predicting house price? Does that match your intuition? Now deliberately remove the top 2 features and retrain — how much does accuracy drop? This reveals how much of the model's power comes from just a few features.

---

## Exercise 7 — Cost-Complexity Pruning

Scikit-learn provides `ccp_alpha` for post-training pruning. Use `tree.cost_complexity_pruning_path(X_train, y_train)` to get the sequence of alpha values and their corresponding number of leaf nodes. Plot the number of leaves vs alpha. Then train one tree for each alpha value and plot training and test accuracy vs alpha. You'll see that small alpha means a complex overfit tree, and large alpha means a heavily pruned simple tree — find the alpha that maximises test accuracy.

---

## Exercise 8 — Decision Tree for Regression

Load the diabetes dataset (`sklearn.datasets.load_diabetes()`). This is a regression problem — predict disease progression from 10 features. Train decision tree regressors with max_depth from 1 to 15. For each, record training RMSE and test RMSE. Plot both curves. Compare the best tree's RMSE to a simple linear regression baseline. Write a short comment: in what situations would a regression tree outperform linear regression?

---

## Exercise 9 — Interpretability Challenge

Train a decision tree on the Titanic dataset (you used it in logistic regression Exercise 10). Keep max_depth ≤ 3 to ensure the tree is interpretable. Use `export_text()` to print the tree as text and `plot_tree()` to visualise it. Then write, in plain English (as a code comment), the exact survival rules your tree learned. For example: "Passengers who were female and in 1st or 2nd class had a 95% survival rate." Can you extract at least 4 distinct rules from the tree? Do they match historical accounts of the Titanic disaster?

---

## Exercise 10 — Decision Tree vs Logistic Regression Head-to-Head

Pick a dataset you've worked with before (breast cancer or Titanic). Train both a tuned logistic regression (with optimal C) and a tuned decision tree (with optimal max_depth). For each model: report test accuracy, precision, recall, F1, and ROC-AUC. Plot both ROC curves on the same axes. Then write a paragraph comparing them on these dimensions: (1) which has higher accuracy? (2) which is easier to explain to a non-technical person? (3) which is more sensitive to the random seed / data split? (4) which would you choose for a medical application, and why?
