# 📝 Support Vector Machines — Exercises

## Exercise 1 — Visualising the Margin
Create a 2D dataset with two clearly separable classes (use `make_classification(n_features=2, n_redundant=0)`). Train a linear SVM. Visualise the decision boundary, the two margin lines (w·x + b = ±1), and highlight the support vectors with a special marker (e.g., circled points). Annotate the margin width on the plot.

## Exercise 2 — Effect of C Parameter
On the moons dataset, train SVMs with C = 0.001, 0.01, 0.1, 1, 10, 100. Plot the decision boundary for each in a 2×3 subplot grid. Observe how small C creates a smooth, wide-margin boundary and large C creates a tight, wiggly boundary. Record training and test accuracy for each C. At what C does overfitting start? At what C does underfitting occur?

## Exercise 3 — Kernel Comparison
On the same moons dataset, train SVMs with linear, polynomial (degree=3), RBF, and sigmoid kernels, all with C=1. Plot the decision boundary for each. Which kernel gives the most sensible boundary for this crescent-shaped data? Report test accuracy for each. Write a comment explaining *why* the RBF kernel works better than linear for this data.

## Exercise 4 — Gamma Parameter for RBF Kernel
With a fixed C=1 and RBF kernel, vary gamma from 0.001 to 100 (use log spacing). Plot the decision boundary for gamma = 0.01, 0.1, 1, 10, 100. Observe how very small gamma creates a nearly linear boundary (global influence) and very large gamma creates tiny "bubbles" around each training point (local influence → overfitting). Record training and validation accuracy for each gamma.

## Exercise 5 — SVM vs Logistic Regression on Breast Cancer
Train both an SVM (with tuned C and gamma) and a logistic regression (with tuned C) on the breast cancer dataset. Report test accuracy, precision, recall, F1, and training time for each. Plot both ROC curves on the same axes. Write a paragraph comparing them: which is better? Which is faster to train? Which would you use in practice and why?

## Exercise 6 — Text Classification with Linear SVM
Download the 20 Newsgroups dataset (`sklearn.datasets.fetch_20newsgroups`). Use only 4 categories. Convert text to TF-IDF features (`TfidfVectorizer`). Train a `LinearSVC` (much faster than SVC for text). Report accuracy and a classification report. Inspect which words have the highest positive and negative weights for each class — these are the words most associated with each topic.

## Exercise 7 — GridSearchCV for C and Gamma
On the breast cancer dataset, systematically search over `C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]` and `gamma = ['scale', 0.001, 0.01, 0.1, 1]` using 5-fold cross-validation. Create a heatmap of the cross-validation accuracy grid (C on one axis, gamma on the other). Identify the best region. What is the final test accuracy of the optimally tuned SVM?

## Exercise 8 — Support Vectors Analysis
Train a linear SVM on a 2D dataset. Access `model.support_vectors_` and `model.support_`. How many support vectors are there? What percentage of the training set are they? Now re-train on only those support vectors — the model should give identical results, because support vectors are the *only* training points that define the boundary. Verify this.

## Exercise 9 — SVR for Regression
Load the diabetes dataset. Train a `SVR` with different kernels and C values. Compare RMSE against `LinearRegression` and `RandomForestRegressor`. Plot actual vs predicted for the best SVR. When does SVR give better results than linear regression? When does the random forest beat SVR?

## Exercise 10 — Real World: Handwritten Digit Recognition
Load the MNIST digits dataset (`sklearn.datasets.load_digits()`). This has 1,797 samples of 8×8 pixel handwritten digit images (10 classes). Train an SVM with RBF kernel. Use 5-fold cross-validation to tune C and gamma. Report final test accuracy and a full confusion matrix heatmap. Which digit pairs does the model most often confuse? Visualise 5 misclassified examples — can you see why the model got confused?
