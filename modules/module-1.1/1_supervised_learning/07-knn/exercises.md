# 📝 K-Nearest Neighbours — Exercises

## Exercise 1 — KNN from Scratch
Implement KNN classification without sklearn. Your function should accept X_train, y_train, X_new, and K. For each test point, compute Euclidean distance to all training points, find the K smallest, and return the majority vote. Test on the Iris dataset (2 features only so you can visualise). Compare accuracy to sklearn's KNeighborsClassifier.

## Exercise 2 — Effect of K on Decision Boundaries
Using 2D Iris data (petal length and petal width), plot the KNN decision boundaries for K = 1, 3, 7, 15, 30. Create a 1×5 subplot grid. Use a mesh grid to colour each region. Annotate each plot with the test accuracy. Observe the progression from jagged (K=1) to smooth (K=30) boundaries and notice where each K model makes errors.

## Exercise 3 — Feature Scaling Impact
On the KNN classifier (K=5) for the breast cancer dataset, compare test accuracy with and without StandardScaler. Also try MinMaxScaler. Plot a bar chart of accuracies for: no scaling, StandardScaler, MinMaxScaler. The difference should be dramatic — breast cancer features have very different scales, making unscaled KNN essentially use only the largest-scale features.

## Exercise 4 — Distance Metric Comparison
On the Iris dataset (all 4 features, scaled), train KNN classifiers using Euclidean (p=2), Manhattan (p=1), and Chebyshev (p=inf) distances, all with K=5. Report test accuracy for each. Then try K = 1, 5, 10, 20 with each metric — create a table of accuracies. Which metric is most robust across different K values? Research when you'd prefer Manhattan over Euclidean distance (hint: consider outliers and high dimensions).

## Exercise 5 — Cross-Validation Curve for K
On the wine dataset, compute 5-fold CV accuracy for K from 1 to 50. Plot the curve. Add error bars using the standard deviation of the 5 fold scores. Identify: (1) the optimal K, (2) the range of K values within one standard deviation of the best, (3) whether the optimal K is stable or sensitive (i.e., does the curve have a sharp peak or a flat plateau?). A flat plateau means the model is robust to K choice.

## Exercise 6 — Weighted vs Uniform Voting
Compare KNeighborsClassifier with weights='uniform' versus weights='distance' on the breast cancer dataset for K = 1, 3, 5, 10, 15, 20. Plot two accuracy curves. At which K values does weighted voting help the most? Why does weighted voting especially help when K is large and some neighbours are much further away than others?

## Exercise 7 — KNN for Anomaly Detection
Generate a dataset of 200 "normal" 2D points from a Gaussian distribution centred at (0,0), and 10 "anomalous" points scattered far from the centre. Train KNN on normal points only. For anomaly scoring, use the average distance to the 5 nearest neighbours — a high score indicates the point is far from all normal points (anomalous). Plot the data, colour-coding actual normals vs anomalies. Plot the anomaly score distribution and find a threshold that correctly identifies all 10 anomalies.

## Exercise 8 — KNN Regression vs Linear Regression
Load the California housing dataset. Train KNeighborsRegressor for K = 1, 5, 10, 25 and LinearRegression. Report RMSE and R² for all. Plot learning curves (training set size vs test RMSE) for KNN(K=10) and LinearRegression. In what regions of the feature space does KNN outperform linear regression? In what regions does it underperform?

## Exercise 9 — The Curse of Dimensionality (Demonstration)
Create 1,000 training points uniformly distributed in the unit hypercube [0,1]^d for d = 1, 2, 5, 10, 20, 50, 100 dimensions. For each dimensionality, sample 100 query points and compute the distance from each to its 1-nearest neighbour in the training set. Plot mean nearest-neighbour distance vs dimensionality. Observe how quickly the distance grows — making "nearest" less and less meaningful. Also measure KNN (K=5) test accuracy on a simple synthetic classification task across dimensions. When does accuracy start dropping?

## Exercise 10 — Movie Recommendation System
Build a collaborative filtering recommendation system using KNN. Use the MovieLens Small dataset (available at https://grouplens.org/datasets/movielens/latest/). Create a user-item rating matrix. For a given user, find the K most similar users (by cosine similarity on their rating vectors). Recommend movies that these similar users rated highly but the target user hasn't seen. Evaluate using leave-one-out validation: hide each user's most recent rating, recommend, and check if that movie appears in the top-10 recommendations. Compute the hit rate.
