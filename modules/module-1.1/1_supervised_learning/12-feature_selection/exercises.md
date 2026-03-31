# 📝 Feature Selection — Exercises

## Exercise 1 — Manual Variance Threshold
Without using sklearn, implement a variance threshold filter: compute the variance of each feature column in the breast cancer dataset and sort features from lowest to highest variance. Plot a bar chart of all 30 feature variances. Set a threshold at the 25th percentile (drop the bottom 25% of features by variance). Retrain a LogisticRegression on the filtered features and compare 5-fold CV accuracy to using all 30 features. Does removing low-variance features help or hurt?

## Exercise 2 — Comparing Filter Criteria
On the breast cancer dataset, compute three filter scores for all 30 features: ANOVA F-test, Chi-square (use MinMaxScaler first to make all features non-negative), and mutual information. Create a DataFrame with all three scores and rank features by each. Plot a scatter matrix of the three rankings. Which features consistently score high across all three criteria? Which features are ranked very differently by different criteria — and why might that happen?

## Exercise 3 — Sequential Feature Selection
Implement forward stepwise selection from scratch. Start with an empty feature set. In each round, try adding each remaining feature, compute 5-fold CV accuracy, and permanently add the best one. Stop when adding any feature fails to improve accuracy by more than 0.1%. Plot the accuracy at each step. How many features are selected? Compare to RFECV's result.

## Exercise 4 — RFE vs RFECV
On the wine quality dataset (a regression problem with target = wine quality score), compare: (1) RFE selecting the top 5 features, (2) RFECV automatically finding the optimal number of features, and (3) SelectFromModel using a RandomForestRegressor's feature importance. For each, retrain a LinearRegression on the selected features and report test RMSE. Does selecting fewer features always reduce RMSE, or is there a sweet spot?

## Exercise 5 — Lasso as Feature Selector
On a dataset with 50 features (use make_regression with n_informative=5), train a LassoCV (which automatically cross-validates to find the optimal alpha). After fitting, count how many coefficients are exactly zero vs non-zero. Plot the LassoCV's coefficient path (each feature's coefficient vs log(alpha)). Do the 5 truly informative features have non-zero coefficients? At what alpha do the first noise features go to zero? At what alpha do the informative features go to zero?

## Exercise 6 — Feature Selection Leakage Warning
This exercise demonstrates a common mistake. On the breast cancer dataset: (1) Correct approach: run RFECV inside a cross-validation loop (i.e., do feature selection separately on each fold's training data, then evaluate on the fold's test data). (2) Wrong approach: select features on the entire dataset first, then cross-validate. Report both accuracy estimates. The wrong approach will likely give an optimistically biased estimate. Explain why this is data leakage.

## Exercise 7 — Interaction Features
The best features are sometimes combinations of existing ones. On the California Housing dataset: start with the 8 raw features. Use PolynomialFeatures(degree=2, interaction_only=True) to add all pairwise interaction features (8 original + 28 interactions = 36 total). Train LinearRegression on all 36 and compare RMSE to using only the 8 originals. Now use RFECV to select the best subset of the 36 features. Which interactions are selected? Do any interaction features outperform individual features?

## Exercise 8 — Permutation Importance for Feature Selection
Instead of removing features and retraining, use permutation importance: train a RandomForestClassifier on all features. For each feature, shuffle its values 10 times, compute the accuracy drop each time, and average. Features whose shuffling causes a large accuracy drop are important. Plot permutation importance with error bars. Set a threshold (e.g., only keep features where shuffling causes > 1% accuracy drop). Retrain with only those features and compare test accuracy.

## Exercise 9 — Feature Selection for Text
Load the 20 Newsgroups dataset with 4 categories. Build a TF-IDF matrix (it will have thousands of features). Apply SelectKBest with chi-square or mutual information to select the top 500, 200, 100, and 50 words. For each subset, train a MultinomialNB and record test accuracy. Plot accuracy vs number of features. Does accuracy drop off sharply or gradually as you reduce features? What's the minimum number of words you can use while maintaining 90% of peak accuracy?

## Exercise 10 — End-to-End Feature Engineering Pipeline
Take the Titanic dataset. Start with a brainstorm: what features might predict survival beyond the raw ones given? Consider: title extracted from Name (Mr, Mrs, Miss, Dr), family size (SibSp + Parch + 1), whether passenger was alone (family_size == 1), cabin deck (first letter of Cabin), fare per person (Fare / family_size). Add all these engineered features. Then run the full feature selection pipeline: (1) variance threshold to remove near-constant features, (2) mutual information to rank all features, (3) RFECV to find the optimal number. Report how the final model accuracy compares to a model without any feature engineering.
