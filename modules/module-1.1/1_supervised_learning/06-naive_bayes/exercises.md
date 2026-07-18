# 📝 Naive Bayes — Exercises

## Exercise 1 — Implementing Gaussian Naive Bayes from Scratch
Without using sklearn, implement GaussianNB. During training, for each class compute the mean and variance of each feature (using numpy only). During prediction, compute the log-likelihood of each class using the Gaussian PDF formula: log P(x|class) = -0.5 × log(2πσ²) − (x−μ)² / (2σ²). Add the log prior. The class with the highest total log score is the prediction. Test on the Iris dataset and confirm your accuracy matches sklearn's GaussianNB.

## Exercise 2 — Spam Classifier with MultinomialNB
Download the SMS Spam Collection dataset (available at UCI ML Repository). Build a pipeline: `CountVectorizer(ngram_range=(1,2))` → `MultinomialNB`. Train on 80% of the data and evaluate on 20%. Report accuracy, precision, recall, and F1 for the spam class specifically. Experiment with different alpha values (0.001, 0.01, 0.1, 1.0, 10.0) and plot F1 score vs alpha to find the optimal smoothing.

## Exercise 3 — Prior Probability Effect
On an imbalanced binary dataset (95% class 0, 5% class 1), train GaussianNB with default priors (which use the training class proportions) and then with manually overridden priors (try priors=[0.5, 0.5] to give equal weight to both classes). Compare recall for the minority class under each setting. This demonstrates how the prior acts as a way to incorporate domain knowledge about expected class frequencies.

## Exercise 4 — Laplace Smoothing Deep Dive
Create a small binary classification problem with very sparse features (many zeros). Train MultinomialNB with alpha = 0, 0.1, 0.5, 1.0, and 5.0. For alpha=0, intentionally include a test example that has a feature value that was never seen for one of the classes during training — observe that the prediction fails (zero probability). For each non-zero alpha, record test accuracy and explain in a comment why alpha=0 causes problems and how larger alpha values create a more robust model.

## Exercise 5 — Naive Bayes vs Logistic Regression on Text
On the 20 Newsgroups dataset (4 categories), compare MultinomialNB against LogisticRegression with L2 regularisation. Report accuracy, training time, and memory usage for each. Plot ROC-AUC for both (one-vs-rest). Which model performs better? Which is faster? In what production scenarios would you prefer the Naive Bayes despite potentially lower accuracy?

## Exercise 6 — Feature Analysis: What the Model Learned
After training a MultinomialNB on the 20 Newsgroups dataset, extract `clf.feature_log_prob_` (the log probability of each word given each class). For each of the 4 categories, print the 15 words with the highest log probability — these are the most "diagnostic" words for each topic. Do these words make intuitive sense? Also find the 5 words that are roughly equally likely across all classes — these are the least discriminative words. Why would you want to know this?

## Exercise 7 — Bernoulli vs Multinomial for Text
On the same 20 Newsgroups data, compare BernoulliNB (binary presence/absence of words) against MultinomialNB (word count frequencies). Which performs better overall? At what document lengths does Bernoulli start to lose information that Multinomial retains? Train both on short (< 100 words) and long (> 500 words) documents separately and compare their performance on each group.

## Exercise 8 — Gaussian NB on Real-Valued Medical Data
Load the breast cancer dataset (30 continuous features). Train GaussianNB. Plot the learned Gaussian distributions for the 3 most important features — show two overlapping curves per feature (one for each class). The visual overlap tells you how discriminative each feature is. Use these plots to explain to a non-technical person what the model is doing: "The model knows that malignant tumours tend to have larger values of feature X, with a spread of Y."

## Exercise 9 — Online Learning with Partial Fit
One huge advantage of Naive Bayes is that it supports online (incremental) learning — you can update the model with new data without retraining from scratch. Use `GaussianNB.partial_fit()` to simulate streaming data: start with 10% of the breast cancer training set, fit the model, then add 10% more with `partial_fit`, repeat until all data is seen. Record accuracy after each batch. Compare the final accuracy to a model trained on all data at once. This shows how the model can be updated in production as new data arrives.

## Exercise 10 — End-to-End News Article Classifier
Build a complete news article classification system. Use the full 20 Newsgroups dataset (20 categories). Build a pipeline with TF-IDF features (not raw counts) and MultinomialNB. Tune alpha and max_features via GridSearchCV. Report final test accuracy and a heatmap confusion matrix (all 20 classes). Identify the 3 pairs of categories most often confused with each other — this tells you where the model's weaknesses lie. Finally, test your model on 5 news headlines you write yourself and see if it classifies them correctly.
