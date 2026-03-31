# 📝 Ensemble Methods — Exercises

## Exercise 1 — Diversity is the Key
Create 10 different classifiers on the breast cancer dataset (different algorithms, hyperparameters, random seeds). Compute a "diversity matrix": for each pair of classifiers, compute what percentage of test examples they *disagree* on. Plot this as a heatmap. Identify the two most diverse classifiers (highest disagreement). Build a voting ensemble from those two — does it outperform either individually? Now pick two very similar classifiers (low disagreement) and ensemble them. Does it help?

## Exercise 2 — Hard vs Soft Voting
On the Iris dataset (3 classes), compare hard voting vs soft voting using the same set of 3 classifiers. Deliberately create a case where they disagree: find a test example where one classifier outputs a very low-confidence prediction. Verify that soft voting correctly ignores this weak vote, while hard voting gives it the same weight as a highly-confident prediction.

## Exercise 3 — Bagging from Scratch
Implement a BaggingClassifier from scratch using only NumPy and sklearn's DecisionTreeClassifier. Train 50 decision trees (each on a different bootstrap sample). Average their predicted probabilities for the final prediction. Compare accuracy to a single decision tree and sklearn's BaggingClassifier. The manually-built bagging ensemble should closely match sklearn's result.

## Exercise 4 — Optimal Meta-Learner for Stacking
On the breast cancer dataset, use LogisticRegression, RandomForest, and GradientBoosting as base models. Compare three different meta-learners: LogisticRegression, Ridge, and another RandomForest. Which meta-learner works best? Also compare passthrough=True (meta-learner sees both base predictions AND original features) vs passthrough=False. Does giving the meta-learner the original features help?

## Exercise 5 — Kaggle Ensemble: Reproducing a Winning Strategy
Read about ensembling strategies used in winning Kaggle solutions (the "Two Sigma Financial Modelling Challenge" or "Otto Group Product Classification" are well-documented). Implement their specific approach: they typically use cross-validated stacking with XGBoost, LightGBM, and neural networks as base models and a simple logistic regression meta-learner. Apply this pipeline to the breast cancer dataset and compare results to simpler ensembles.

## Exercise 6 — Snapshot Ensemble
A snapshot ensemble trains a single model but saves "snapshots" of its weights at multiple points during training (usually at the end of each cosine annealing cycle). Train a neural network on MNIST with a cosine learning rate schedule that completes 5 cycles. Save the model weights at the end of each cycle. Use all 5 saved models as an ensemble — compare this ensemble's accuracy to any single snapshot. This gives ensemble benefits with only one model's worth of training.

## Exercise 7 — Test-Time Augmentation (TTA)
TTA is a form of ensembling at prediction time. Train a Keras model on MNIST. At test time, instead of predicting once on the original image, predict on the original PLUS slightly rotated/shifted versions (say, 10 augmented copies), then average the predictions. Report accuracy with and without TTA. TTA typically gives a 0.1–0.5% boost on image tasks.

## Exercise 8 — Weighted Ensemble
Instead of giving each model an equal vote, optimise the weights using the validation set. Train 5 classifiers. Their predictions on the validation set are [p1, p2, p3, p4, p5] (probability vectors). Find weights w = [w1, w2, w3, w4, w5] (summing to 1, all ≥ 0) that maximise validation accuracy. Use scipy.optimize.minimize. Compare the optimally-weighted ensemble to the equal-weight ensemble.

## Exercise 9 — Feature-Level Ensemble
Instead of ensembling predictions, ensemble the feature representations. Train two different models on the breast cancer data (say, PCA-transformed features + neural network, and raw features + random forest). Extract the internal representations (the penultimate layer of the neural network and the leaf node embeddings of the forest). Concatenate these embeddings and train a final logistic regression on the combined representation. Does this "feature-level blending" beat standard prediction-level ensembling?

## Exercise 10 — Full Stacking Pipeline on a Real Competition
Using the Titanic dataset, implement a complete stacking pipeline from scratch (without sklearn's StackingClassifier): (1) Split training data into 5 folds. (2) For each fold, train LogisticRegression, RandomForest, GradientBoosting, and SVM on the 4 other folds and generate predictions for that fold. (3) After all folds, concatenate predictions to get out-of-fold predictions for all training examples. (4) Train a logistic regression meta-learner on these out-of-fold predictions. (5) To predict on the test set, average each base model's predictions across all 5 folds, then pass through the meta-learner. Report final test accuracy and compare to the best single model.
