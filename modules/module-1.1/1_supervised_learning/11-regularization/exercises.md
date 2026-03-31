# 📝 Regularisation — Exercises

## Exercise 1 — Visualising Overfitting and the Fix
Generate 30 data points from y = sin(2πx) + noise. Fit polynomial regression of degree 1, 5, 10, and 15 (no regularisation), then fit degree-15 with Ridge alpha=0.001, 0.1, 10. Plot all 7 curves on the same figure. Write a comment for each: is it underfitting, just right, or overfitting? This makes the bias-variance tradeoff tangible.

## Exercise 2 — Lasso as Feature Selector
Generate a dataset with 50 features but only 5 that truly affect the target (use make_regression with n_informative=5). Train Lasso with alpha=0.1. Check which coefficients are exactly zero. Verify that Lasso correctly identifies roughly the 5 truly informative features. Then try Ridge — do all 50 coefficients stay non-zero? Plot a bar chart of both models' coefficients side by side.

## Exercise 3 — Cross-Validated Regularisation Strength
Use RidgeCV and LassoCV to automatically find the optimal alpha on the California Housing dataset. These use leave-one-out cross-validation (Ridge) and K-fold (Lasso) internally. Print the best alpha found by each. Compare the test RMSE with optimal alpha vs alpha=0 (no regularisation) and alpha=1000 (too strong). Plot the CV score as a function of alpha for Lasso.

## Exercise 4 — Elastic Net Regularisation Path
On the California Housing dataset, train ElasticNet models sweeping l1_ratio from 0.0 (pure Ridge) to 1.0 (pure Lasso) in steps of 0.1, with alpha=0.5. For each, count the number of non-zero coefficients and record validation RMSE. Plot both on the same graph with two y-axes. At what l1_ratio do coefficients start disappearing? What l1_ratio gives the best RMSE?

## Exercise 5 — Dropout Rate Experiment
On the MNIST dataset, train a 3-layer dense neural network with Dropout rates of 0.0, 0.1, 0.2, 0.5, and 0.8 applied after each hidden layer. For each, plot the training and validation accuracy curves. Identify: at what rate does dropout cause underfitting? What rate gives the best validation accuracy? Note how larger dropout rates require more epochs to converge.

## Exercise 6 — Early Stopping Deep Dive
Train a neural network on the breast cancer dataset with 500 maximum epochs and no early stopping. Plot the training and validation loss and accuracy curves. Identify the epoch where validation accuracy peaks — call this epoch N*. Now train another network with EarlyStopping(patience=20, restore_best_weights=True). Compare the final weights: does restoring best weights matter (i.e., is the model at convergence better than the model at the early-stop epoch)?

## Exercise 7 — L1 vs L2 on Correlated Features
Create a dataset where feature 1 and feature 2 are perfectly correlated (feature2 = feature1 + tiny noise). Train Ridge and Lasso on this dataset. What does Ridge do? (It should split the weight roughly equally between them.) What does Lasso do? (It should pick one and zero out the other — but which one? Run with different random seeds.) Explain why this matters for model interpretability.

## Exercise 8 — Weight Decay in Neural Networks
Implement L2 regularisation (weight decay) in a Keras neural network using kernel_regularizer=regularizers.l2(lambda_val). Train on breast cancer with lambda=0, 0.001, 0.01, 0.1. Plot validation accuracy and the magnitude of weights (compute np.mean(np.abs(layer.get_weights()[0])) for each layer) vs lambda. Confirm that higher lambda produces smaller weights. At what lambda does regularisation start hurting accuracy?

## Exercise 9 — Batch Normalisation as Regularisation
Train a neural network on the California Housing dataset with and without BatchNormalization layers. Compare how much dropout is needed to achieve similar validation performance in each case. Batch normalisation has a regularisation effect that reduces the need for explicit dropout. Verify this by finding the minimum dropout rate needed (with and without BatchNorm) to prevent validation loss from exceeding training loss by more than 5%.

## Exercise 10 — Full Regularisation Audit
Take the overfit network from Exercise 5 of the Neural Networks lesson (the 512-neuron wide network trained on only 200 samples). Apply a combination of regularisation techniques simultaneously: L2 weight decay (0.001), Dropout(0.4) after each dense layer, and BatchNormalization between the dense and activation layers. Use early stopping with patience=20. Document the improvement in validation accuracy and the reduction in the train-val accuracy gap. Write a paragraph summarising which regularisation technique had the most impact for this specific problem.
