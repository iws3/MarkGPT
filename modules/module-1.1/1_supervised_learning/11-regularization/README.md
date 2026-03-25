# Regularization Techniques

## Fundamentals

Regularization is a fundamental technique in machine learning that prevents overfitting by adding a penalty term to the loss function that discourages complex models. Regularization methods constrain model weights to stay small, promoting simpler solutions that generalize better to unseen data. Common regularization techniques include L1 (Lasso), L2 (Ridge), Elastic Net, Dropout, and Early Stopping. Understanding regularization is crucial for building production models that perform well on test data. Regularization is not specific to a single algorithm but applies across linear models, neural networks, and tree-based methods.

## Key Concepts

- **L1 Regularization**: Lasso, feature selection
- **L2 Regularization**: Ridge, weight decay
- **Elastic Net**: Combination of L1 and L2
- **Dropout**: Neural network regularization
- **Decay Parameters**: Lambda, alpha tuning

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)


### Overfitting and the Bias-Variance Tradeoff

Overfitting occurs when models learn training data too well, including noise, leading to poor performance on new data. The bias-variance decomposition quantifies this: Total Error = Bias^2 + Variance + Noise. Bias measures how well the model structure can capture the true relationship; high bias means even with infinite data, performance is limited (underfitting). Variance measures sensitivity to training data variations; high variance means small data changes lead to very different models (overfitting). More complex models have lower bias but higher variance. Regularization bias the model slightly toward simpler solutions, trading small bias increases for larger variance reductions, often improving overall performance. The optimal complexity balances bias and variance; this optimal point is rarely at maximum accuracy on training data.

### L1 and L2 Regularization Techniques

Regularization adds penalty terms to the loss function. L2 regularization (ridge) adds λ·||w||^2, penalizing large weights. This encourages smaller, more diffuse weights, leading to smoother decision boundaries. L1 regularization (lasso) adds λ·||w||_1 (sum of absolute values), which encourages sparsity by pushing some weights exactly to zero, effectively performing feature selection. Elastic Net combines both: (1-α)·||w||^2 + α·||w||_1. The regularization strength λ controls the tradeoff; larger λ creates stronger regularization. Early stopping stops training when validation error stops improving, another form of regularization. Dropout (randomly deactivating neurons) regularizes neural networks. These techniques apply across algorithm families, making them universally important.

### Dropout and Modern Regularization

Dropout in neural networks randomly deactivates neurons with probability p during training, forcing networks to learn redundant representations. At test time, all neurons are active but weights are scaled. Dropout prevents co-adaptation, where neurons rely on specific other neurons, reducing generalization. Batch normalization normalizes layer outputs, providing regularization effects alongside computational benefits. Layer normalization and other normalization techniques serve similar purposes. Data augmentation creates variations of training examples (rotating images, paraphrasing text), increasing effective training set size without new data.

### Cross-Validation and Hyperparameter Selection

Proper regularization hyperparameter selection requires validation. k-fold cross-validation divides data into k folds; each fold serves as validation while others train. This provides more robust performance estimates than single train/test splits. Nested cross-validation uses outer loops for final performance estimation and inner loops for hyperparameter selection, avoiding selection bias. Regularization paths show performance across regularization values; visualization helps understand optimal points. Grid search exhaustively evaluates hyperparameter combinations while random search samples randomly. Bayesian optimization uses performance history to intelligently select promising hyperparameter values. Selecting regularization parameters requires balancing computational cost against robustness; more computation (larger k, finer grids) provides better estimates but increased cost.