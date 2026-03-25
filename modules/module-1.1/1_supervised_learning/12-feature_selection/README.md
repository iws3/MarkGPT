# Feature Selection

## Fundamentals

Feature Selection is the process of selecting a subset of relevant features from the original feature set to improve model performance, reduce overfitting, and decrease computational cost. Feature selection can be based on statistical tests, model-based importances, or search algorithms. Effective feature selection improves interpretability by focusing on the most important predictors and can prevent the curse of dimensionality. Feature selection is a critical preprocessing step in machine learning pipelines, especially when working with high-dimensional data from text, images, or sensor networks.

## Key Concepts

- **Filter Methods**: Statistical tests independent of model
- **Wrapper Methods**: Use model performance for selection
- **Embedded Methods**: Feature selection during training
- **Univariate Selection**: Individual feature evaluation
- **Multivariate Selection**: Feature interaction consideration

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Why Feature Selection Matters

Feature selection (identifying relevant features) improves model generalization, reduces computational cost, and enhances interpretability. Irrelevant features add noise without useful information, increasing model variance. High-dimensional data suffers from the curse of dimensionality: in high dimensions, distances become meaningless and models require exponentially more data. Removing irrelevant features combats this. In interpretability, fewer features create simpler, understandable models easier to debug and trust. In privacy-sensitive applications, using fewer features is better; fewer accessible features reduce privacy risk. Feature selection is distinct from dimensionality reduction; selection chooses subsets of original features while reduction combines features.

### Filter and Wrapper Methods

Filter methods rank features based on statistical properties independent of the learning algorithm. Mutual information measures dependency between feature and target. Correlation measures linear relationships; high absolute correlation suggests relevance. Information gain measures entropy reduction from feature splits. Univariate filters are fast but ignore feature interactions. Wrapper methods train models with different feature subsets, ranking subsets by model performance. This accounts for algorithm-specific preferences but is computationally expensive. Forward selection starts with no features, adding features that most improve performance. Backward selection starts with all features, removing features that least hurt performance. Wrapper methods often select better feature subsets but require many model trainings.

### Embedded Methods and Feature Importance

Embedded methods select features during training. L1 regularization (lasso) naturally performs feature selection by driving irrelevant weights to zero. Tree-based methods implicitly select features by building trees; information gain is used in splits. Feature importance measures how much each feature contributes to predictions. In tree models, importance sums the information gain from splits using each feature. In linear models, importance relates to coefficient magnitudes. In neural networks, gradient-based importance measures sensitivity to feature perturbations. Permutation importance measures performance loss when feature values are randomly shuffled. SHAP values provide game-theoretic importance connecting predictions to individual features.

### Practical Feature Selection Guidelines

Feature selection strategy depends on goals. For interpretability, wrapper methods and embedded methods identifying features with strong individual impact are preferred. For dimensionality reduction before training complex models, filter methods provide speed. For maximum performance, embedded methods often excel by jointly optimizing with the model. Domain knowledge guides feature selection; subject matter experts should inform which features are meaningful. Temporal ordering matters; avoid using future information to predict the past. Multicollinearity (correlated features) is problematic; selecting among correlated features requires care. Feature selection is not a replacement for careful data collection; good features require meaningful measurements. Validation on hold-out data ensures selected features generalize; selection bias occurs if feature selection uses test data.