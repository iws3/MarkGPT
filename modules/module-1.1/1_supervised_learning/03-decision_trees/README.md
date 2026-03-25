# Decision Trees

## Fundamentals

Decision Trees are versatile supervised learning algorithms that build tree-like structures to make predictions by recursively partitioning the data based on feature conditions. Unlike linear models, decision trees can capture non-linear relationships and complex feature interactions naturally. The tree structure makes them highly interpretable as they mimic human decision-making processes. Decision trees form the foundation for powerful ensemble methods like Random Forests and Gradient Boosting. They're used across industries for classification, regression, and feature importance analysis, ranging from medical diagnosis to customer segmentation.

## Key Concepts

- **Splitting Criteria**: Gini Impurity, Information Gain (Entropy)
- **Tree Depth**: Controls model complexity
- **Leaf Nodes**: Final decision/prediction
- **Pruning**: Reduces overfitting

## Applications

- Customer segmentation
- Credit risk assessment
- Disease diagnosis
- Feature importance identification
- Business decision rules

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Tree Construction and Splitting Criteria

Decision trees recursively partition the feature space into regions where each region is assigned a class label or prediction. The tree construction process begins with all training data at the root node and recursively selects the feature and split value that best separates the data into homogeneous subsets. Common splitting criteria for classification include Gini impurity and Information Gain based on entropy. Gini impurity measures the probability of misclassifying a randomly chosen element and is defined as 1 - Σ(p_i)², where p_i is the proportion of class i. Information Gain uses entropy, H = -Σ(p_i * log(p_i)), to measure the disorder in a dataset and selects splits that maximize the reduction in entropy. For regression tasks, variance reduction is used as the splitting criterion, selecting splits that minimize the variance within resulting subsets.

### Pruning and Preventing Overfitting

Decision trees have a natural tendency to overfit training data, growing until each leaf node contains only instances of a single class. Pruning is a technique that removes branches from a fully grown tree to improve generalization on unseen data. Cost complexity pruning introduces a regularization parameter that penalizes tree size, creating a relationship between tree complexity and prediction error. Reduced Error Pruning removes nodes if the error on a validation set does not increase or decreases after removal. Setting a maximum tree depth, minimum samples per leaf, and minimum samples for splitting are hyperparameters that control tree growth and can prevent overfitting. Cross-validation helps select optimal values for these hyperparameters by evaluating performance on held-out data.

### Feature Interactions and Interpretability

One of the major advantages of decision trees is their interpretability; the decision rules can be easily extracted and visualized as a tree structure. This makes decision trees valuable in applications where model transparency is important, such as healthcare or finance. Decision trees naturally capture feature interactions without requiring explicit feature engineering, as the model learns hierarchical combinations of conditions. A complex interaction that might require manual feature engineering in linear models emerges naturally through the tree structure. However, this interpretability comes with a cost: decision trees tend to create boundaries parallel to feature axes and can be unstable, where small changes in training data lead to significantly different tree structures.

### Handling Missing Values and Categorical Features

Decision trees handle both categorical and continuous features naturally without requiring extensive preprocessing. For categorical features, binary splits can group categories or make simple left-right decisions. For continuous features, the optimal split point is found by sorting unique feature values and evaluating each potential threshold. Missing values can be handled through surrogate splits, which identify alternative features that produce similar splits when the primary feature is missing. Some implementations use weighted samples to handle missing data, distributing samples that contain missing values down both left and right branches proportionally. This flexibility makes decision trees particularly useful in real-world scenarios where data completeness cannot be guaranteed.

### Gini Impurity vs Entropy: Theoretical and Practical Differences

Gini impurity and entropy are the two main splitting criteria, with subtle but important differences. Gini impurity: G = 1 - Σ(p_i)^2, ranges from 0 (pure) to 1-1/|C| (maximally impure for |C| classes). Entropy: H = -Σ(p_i * log(p_i)), ranges from 0 to log(|C|). Computationally, Gini is faster (no logarithm). Empirically, they produce similar trees; choice rarely matters much. Information gain (entropy-based) sometimes splits differently than Gini gain, particularly with imbalanced classes. Gini is more lenient with minority classes because it's based on squared probabilities; entropy-based information gain is more sensitive to class distribution changes. In practice, trying both via `criterion='gini'` vs `criterion='entropy'` in scikit-learn takes seconds via cross-validation. Most practitioners use Gini (default) due to computational efficiency. For imbalanced data, carefully comparing both is worthwhile.

### Tree Depth and Preventing Overfitting

Fully grown trees overfit by learning training data noise; trees with depth equal to log(n) samples often suffice. The `max_depth` hyperparameter is the most powerful regularization; setting it to 5-20 usually produces reasonable trees. `min_samples_split` (minimum samples to create a child node) and `min_samples_leaf` (minimum samples in leaf) prevent growing trees that separate only few samples. For n=1000 samples, `min_samples_leaf=5` means leaves have at least 5 samples, preventing severe overfitting. `max_features` limits features considered at each split; 'sqrt' is common for classification. Pruning post-hoc via cost-complexity pruning removes subtrees that don't improve validation performance. Cross-validation reveals optimal depths: plot validation error vs. depth; choose depth where validation error minimizes. Generally, training error decreases monotonically with depth, but validation error follows a U-shape: decreasing until depth minimizes bias-variance tradeoff, then increasing as overfitting dominates.

### Feature Interactions and Tree Depth Requirements

Decision trees naturally capture feature interactions without explicit feature engineering. A simple rule like 'buy if (price < $500) AND (rating > 4)' emerges naturally. Tree depth required to model interactions increases exponentially: depth 1 handles linear boundaries, depth 2 handles simple AND/OR combinations, depth 3 handles more complex interactions. For d-dimensional interactions, depth ≈ d is needed; trying to model 10-way interactions requires very deep trees (overfitting risk). This is a limitation: shallow trees can't capture complex interactions. Random Forests (ensemble of shallow trees) handle this better through combining many trees. Another approach: manually create interaction features, allowing shallow trees to use them. Understanding which interactions matter is crucial: domain expertise guides creating meaningful interactions. In text classification, feature interactions might be word pairs; in finance, price×volatility interactions matter for options.

### Handling Missing Values in Trees

Decision trees handle missing values naturally via surrogate splits. When a feature has missing values, the same splitting rule is applied: if feature is missing, samples flow down both left and right branches proportionally. Weights on surrogate splits indicate their quality as substitutes for the primary split. This enables training and prediction with missing data without explicit imputation. Alternatively, treat 'missing' as a category; for numerical features, create a binary indicator of missingness. Some practitioners use mean imputation before fitting; this ignores the information that data was missing. Scikit-learn's tree doesn't directly support missing values (requires handling beforehand). XGBoost and LightGBM natively handle missing data. For missing features during prediction, samples follow surrogate splits learned during training. This flexibility makes trees practical for real-world messy data where missing values are common.