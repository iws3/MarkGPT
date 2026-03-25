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