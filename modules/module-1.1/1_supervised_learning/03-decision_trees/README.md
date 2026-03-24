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