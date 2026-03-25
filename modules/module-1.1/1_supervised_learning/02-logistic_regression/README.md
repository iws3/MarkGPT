# Logistic Regression

## Fundamentals

Logistic Regression is a fundamental supervised learning algorithm used for binary and multi-class classification problems. Despite its name containing "regression," it is primarily a classification algorithm that models the probability of a sample belonging to a particular class. It extends linear regression by applying a sigmoid function to map continuous outputs to probabilities between 0 and 1. Logistic regression is interpretable, computationally efficient, and serves as the foundation for understanding more complex classification techniques like deep neural networks. It's widely used in medical diagnosis, spam detection, credit risk assessment, and customer churn prediction.

## Key Concepts

- **Hypothesis**: $P(y=1|x) = \frac{1}{1 + e^{-z}}$ (Sigmoid function)
- **Cost Function**: Binary Cross-Entropy Loss
- **Decision Boundary**: Probability threshold (typically 0.5)
- **Multi-class Extension**: One-vs-Rest or Softmax

## Applications

- Medical diagnosis (disease detection)
- Spam email detection
- Customer churn prediction
- Credit risk assessment
- Default probability estimation

## When to Use

Use logistic regression when:
- Binary or multi-class classification is needed
- Interpretability is important
- You need probability estimates, not just predictions
- Data is approximately linearly separable

## When NOT to Use

- Highly non-linear classification boundaries
- Very high-dimensional data without dimensionality reduction
- When you need to capture complex feature interactions

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### The Sigmoid Function and Probability

Unlike linear regression which predicts continuous values, logistic regression is designed for binary classification problems. The core of logistic regression is the sigmoid function, defined as σ(z) = 1 / (1 + e^(-z)), which maps any input to a value between 0 and 1. This output is interpreted as the probability that an instance belongs to the positive class. The sigmoid function has a smooth S-shaped curve that ensures the probability estimates are always bounded between 0 and 1, making it ideal for modeling classification probabilities. The decision boundary is typically set at 0.5, where instances with probability above 0.5 are classified as positive and those below are classified as negative.

### Cross-Entropy Loss and Cost Function

While logistic regression models probabilities, the cost function used for optimization is different from linear regression. The cross-entropy loss (also called log loss) is defined as -[y*log(h(x)) + (1-y)*log(1-h(x))], where y is the true label and h(x) is the predicted probability. This cost function has several desirable properties: it penalizes confident but incorrect predictions heavily, rewards confident correct predictions, and creates a convex optimization landscape. The logarithmic scale ensures that as the predicted probability moves away from the true label, the penalty increases exponentially. This property makes the cross-entropy loss particularly effective for training classification models as it naturally encourages the model to produce well-calibrated probability estimates.

### Multiclass Classification Extension

While binary logistic regression handles two-class problems, many real-world applications require classifying instances into more than two classes. Multiclass logistic regression extends the binary case using the softmax function, which generalizes the sigmoid function to multiple classes. The softmax function computes probabilities for each class such that they sum to 1, allowing the model to rank all classes and select the one with the highest probability. The cost function for multiclass problems becomes categorical cross-entropy, computed as -Σ(y_i * log(p_i)) where y_i is the true label indicator and p_i is the predicted probability for class i. This framework enables logistic regression to handle complex multi-class problems while maintaining the interpretability of probability estimates.

### Practical Implementation Considerations

Implementing logistic regression requires careful attention to several practical aspects beyond the theoretical foundations. Feature engineering remains critical; categorical variables must be encoded, and continuous features should be scaled to improve convergence. Class imbalance, where one class is significantly more prevalent than others, can bias the model toward the majority class. Techniques like stratified sampling, class weights, and resampling can address this issue. Additionally, threshold selection for classification is not always 0.5; depending on the application, moving the decision threshold can optimize metrics like precision, recall, or F1-score. Finally, regularization strategies apply to logistic regression just as they do to linear regression, and practitioners should use techniques like L1 or L2 regularization to prevent overfitting and improve model generalization.

### Probability Calibration and Threshold Selection

Logistic regression outputs probabilities that are inherently calibrated; predicted probability p that a sample belongs to the positive class can be directly interpreted. Modern implementations (scikit-learn, TensorFlow) compute probabilities via softmax for multi-class. However, models are often overconfident: predicted probabilities don't match empirical frequencies. Calibration curves show this mismatch; well-calibrated models have calibration curves on the diagonal. Platt scaling and isotonic regression post-process predictions for better calibration. The decision threshold (usually 0.5) can be adjusted: lowering it increases recall at the cost of precision. ROC curves visualize precision-recall trade-offs; optimal threshold depends on domain. In imbalanced datasets (rare positive class), moving the threshold reduces false negatives, crucial in medical diagnosis where missing disease is costly.

### Handling Imbalanced Data

Imbalanced datasets (e.g., 1% positive, 99% negative) are common in fraud detection, disease diagnosis, and rare event prediction. Standard logistic regression is biased toward the majority class; a model predicting always negative achieves 99% accuracy despite uselessness. Solutions include: (1) class weighting, penalizing minority class errors more; (2) resampling via oversampling minorities or undersampling majorities; (3) synthetic data generation (SMOTE); (4) appropriate metrics like F1-score, precision-recall AUC instead of accuracy. Class weights via `class_weight='balanced'` in scikit-learn inverse the class frequencies in loss computation. Stratified k-fold ensures each fold maintains class proportions. SMOTE generates synthetic minority examples via interpolation between existing examples and k-nearest neighbors. Cost-sensitive learning assigns different misclassification costs to classes.

### Multi-class Logistic Regression in Practice

Multi-class logistic regression extends binary through one-vs-rest or multinomial approaches. One-vs-rest trains |K| binary classifiers, each separating one class from others; predictions come from classifier with highest confidence. Multinomial logistic regression (softmax regression) directly models all classes jointly via softmax output, theoretically superior. Predictions are the class with highest softmax probability. In scikit-learn, `LogisticRegression(multi_class='multinomial')` handles this. Rare classes can disappear in one-vs-rest but are considered in multinomial. For balanced classes, both approaches perform similarly; multinomial is preferred for imbalanced multi-class. Interpreting coefficients in multi-class requires care: coefficients should be compared within the one-vs-rest classifier for a specific class, not across classifiers.

### Logistic Regression vs Neural Networks

Logistic regression is a special case of neural networks with no hidden layers. Given neural network flexibility, when is logistic regression preferable? Logistic regression is interpretable: coefficients directly indicate feature impact and directions. Explainability is crucial in regulated domains (finance, healthcare). Logistic regression requires less data and computational power; neural networks need substantial data to avoid overfitting. Logistic regression's simplicity enables faster training, deployment, and debugging. Features for logistic regression should be engineered carefully, handling non-linear relationships explicitly. Neural networks learn features automatically but require tuning architecture and hyperparameters. For problems with clear linear/simple non-linear separability, logistic regression often matches neural network performance. Logistic regression serves as an excellent baseline; if neural networks only marginally outperform, logistic regression's simplicity wins. In production, simpler models reduce maintenance burden and debugging complexity.

### Practical Implementation Tips

Scaling is critical for logistic regression with regularization; StandardScaler centers and scales features to zero mean and unit variance. Categorical variables should be one-hot encoded; ordinal variables can be treated as continuous if ordering makes sense. In scikit-learn: `LogisticRegression(solver='lbfgs', max_iter=1000)` uses L-BFGS optimizer. For large datasets, 'saga' or 'sag' solvers are more efficient. The C parameter (inverse of regularization strength) controls regularization; larger C = less regularization. Probability estimates come from `predict_proba()`, useful for downstream decision-making. Feature importance can be inferred from coefficient magnitudes (after scaling). Regularization parameter tuning via cross-validation: plot cross-validation scores across C values to find the sweet spot. Always validate on completely held-out test data; data leakage from validation into training ruins generalization estimates. Confusion matrices reveal class-specific performance; recall and precision might differ substantially.