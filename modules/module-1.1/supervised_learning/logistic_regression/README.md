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

