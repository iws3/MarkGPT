# Naive Bayes

## Fundamentals

Naive Bayes is a probabilistic classifier based on Bayes' theorem with the assumption of feature independence conditional on the class label. Despite this strong assumption often being violated in practice, Naive Bayes is surprisingly effective due to its simplicity, computational efficiency, and strong bias that can lead to good generalization. It's particularly suited for high-dimensional data and categorical features, making it the go-to algorithm for text classification, spam detection, and sentiment analysis. The algorithm's interpretability comes from its probabilistic framework, where model decisions can be explained through probability calculations.

## Key Concepts

- **Bayes' Theorem**: Posterior probability calculation
- **Conditional Independence**: Naive assumption
- **Feature Likelihood**: Probability of features given class
- **Variants**: Multinomial, Gaussian, Bernoulli

## Applications

- Spam email detection
- Sentiment analysis
- Text classification
- Spam filtering
- Medical diagnosis

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Bayes' Theorem and Probabilistic Classification

Naive Bayes classifiers are based on Bayes' theorem, which describes how to update probabilities based on evidence. Bayes' theorem states: P(y|x) = P(x|y)·P(y) / P(x), where P(y|x) is the posterior probability of class y given features x, P(x|y) is the likelihood, P(y) is the prior probability, and P(x) is the evidence. For classification, we want to find the class y that maximizes P(y|x). Since P(x) is constant across classes, we need to maximize P(x|y)·P(y). The prior P(y) is estimated from the proportion of training examples in each class. The likelihood P(x|y) requires computing the joint probability of all features given a class, which becomes problematic in high dimensions due to sparse data. This is where the naive assumption becomes crucial.