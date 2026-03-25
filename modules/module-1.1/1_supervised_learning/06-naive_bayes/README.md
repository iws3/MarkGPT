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

### The Naive Conditional Independence Assumption

The naive assumption is that features are conditionally independent given the class label, meaning P(x₁, x₂, ..., xₚ|y) = ∏P(xᵢ|y). This assumption is rarely true in practice—features are often correlated—but it drastically simplifies the model and computation. Instead of estimating a complex high-dimensional distribution, we only need to estimate p univariate distributions. For each feature and class combination, we estimate P(xᵢ|y) from training data. For discrete features, this is simply the proportion of class y examples where feature i takes value xᵢ. For continuous features, we typically assume a Gaussian distribution and estimate the mean and variance for each feature-class combination. The strong independence assumption, while often violated, frequently leads to effective classifiers because it reduces variance through model simplification, sometimes outperforming more complex models despite the unrealistic assumptions.

### Gaussian, Multinomial, and Bernoulli Variants

The choice of how to model feature distributions leads to different Naive Bayes variants. Gaussian Naive Bayes assumes continuous features follow a normal distribution, computing P(xᵢ|y) = (1/√(2π·σ²ᵧᵢ))·exp(-(xᵢ-μᵧᵢ)²/(2σ²ᵧᵢ)). Multinomial Naive Bayes is designed for discrete count data and is commonly used in text classification, where features represent word frequencies or term frequencies. Bernoulli Naive Bayes is used when features are binary (present or absent), appropriate for document classification with binary feature vectors indicating word presence. Each variant models feature distributions appropriately for the data type, yet all follow the same classification principle of selecting the class with maximum posterior probability. Understanding which variant matches your data type and problem domain is crucial for effective implementation.

### Advantages, Limitations, and Practical Applications

Naive Bayes offers several practical advantages: it is computationally efficient, training is linear in the number of features and training examples, and it requires relatively small amounts of training data compared to more complex models. The algorithm produces calibrated probability estimates and naturally handles missing data by ignoring missing features during inference. However, Naive Bayes struggles with correlated features due to the independence assumption and may underperform complex models on problems with intricate feature interactions. The assumption that class priors remain constant can be problematic with imbalanced datasets. Despite its simplicity, Naive Bayes remains highly effective for text classification, spam detection, and sentiment analysis. Many practitioners use it as a baseline model; if more sophisticated algorithms achieve only marginally better performance, the simplicity and interpretability of Naive Bayes make it the preferred choice.

### Laplace Smoothing and Handling Zero Probabilities

A critical issue in Naive Bayes: if a feature value never appears with a class in training data, P(feature_value|class) = 0. This causes problems: multiplication by zero makes posterior zero regardless of other features. Laplace smoothing adds a constant α to numerator and α*|values| to denominator: P(value|class) = (count + α) / (total + α*|values|). With α=1, it's called Laplace smoothing; α > 1 is Lidstone smoothing. This prevents zero probabilities and accounts for unseen feature-class combinations. Even with Laplace smoothing, rare feature values in training might not appear with certain classes; smoothing assumes uniform priors for unseen combinations. For categorical features with many values (e.g., vocabulary in text classification), smoothing is essential. The strength α is a hyperparameter: α=1 is common; sometimes α ∈ {0.01, 0.1, 1, 10} is tested via cross-validation. Without smoothing, single missing combinations can ruin classification.

### Naive Bayes for Text Classification

Naive Bayes is canonical for text classification (spam detection, sentiment analysis, topic classification). Text is vectorized via TF-IDF (term frequency-inverse document frequency) or one-hot encoding (word presence). Each word is a feature; word frequencies or presence/absence are values. Naive Bayes assumes word independence given class (violates reality but works well). Multinomial Naive Bayes models word counts; Bernoulli Naive Bayes models word presence. For spam detection: P(spam | words) ∝ P(words | spam) * P(spam). Words like 'buy', 'free', 'prize' have high P(word | spam), leading to spam classification. Laplace smoothing handles new words (not in training) by assigning them base probability. Feature engineering (removing stop words, stemming, n-grams) improves performance. Naive Bayes is fast, requires little data, and is highly interpretable: feature probabilities directly show information flow. These advantages make it dominant in text applications.