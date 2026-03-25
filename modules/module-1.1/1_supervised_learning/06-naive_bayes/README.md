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

### Continuous vs Discrete Features

Gaussian Naive Bayes assumes continuous features follow Gaussian distributions within classes: P(x|class) ~ N(μ_class, σ_class). Parameters μ and σ are estimated from training data. For each class, mean and variance of each feature are computed; predictions use these Gaussians. An alternative for discrete features: assume multinomial distributions (feature values are counts). Binary features use Bernoulli distributions. Mixing feature types (some discrete, some continuous) requires different distributions. Kernel density estimation can model continuous features non-parametrically (no Gaussian assumption), but adds computational cost. In practice, discretizing continuous features (binning) enables using discrete Naïve Bayes, avoiding Gaussian assumptions that might not hold. Feature transformation (log-transform for skewed features) can better satisfy Gaussian assumptions. The choice depends on data: visualize feature distributions within classes; if Gaussian is a reasonable fit, Gaussian Naive Bayes is appropriate.

### Advantages in Low-Data Regimes

Naive Bayes requires minimal data: with n samples and p features, estimating parameters needs only p values per class (p * |C| parameters total). In contrast, discriminative models often require O(n) samples; complex models worse. This makes Naive Bayes valuable when data is scarce. In medical diagnosis with rare diseases, Naive Bayes can work with few positive examples. In new domains, Naive Bayes quickly gets reasonable results. The naive independence assumption, despite being false, actually helps: by reducing parameters, it reduces variance. This is similar to regularization: adding bias to reduce variance. As more data becomes available, less-naive models (capturing feature dependencies) typically improve. But Naive Bayes remains competitive on limited data. This data efficiency explains its popularity in emerging applications; practitioners can prototype quickly without large data collection efforts.

### Practical Tuning and Limitations

Naive Bayes has few hyperparameters: Laplace smoothing strength α is the main one. Prior class probabilities P(class) can be uniform or reflect training data class proportions (`prior` parameter in scikit-learn). For imbalanced data, matching training proportions or using balanced priors helps. Feature selection before Naive Bayes sometimes improves accuracy: irrelevant features add noise, particularly when they're uncorrelated with dependent features (violating the assumption). Naive Bayes doesn't directly handle continuous numerical target values (requires classification, though extensions exist). It assumes feature independence strongly: if features are highly dependent, performance degrades. Word order is ignored in text (bag-of-words), losing sequential information. N-grams (pairs/triples of words) capture some sequential patterns. Overall, Naive Bayes is simple, fast, interpretable, and sometimes surprisingly effective. It's excellent as a baseline; if other algorithms only marginally outperform, the simplicity of Naive Bayes often wins in production.