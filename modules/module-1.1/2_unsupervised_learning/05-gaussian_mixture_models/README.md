# Gaussian Mixture Models (GMM)

## Fundamentals

GMM is a probabilistic clustering model that assumes data is generated from a mixture of Gaussian distributions. GMM provides soft assignments (probabilities) rather than hard cluster assignments, and naturally handles uncertainty and outliers. GMM is more flexible than K-Means and can model elliptical clusters. Maximum likelihood estimation with the EM algorithm makes GMM powerful for complex data distributions.

## Key Concepts

- **Mixture Components**: K Gaussian distributions
- **Soft Assignments**: Probability of belonging to each cluster
- **Expectation-Maximization**: Iterative optimization
- **Covariance Types**: Spherical, tied, diagonal, full
- **Model Selection**: BIC, AIC criteria

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Probabilistic Clustering and Gaussian Mixture Models

Gaussian Mixture Models (GMM) treat clustering as a probabilistic problem where data is assumed to come from a mixture of k Gaussian distributions. Each cluster is represented by a Gaussian with parameters (mean, covariance) and a mixing weight (proportion of data from that Gaussian). Unlike k-means which assigns points hard cluster membership, GMM provides soft assignments—a probability that each point belongs to each cluster. The likelihood function for GMM is p(X|θ) = Π Σ π_k·N(x_i|μ_k, Σ_k), where π_k are mixing weights, μ_k are means, and Σ_k are covariances. Maximum likelihood estimation of parameters is performed via Expectation-Maximization (EM), which iterates between computing expected cluster membership (E-step) and updating parameters (M-step). GMM provides a principled probabilistic framework for clustering, including uncertainty quantification through cluster membership probabilities.