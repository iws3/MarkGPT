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

### The Expectation-Maximization Algorithm

The Expectation-Maximization algorithm is a general framework for maximum likelihood estimation with latent variables. In GMM context, cluster assignments are latent (unknown). The E-step computes the responsibility (posterior probability) that each point belongs to each cluster: γ(z_{ik}) = π_k·N(x_i|μ_k, Σ_k) / Σ_j π_j·N(x_i|μ_j, Σ_j). The M-step updates parameters: π_k ← 1/n·Σ γ(z_{ik}), μ_k ← Σ γ(z_{ik})·x_i / Σ γ(z_{ik}), and Σ_k is updated similarly. EM alternates between E and M steps with guaranteed monotonic increase in likelihood, converging to a local maximum. EM provides soft probabilistic assignments during and after training, contrasting with k-means' hard assignments. This probabilistic treatment allows expressing clustering uncertainty and using probability-based selection criteria.

### Model Selection and Covariance Constraints

Selecting the number of components (equivalent to selecting k) uses information criteria. The Bayesian Information Criterion (BIC) is BIC = -2 log L + p log n, balancing likelihood fit with model complexity (p is parameter count). Akaike Information Criterion (AIC) similarly balances fit and complexity with different penalties. Lower BIC/AIC indicates better models, so these metrics can be computed for different k values and the best k selected. The problem of GMM over-parameterization (unrestricted covariance matrices have O(k·d²) parameters) can be addressed by constraining covariance structure. Full covariance matrices allow Σ_k = [arbitrary symmetric positive-definite matrix] (most flexible). Diagonal covariance assumes independence between features within clusters. Spherical covariance assumes equal variance in all directions Σ_k = σ²_k·I. These constraints reduce parameters and regularize against overfitting.