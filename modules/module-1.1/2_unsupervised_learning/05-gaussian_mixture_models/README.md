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

### Advantages and Limitations

GMM advantages include probabilistic interpretation, soft cluster assignments enabling uncertainty quantification, and principled model selection through likelihood. GMM is well-suited for model-based clustering where underlying data generation matches Gaussian mixture assumptions. The soft assignments are useful for applications requiring confidence or where hard assignments are inappropriate. EM convergence guarantees local optima though not global optima, and multiple random initializations often improve results. However, GMM is sensitive to initialization and may converge to poor local optima. Computational complexity is O(k·d²·n·iterations), higher than k-means for complex covariance structures. GMM assumes Gaussian cluster shapes; non-Gaussian clusters may be poorly modeled. For very high-dimensional data, covariance estimation becomes unstable. Despite limitations, GMM provides a principled probabilistic framework valuable in applications where soft cluster assignments and likelihood-based model selection are important.

### EM Algorithm Convergence and Local Maxima

The Expectation-Maximization algorithm monotonically increases likelihood but converges to local maxima (not guaranteed global). Starting from poor initializations, EM can converge to poor solutions. Multiple random restarts are necessary: run EM from 10 different initializations, select maximum likelihood result. Poor solutions are identified by low likelihood compared to restarts or fit quality. Tuning number of iterations: EM usually converges quickly (10-50 iterations). Monitoring log-likelihood reveals convergence: stable log-likelihood over iterations indicates convergence. Early stopping criteria (tolerance on likelihood change) prevent unnecessary iterations. Scaled versions of data improve numerical stability; features with different scales cause numerical issues. Responsibility values (soft assignments) should be checked: if a component has near-zero assigned samples, that component isn't learning (degenerate). Careful initialization via k-means preprocessing helps avoid degeneracy.

### Determining Optimal Number of Components

Selecting k (number of Gaussian components) uses information criteria. Bayesian Information Criterion (BIC) balances likelihood (fit) and complexity (parameter count): BIC = -2 * log(L) + p * log(n). Lower BIC indicates better fit relative to complexity. Akaike Information Criterion (AIC) uses: AIC = -2 * log(L) + 2 * p. AIC penalizes complexity less than BIC; for large n, BIC is preferred. Cross-validation: split data, train on k-1 folds, evaluate on k-th fold via likelihood. Averaging across folds provides robust estimate; choose k minimizing cross-validation error. Unlike k-means (no natural likelihood), GMM's probabilistic framework enables principled model selection. Plotting BIC/AIC/cross-validation error vs k reveals elbow; this elbow suggests optimal k. Running grid search over k values (e.g., k in 1:20) with cross-validation is standard practice.

### Covariance Matrix Constraints and Regularization

GMM with full covariance matrices (unconstrained) has O(k * d²) parameters; for d = 100 and k = 10, this is thousands of parameters. With limited data, this overfits. Constraints reduce parameters: spherical covariance (σ² * I, same variance all directions) has k parameters. Diagonal covariance (σ²_d per dimension) has k * d parameters. Full covariance (most flexible) has k * d * (d+1) / 2 parameters. Constraints are selected based on data and computational resources. For d > 100 (high-dimensional), diagonal or spherical is necessary. Regularization: adding small constant to diagonal prevents singular covariance matrices (numerical issues). In scikit-learn, covariance_type parameter selects constraint. Empirically trying multiple types via cross-validation finds best fit.

### Probabilistic Interpretation and Soft Clustering

Unlike k-means (hard assignments), GMM provides soft assignments: each sample has probability vector over components. This probability represents uncertainty: probability [0.9, 0.05, 0.05] means sample is likely component 1 but somewhat uncertain. Soft assignments are valuable for uncertainty quantification, probabilistic inference, and downstream applications requiring confidence. Samples on cluster boundaries get intermediate probabilities. This inherent uncertainty is useful: critical decisions (medical diagnosis) use high-confidence predictions, uncertain cases warrant manual review. Soft assignments enable smooth boundaries: moving a sample gradually through space results in gradual probability changes (unlike k-means discontinuity). For generative modeling: GMM defines a distribution; sampling from GMM creates synthetic data. Component selection (sample from component probability) followed by Gaussian sampling generates realistic data.

### Applications: Density Estimation and Anomaly Detection

GMM's probabilistic framework enables density estimation: P(x) = Σ π_k * N(x | μ_k, Σ_k), fully specifying data distribution. This enables likelihood computation for any sample; likelihood quantifies how typical the sample is. Anomalies have low likelihood. This is a principled approach to anomaly detection versus k-means clustering. In quality control, product features follow a distribution (normal operation); products with low likelihood under learned GMM are defective. In cybersecurity, network traffic patterns are learned via GMM; attacks have low likelihood. This probabilistic approach is more theoretically grounded than distance-based methods. GMM also enables missing data imputation: conditional expectations given partial observations impute missing values. For example, imputing missing gene expression values from partially observed samples. Overall, GMM's probabilistic nature enables sophisticated downstream tasks beyond clustering.