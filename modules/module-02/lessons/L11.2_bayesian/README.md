# Bayesian Methods and Probabilistic ML
## Comprehensive Learning Guide

## Bayes' Theorem and Conditional Probability

Conditional probability P(A|B) measures probability of A given B occurred.

Bayes' theorem relates forward and reverse: P(A|B) = P(B|A) × P(A) / P(B).

Priors encode initial beliefs before seeing data influencing inference.

Likelihood shows how probable data is for fixed parameters.

## Bayesian Inference and Computation

Computing posteriors exactly is impossible due to intractable integrals.

Markov Chain Monte Carlo samples from posterior distributions successfully.

Gibbs sampling specializes MCMC when conditional distributions are tractable.

Variational inference approximates posteriors faster but with approximation bias.

## Applications and Advantages

Bayesian networks use directed graphs to represent conditional dependencies.

Bayesian methods shine in small-data regimes with limited samples.

Decision making under uncertainty is fundamentally Bayesian in frameworks.

The Bayesian vs. Frequentist debate reflects different philosophical approaches.


## Bayesian Model Selection

Model comparison uses Bayes factors for relative evidence.

Marginal likelihood quantifies model fit accounting for complexity.

Cross-validation estimates predictive performance.

Pareto smoothed importance sampling estimates without refitting.

Widely Applicable Information Criterion (WAIC) approximates posterior.

Leave-One-Out Cross-Validation efficiently estimates predictive power.


## Bayesian Computation Advanced

Hamiltonian Monte Carlo explores high-dimensional posteriors.

No-U-Turn Sampler (NUTS) adaptively sets trajectory length.

Automatic Differentiation Variational Inference (ADVI) scales variational.

Expectation-Propagation provides deterministic approximation.

Assumed Density Filtering processes streams of data.

