# Ensemble Methods

## Fundamentals

Ensemble Methods combine multiple weak or strong learners to create a powerful predictor, often outperforming any individual model. The key principle is diversity: different models make different mistakes, and by combining them strategically, the ensemble can achieve better generalization and robustness. Major ensemble techniques include voting, averaging, stacking, and blending. Ensemble methods have dominated machine learning competitions for decades and are standard practice in industry for improving model robustness and reducing overfitting. Understanding ensemble principles is fundamental to building state-of-the-art prediction systems.

## Key Concepts

- **Voting**: Majority or soft voting
- **Stacking**: Meta-learner combining base learners
- **Blending**: Train-test split for meta-features
- **Diversity**: Different model types or hyperparameters

## Applications

- Competition-winning solutions
- High-stakes prediction systems
- Robust production models
- Uncertainty estimation
- Hybrid modeling

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Ensemble Learning Principles

Ensemble methods combine multiple weak learners to create a strong learner, leveraging the principle that diverse models make better collective predictions. Errors from individual models may be uncorrelated; combining predictions can cancel out individual errors. This requires diversity: if models make the same mistakes, combining them provides no benefit. Ensemble diversity comes from different algorithms, different training data samples (bagging), or sequential training to correct prior mistakes (boosting). Voting (classification) or averaging (regression) aggregates predictions. More sophisticated methods use stacking, where predictions from base learners train a meta-learner, which makes final predictions. Theoretical bounds show ensemble generalization error decreases with diversity and individual learner accuracy.

### Bagging and Bootstrap Aggregating

Bagging trains multiple models on different random samples (with replacement) from training data. Each bootstrap sample has approximately 63% unique examples; the remaining 37% appear multiple times or not at all. Models trained on different samples learn slightly different patterns, creating diversity. Aggregating predictions through averaging or voting reduces variance while maintaining similar bias. Random forests extend bagging to decision trees with additional feature randomization. Bagging is particularly effective for high-variance learners like deep trees. Out-of-bag samples provide internal validation without separate test sets. Bagging requires models to be trainable independently; models are trained in parallel, making bagging computationally efficient for parallelization.