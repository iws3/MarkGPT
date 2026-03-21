# Policy Gradient Methods

## Fundamentals

Policy Gradient methods directly optimize the policy by ascending the gradient of expected cumulative reward. Unlike Q-Learning (value-based), policy gradient methods parameterize the policy and learn it directly. REINFORCE and Actor-Critic (policy-critic) are popular variants. Policy gradient methods naturally handle stochastic and continuous action spaces.

## Key Concepts

- **Policy Parameterization**: Neural network policy
- **Policy Gradient Theorem**: Gradient of expected return
- **REINFORCE Algorithm**: Monte Carlo policy gradient
- **Baseline Subtraction**: Variance reduction
- **On-Policy Learning**: Learning from current policy

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)

