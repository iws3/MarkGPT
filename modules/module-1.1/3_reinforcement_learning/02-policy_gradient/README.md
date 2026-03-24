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



### Policy Parameterization and Gradient Estimation

Policy gradient methods directly optimize the policy function π(a|s; θ) parameterized by θ (often neural network weights) to maximize expected cumulative reward J(θ) = E_π[Σ γ^t r_t]. Rather than learning state-action values, we learn the policy directly. The policy gradient theorem establishes: ∇_θ J(θ) = E[∇_θ log π(a|s; θ) Q(s, a)], relating policy gradient to Q-values. This gradient indicates how to adjust policy parameters to increase probability of better actions (high Q(s, a)) and decrease probability of worse actions (low Q(s, a)). Policy parameterization using neural networks is flexible: discrete action spaces use softmax output for action probabilities, continuous spaces use Gaussian distributions with mean and variance networks. The log probability derivative ∇_θ log π(a|s; θ) is easily computed; for softmax outputs it reduces to feature vectors minus expected features under the policy.