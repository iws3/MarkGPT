# Actor-Critic Methods

## Fundamentals

Actor-Critic combines policy-based (actor) and value-based (critic) approaches. The actor learns the policy while the critic estimates state values, providing low-variance gradient estimates. Actor-Critic is on-policy and bridges policy gradient and temporal difference methods. Variants like A3C and PPO are state-of-the-art for continuous control.

## Key Concepts

- **Actor**: Policy network parameterizing behavior
- **Critic**: Value network estimating state values
- **Advantage Function**: Actor update signal
- **Temporal Difference Error**: Critic update signal
- **Synchronous vs Asynchronous**: A3C parallelization

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Actor-Critic Architecture

Actor-Critic methods combine policy-based and value-based learning by maintaining two components: an actor that learns the policy π(a|s; θ) and a critic that learns the value function V(s; φ). The actor generates actions based on the learned policy; the critic provides feedback through temporal difference (TD) errors: δ_t = r_t + γV(s_{t+1}; φ) - V(s_t; φ). The TD error indicates whether the reward was better or worse than expected: positive error suggests the action was better than average and should be encouraged, negative error suggests it was worse. Policy gradients use advantage estimates from the critic: ∇_θ J(θ) ≈ ∇_θ log π(a|s; θ) · δ_t. The actor updates using policy gradient with advantage from the critic. The critic updates using TD learning: φ ← φ + β·δ_t·∇_φ V(s; φ). This combination provides lower-variance gradient estimates than pure policy gradients (critic reduces variance) while maintaining faster convergence than pure value-based methods (policy gradient provides better direction).

### Advantage Actor-Critic and Asynchronous Methods

Advantage Actor-Critic (A2C) generalizes basic actor-critic with more sophisticated advantage estimation. Generalized Advantage Estimation (GAE) provides a principled way to estimate advantages: A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}. The parameter λ ∈ [0, 1] interpolates between TD advantage (λ=0, high bias, low variance) and full-return advantage (λ=1, low bias, high variance). Intermediate values balance bias and variance. Asynchronous Advantage Actor-Critic (A3C) parallelizes learning by running multiple agents in parallel environments, each with their own actor and critic. Periodically, gradients accumulated by parallel agents are applied to global actor and critic networks. This parallelization reduces wall-clock training time substantially while improving sample efficiency through diverse exploration. A3C was highly influential in demonstrating that synchronized non-policy methods like supervised learning could be parallelized effectively. Distributed variants of A2C and improvements like IMPALA extend these ideas further.

### Entropy Regularization and Exploration

Policy gradient methods can converge prematurely to deterministic policies with low entropy. Entropy regularization adds an entropy bonus to the objective: J(θ) = E[log π(a|s; θ)·A + β·H(π(·|s; θ))], where H is policy entropy and β controls the trade-off. Entropy regularization encourages maintaining exploratory randomness even as the policy improves. The entropy bonus is H = -Σ π(a|s; θ) log π(a|s; θ). Tuning β is important; too low a β provides insufficient exploration, too high encourages excessive randomness. Entropy-regularized actor-critic learning naturally balances exploration and exploitation. This approach avoids explicit exploration strategies like ε-greedy and maintains exploration benefits that improve robustness. Entropy regularization is used in many modern algorithms including PPO and A3C.