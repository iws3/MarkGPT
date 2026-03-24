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