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

