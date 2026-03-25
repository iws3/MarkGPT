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

### Stability, Convergence, and Practical Considerations

Actor-critic methods can suffer from instability due to non-stationary targets (critic changes while learning policy) and correlated experience (sequential states are highly correlated). Modern variations address these issues through experience replay, target networks, and synchronized updates across multiple parallel workers. The choice of network architectures for actor and critic significantly impacts performance; shared layers between actor and critic can improve learning efficiency while separate networks provide more flexibility. Careful hyperparameter tuning is essential: learning rates for actor and critic often differ, GAE parameter λ requires tuning, and entropy regularization strength β must be selected. Despite challenges, modern actor-critic methods (A3C, PPO, TRPO) have become some of the most practical and effective reinforcement learning algorithms, achieving strong performance on diverse continuous control and game-playing tasks. The combination of value and policy-based learning through actor-critic provides complementary strengths.

### GAE Implementation and Advantage Normalization

Implementing Generalized Advantage Estimation requires accumulating rewards and values along trajectories. For each timestep t: δ_t = r_t + γV(s_{t+1}) - V(s_t). GAE advantages: A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}. Computationally: A_t = δ_t + γλ * A_{t+1}, computed backward from episode end. This recursion is efficient. Advantage normalization (subtracting mean, dividing by std) standardizes advantages across episodes, stabilizing learning. Without normalization, advantage magnitudes vary widely (different episode lengths, reward scales); normalization ensures consistent gradient magnitudes. In practice: compute all advantages for batch, normalize, then update networks. This preprocessing step significantly improves convergence.

### A2C vs A3C: Synchronous vs Asynchronous Learning

A2C (Advantage Actor-Critic) is synchronous: collect experiences from n parallel environments, compute gradients from all n, then apply updates. This is efficient with distributed computing. A3C (Asynchronous Advantage Actor-Critic) is truly asynchronous: each worker independently runs environment, computes gradients, applies to global network. No synchronization needed; agents start updating at different times. A3C's asynchrony enables parallelization without specialized distributed infrastructure; each worker is independent. Both achieve similar final performance; A2C is more efficient with proper distributed setup, A3C is simpler to implement. In modern frameworks (PyTorch with distributed data parallel), A2C-style architecture is easier. The asynchronous nature of A3C doesn't matter if underlying compute is synchronized anyway.

### Continuous Action Spaces: Gaussian Policies

For continuous action spaces (robot control, autonomous vehicles), policies output continuous actions. Gaussian policies are standard: the network outputs mean μ(s) and std σ(s) for action distribution. Actions are sampled: a ~ N(μ(s), σ²(s)). Log-probability: log π(a|s) = -0.5 * log(2π) - log(σ) - 0.5 * ((a-μ)/σ)². Policy gradient updates are straightforward; the network learns to output appropriate means and stds. σ can be state-dependent or globally learned.  State-dependent stds enable adaptive exploration; exploratory actions where uncertain, deterministic where confident. For stability, σ is often represented as log σ (unconstrained) or bounded via softplus. Another approach: squashing network output via tanh to [-1,1], appropriate for bounded actions (e.g., robot joint limits). Transformed policy gradients account for Jacobian of squashing transformation.

### Distributional Perspectives and Value Range

Tradition actor-critic estimates scalar values V(s). Distributional RL estimates full return distributions, not just expectations. This richer representation improves learning: agents understand return variance (uncertainty about future rewards). Categorical distributional RL represents returns as discrete probability distributions over value bins. QR-DQN uses quantile regression, learning quantiles of return distributions. These approaches show empirical improvements and theoretical benefits: better exploration (high-uncertainty actions), more robust to outliers, improved credit assignment. However, implementation is more complex; most practitioners use simplified scalar value functions. State-of-the-art methods (Rainbow DQN) combine distributional approaches with other improvements; the synergy contributes to strong performance.

### Actor-Critic Applications and Benchmarks

Modern benchmarks (OpenAI Gym, DeepMind Control Suite, Atari, robotics simulators) evaluate RL algorithms. Actor-critic methods consistently rank among top performers. PPO (a policy gradient variant with trust regions) is dominant due to simplicity and reliability. SAC (Soft Actor-Critic) uses entropy regularization and distributional perspectives, achieving impressive continuous control results. Robotics simulators (MuJoCo, Bullet) test continuous control; actor-critic methods excel. In game-playing, policy gradients (with distributed training) match or exceed value-based methods (DQN). The practical dominance of actor-critic reflects its balance of simplicity, stability, and sample efficiency. For practitioners starting RL, implementing and tuning PPO or A2C is recommended.