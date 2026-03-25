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

### Advantage Estimation and Variance Reduction

A key challenge in policy gradient learning is high variance in gradient estimates. The basic REINFORCE algorithm uses full trajectory returns as returns: ∇_θ log π(a_t|s_t; θ) · G_t where G_t = Σ_{k=t}^T γ^{k-t} r_k. This unbiased but high-variance gradient estimate requires many samples per update. Advantage Actor-Critic methods use advantage functions A(s, a) = Q(s, a) - V(s) representing how much better an action is than average. The policy gradient becomes ∇_θ J(θ) = E[∇_θ log π(a|s; θ) · A(s, a)]. Advantage estimation reduces variance substantially while introducing small bias. Temporal difference (TD) advantages use: A(s, a) = r + γV(s') - V(s), requiring only one-step lookahead. Generalized Advantage Estimation (GAE) interpolates between TD and full-return advantages, balancing bias and variance through a parameter λ ∈ [0, 1]. Advantage estimation is crucial for practical policy gradient methods.

### Trust Region and Natural Gradient Methods

Unconstrained policy gradient updates can cause divergence through excessively large parameter changes. Trust region methods constrain updates within regions where the linear approximation of policy performance is valid. Trust Region Policy Optimization (TRPO) constrains updates to regions where KL divergence from the old policy stays below δ: constrain KL(π_old || π_new) ≤ δ. This prevents destructively large parameter changes. Proximal Policy Optimization (PPO) simplifies TRPO using a clipped objective: min[r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t] where r_t = π(a|s; θ) / π(a|s; θ_old). Clipping prevents probability ratios from deviating too far from 1, limiting update magnitude. PPO combines computational simplicity with stability, becoming extremely popular. Natural gradient methods use the Fisher information matrix to better adapt step sizes across parameter dimensions, improving convergence. These advances make policy gradient methods stable and practical.

### Policy Gradient Variants and Applications

Batch policy gradient methods (policy gradient + advantage actor-critic) collect transitions for multiple steps, accumulating advantages and computing policy gradients. On-policy methods learn from currently-generated trajectories, discarding old data; off-policy variants like Importance Weighted Policy Gradients correct for distribution mismatch. Continuous control problems benefit from policy gradient methods that naturally handle continuous action spaces. Robotics, control, and game-playing leverage policy gradients. REINFORCE, Actor-Critic, A3C (Asynchronous Advantage Actor-Critic), PPO, and TRPO form a family of increasingly sophisticated policy gradient algorithms. Despite high variance in basic forms, modern policy gradient methods with variance reduction, trust regions, and natural gradients achieve excellent performance. Policy gradients complement value-based methods; combining both in actor-critic architectures provides significant benefits over either alone.

### Policy Gradient Theorem and Gradient Computation

The policy gradient theorem states: ∇_θ J(θ) = E[∇_θ log π(a|s; θ) Q(s,a)]. This elegant result connects policy gradients to value functions. The key insight: ∇_θ log π(a|s; θ) points in direction increasing action probability. Multiplying by Q(s,a) (advantage of action) weights this by actual goodness. For high Q (good action), prob increases; for low Q, prob decreases. The gradient is easy to compute for any differentiable policy. For neural network policies, log probability is typically cross-entropy loss (categorical for discrete actions, Gaussian log-density for continuous). The beauty: no need to estimate true Q function; approximation Q̂ is sufficient (introduces bias but reduces variance). The trade-off: better Q approximations reduce bias; worse approximations reduce variance. Average between two extremes is optimal.

### Advantage Estimation and GAE

Policy gradients use advantage A(s,a) = Q(s,a) - V(s), representing action preference relative to baseline. Baseline V(s) reduces variance without adding bias; policies better than baseline get positive advantages, worse get negative. This significantly improves sample efficiency. Computing advantages: one-step temporal difference A(s,a) = r + γV(s') - V(s) is low-variance but biased. n-step returns A = Σ γ^k r_{t+k} + γ^n V(s_{t+n}) - V(s_t) are higher-variance, lower-bias. Generalized Advantage Estimation (GAE) interpolates: A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}, where δ = r + γV(s') - V(s). Parameter λ ∈ [0,1] controls: λ=0 is one-step (high bias, low variance), λ=1 is full return (low bias, high variance). Empirically λ=0.95 often works well, balancing bias and variance. GAE provides principled advantage estimation crucial for stable training.

### Trust Region Policy Optimization and PPO

Unconstrained policy gradient updates can cause large parameter changes, degrading performance. Trust region methods constrain updates: TRPO (Trust Region Policy Optimization) restricts KL divergence between old and new policies: constrain KL(π_old || π_new) ≤ δ. Solving this constrained optimization exactly is complex; conjugate gradient methods approximately solve. PPO (Proximal Policy Optimization) simplifies via clipping: instead of enforcing constraint, clip probability ratios to [1-ε, 1+ε]. The clipped objective is min[r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t], where r_t = π(a|s; θ) / π(a|s; θ_old). Clipping prevents destructively large updates while enabling beneficial updates. PPO is simpler than TRPO (no second-order derivatives), faster, and empirically outperforms. PPO has dominated recent deep RL; nearly all modern results use PPO or variants. The simplicity and effectiveness make PPO the default policy gradient algorithm.

### Actor-Critic Architecture Integration

Combining policy gradients with value estimation: actor (policy) selects actions, critic (value network) estimates values. The critic provides advantage estimates for policy gradient updates, significantly reducing variance. During training: (1) Actor generates trajectories; (2) Critic estimates values; (3) Compute advantages (critic outputs); (4) Update actor using policy gradients with advantages; (5) Update critic to better estimate values. This two-network architecture is powerful: each network focuses on its task (action selection vs value estimation), improving learning. A3C (Asynchronous Advantage Actor-Critic) parallelizes by running multiple agents in parallel environments, accumulating gradients, then applying to global actor-critic. Parallelization accelerates learning substantially without synchronized sampling. A3C and variants (A2C, PPO) are fundamental in modern deep RL.