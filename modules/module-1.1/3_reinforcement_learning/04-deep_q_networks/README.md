# Deep Q-Networks (DQN)

## Fundamentals

DQN combines Q-Learning with deep neural networks, enabling learning in high-dimensional state spaces like Atari games. Key innovations include experience replay and target networks for stability. DQN demonstrated that reinforcement learning could master complex games from pixel inputs, marking a breakthrough in deep reinforcement learning.

## Key Concepts

- **Neural Network Q-Function**: Deep approximation
- **Experience Replay**: Breaking temporal correlation
- **Target Network**: Stable learning targets
- **Double DQN**: Addressing overestimation
- **Prioritized Replay**: Efficient sampling

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Deep Q-Networks: Function Approximation with Neural Networks

Deep Q-Networks (DQN) applies Q-learning with neural networks to approximate the Q-function: Q(s, a; θ) ≈ Q(s, a). A neural network with parameters θ maps states to action values; the output layer has |A| units (one per action). During training, a batch of states is forward-passed through the network; gradients are computed from the loss (Q(s, a; θ) - target)². However, directly using neural networks for Q-learning creates instability: targets constantly change as network parameters θ change, and sequential states are correlated, violating the IID assumption underlying optimization. These problems caused early Q-learning with neural networks to diverge. DQN introduced two key stabilization techniques: experience replay and target networks, enabling stable deep reinforcement learning.

### Experience Replay and Target Networks

Experience replay stores transitions (s, a, r, s') in a replay buffer, a large memory storing recent experiences. During training, minibatches are sampled randomly from this buffer rather than using sequential transitions. Random sampling breaks correlations between samples, satisfying IID assumptions and stabilizing learning. The target network is a separate copy of the Q-network that is updated infrequently (every C steps or after certain number of gradient steps). During training, the target is computed using the older target network weights: target = r + γ max_{a'} Q(s', a'; θ_target). The main network weights θ are updated to match this target. Periodically, target network weights are synchronized with main network weights: θ_target ← θ. This decoupling provides slowly-changing targets, stabilizing training substantially. The combination of experience replay and target networks was revolutionary, making deep Q-learning stable and practical.

### Double DQN and Dueling Architectures

DQN tends to overestimate Q-values because max_{a'} Q(s', a'; θ_target) uses the same network to select and evaluate actions. Double DQN addresses this by decoupling action selection and evaluation: target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ_target). Actions minimizing overestimation are selected using the main network, then evaluated using the target network. This simple change substantially improves performance. Dueling DQN decomposes the Q-function into state value and advantage streams: Q(s, a; θ) = V(s; θ) + A(s, a; θ), where V(s) is the value of state s and A(s, a) is the advantage of action a. Separate streams learn value and advantages; combining them through addition recovers Q-values. Dueling architecture provides richer learning signals, leveraging advantages without explicit subtraction from Q-values. Combining dueling architecture with double DQN provides further performance improvements. Prioritized Experience Replay samples transitions with probability proportional to TD error magnitude, focusing learning on high-error transitions.

### Extensions and Practical Implementations

Dueling Double DQN combines benefits of both improvements. Rainbow DQN integrates multiple improvements: double Q-learning, prioritized experience replay, dueling networks, multi-step returns, distributional RL, and noisy networks. Distributional RL learns the full distribution of returns rather than just expectations, providing richer value representations. Noisy networks use noise in parameters for exploration rather than ε-greedy, creating more consistent exploration strategies. Modern implementations address computational efficiency and stability through various engineering improvements. Rainbow DQN achieved near-superhuman performance on Atari 2600 games, demonstrating DQN's potential when combined with multiple improvements. Despite advances, DQN-family algorithms remain primarily suitable for discrete action spaces; continuous control requires different approaches like policy gradients or actor-critic methods. DQN demonstrated that combining deep learning with reinforcement learning through careful stabilization techniques enables learning complex behaviors directly from high-dimensional sensory data, founding the deep reinforcement learning field.

### DQN Variants: Dueling, Double, and Noisy Networks

Dueling DQN decomposes Q = V + (A - mean{A}), separating state value and advantage. This architectural change improves learning: the network learns value and advantages separately, focusing learning signals on relevant parts. Empirically, dueling improves performance, particularly when advantages are similar (many actions give similar rewards). Double DQN addresses overestimation via decoupled action selection and evaluation. Noisy networks add learnable noise to weights rather than ε-greedy exploration. The noise is sample-specific (different noise per forward pass), enabling exploration. Unlike ε-greedy (sudden randomness), noisy networks provide continuous exploration. These variants can be combined (dueling + double + noisy = strong ensemble). Empirically, all three improve DQN. Rainbow DQN combines these plus prioritized replay and distributional RL, achieving near-superhuman Atari performance.

### Atari 2600 Benchmark and Model Capacity

Atari games (57 games) form a standard benchmark for deep RL. DQN's breakthrough was achieving superhuman performance on multiple games without game-specific engineering. Raw pixels (84x84 grayscale images) are inputs; game scores are rewards. The challenge: learning from high-dimensional perceptual input and delayed rewards. DQN's success inspired massive follow-up research. Modern methods (Rainbow DQN, MuZero) achieve far higher scores. The benchmark's difficulty lies in: (1) Long credit assignment (rewards come many steps after actions); (2) Sparse rewards (long stretches without reward); (3) Complex state spaces (many possible situations). Analyzing which games are hard: sparse-reward games (Montezuma's Revenge) remain challenging; reward-rich games are solved. This drives algorithm development toward better credit assignment and exploration.

### Continuous Control and DQN Limitations

DQN is designed for discrete action spaces; extending to continuous is non-trivial. Naive approach: discretize continuous actions (e.g., joint angles into 10 bins), apply DQN. This wastes action space resolution and doesn't scale (3 joints × 10 bins = 1000 actions; 5 joints with 10 bins = 100k actions; curse of dimensionality). SVG (Stochastic Value Gradient) extends DQN to continuous actions via differentiable action selection; policy gradient on Q-values. DDPG (Deep Deterministic Policy Gradient) combines ideas: networks output deterministic actions, advantage is q gradient. A3C/PPO (policy gradient methods) naturally handle continuous actions via Gaussian policy outputs. For continuous control, policy gradients are more appropriate than value-based methods; this is why continuous benchmarks (MuJoCo, robotics simulators) use A3C, PPO, SAC rather than DQN variants.

### Model-Based RL and DQN Comparison

DQN is model-free: it learns value functions from experience without learning environment model (dynamics). Model-based RL learns transition model p(s'|s,a), enabling planning: imagine action sequences, predict outcomes, select sequences with best predicted returns. Model-based advantages: sample efficiency (learning dynamics from fewer samples enables many planning simulations). Model-based disadvantages: model errors compound (wrong predictions after 100 steps are unreliable); learning environment models is hard (unpredictable stochasticity). Recent work (AlphaZero, MuZero) combines both: learning implicit value functions via dynamics prediction without explicit model. Practical applications favor model-free in high-stochasticity environments (games, robotics); model-based in low-stochasticity environments (planning, control with known physics).