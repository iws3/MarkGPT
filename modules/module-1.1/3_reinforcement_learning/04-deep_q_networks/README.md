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