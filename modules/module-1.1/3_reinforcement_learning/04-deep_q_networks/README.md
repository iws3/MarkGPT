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