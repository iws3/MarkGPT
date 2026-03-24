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