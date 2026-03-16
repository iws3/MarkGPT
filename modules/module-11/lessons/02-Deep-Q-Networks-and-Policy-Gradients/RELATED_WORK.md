# Related Work on Deep Q-Networks and Policy Gradients

## Deep Q-Networks (DQN)
- Mnih et al. (2013). Playing Atari with deep reinforcement learning
  - Original DQN paper: combined Q-learning with deep networks
  - Breakthrough: learning directly from pixels
  - Key innovations: experience replay, target networks

- Van Hasselt et al. (2015). Deep reinforcement learning with double Q-learning
  - Addressed overestimation bias in DQN
  - Practical improvements to stability

## Policy Gradients
- Williams, R. J. (1992). Simple statistical gradient-following algorithms
  - Original policy gradient theorem
  - REINFORCE algorithm and proof

- Schulman et al. (2015). High-dimensional continuous control using generalized advantage estimation
  - Generalized Advantage Estimation (GAE)
  - Balance variance-bias trade-offs

## Actor-Critic Methods
- Mnih, A., & Gregor, K. (2014). Neural variational inference and learning
  - Unified actor-critic view
  - Practical algorithms

- Haarnoja et al. (2018). Soft actor-critic with automatic entropy adjustment
  - SAC: state-of-the-art for continuous control
  - Entropy regularization

## Proximal Policy Optimization
- Schulman et al. (2017). Proximal policy optimization algorithms
  - PPO: practical and robust
  - Industry standard algorithm

## Stability and Convergence
- Lillicrap et al. (2015). Continuous control with deep reinforcement learning
  - DDPG algorithm
  - Techniques for stabilizing deep RL
