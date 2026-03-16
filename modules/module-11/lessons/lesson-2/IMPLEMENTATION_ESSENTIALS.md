# Lesson 2: Implementation Essentials

## Setting Up DQN Training

For those implementing Deep Q-Networks from scratch, key practical considerations:

**Network Architecture**: Standard DQN uses convolutional layers for image input. The network takes a state (or image history) and outputs Q-values for each action. For Atari: 3 conv layers (32, 64, 64 channels) followed by 2x fully connected (512 units each), output layer with |A| units.

**Target Network Update**: Maintain a separate target network updated every C steps (typically 10,000). The computational cost is minimal—just copying weights. The stability gain is enormous. Soft updates (Q_target ← 0.001 * Q_online + 0.999 * Q_target) are gentler but usually less necessary with proper replay.

**Experience Replay Details**: Use a circular replay buffer sized between 100K-1M depending on task complexity. For discrete actions, store transitions as (state, action, reward, next_state, done). When sampling minibatches, ensure states are preprocessed identically to training. Frame stacking (concatenating last 4 frames as state) is standard for Atari to provide temporal information.

**Exploration Strategy**: Start with ε=1.0 (fully random), decay to ε=0.01 or ε=0.05 over the first million steps. This gives the network time to learn reasonable Q-estimates before acting greedily. The final ε (fraction of random actions taken) remains for continued exploration.

**Loss Function**: Huber loss works better than MSE for Q-learning, preventing instability from large TD errors. The value to regress is: target_Q = r + γ * max_a' Q_target(s', a').

## Common Bugs and Fixes

**Double Q Overestimation**: Values can diverge upward if Q-learning overestimates optimal Q-values. Double Q-learning fixes this by using Q_online for action selection and Q_target for evaluation: target = r + γ * Q_target(s', argmax_a Q_online(s', a)).

**Reward Scaling**: RL training is sensitive to reward magnitudes. Scale rewards to reasonable ranges (typically -1 to 1 per step). Inconsistent reward scaling across parts of the algorithm causes instability.

**Incorrect Masking**: When computing target Q-values, remember to zero out Q-values for terminal states—terminal states have no future value. Set target_Q = r if done else r + γ * max Q_target(s', a').

**Off-By-One Errors**: Careful indexing when constructing batches. Ensure next_states correspond to current states. Many implementation bugs stem from misaligned state-action-nextstate triplets.

## Hyperparameter Tuning

**Learning Rate**: 1e-4 is the standard starting point for DQN. Too high and training diverges; too low and learning is slow. Usually keep constant throughout training. If divergence occurs, reduce learning rate.

**Batch Size**: 32 is standard; 64 works too. Larger batches are stabler but require more memory. The advantage diminishes beyond 64-128.

**Discount Factor (γ)**: 0.99 is standard for most domains. Higher γ values make the agent plan further ahead; lower values myopically optimize immediate rewards. 0.99 is a good default, though 0.95-0.995 work in many settings.

**Replay Buffer Size**: Larger is generally better, but too large causes memory issues and very old data might be outdated. 1M transitions is standard for Atari; resource-constrained settings might use 100K.

**Target Update Frequency**: 10,000 steps is standard. Too frequent (updating every step) loses stability; too infrequent (100K+ steps) causes the target to lag significantly. 10K-50K is a good range.

