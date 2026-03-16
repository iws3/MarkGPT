# Lesson 2: Key Concepts Deep Dive

## 1. Deep Neural Networks as Function Approximators

Neural networks universally approximate any continuous function, making them ideal RL function approximators. Convolutional layers extract local patterns (useful for images), while fully connected layers perform high-level reasoning. The magic: they learn useful features automatically without hand engineering.

Challenges: high-capacity networks memorize rather than generalize, especially with small datasets. Regularization through dropout, weight decay, or architecture constraints improves generalization.

## 2. Temporal Difference Error Analysis

The TD error δ = R + γV(s') - V(s) measures how surprising an outcome is. Large errors indicate learning opportunities; small errors suggest the model is well-calibrated. Healthy learning has decreasing average TD error over time. Persistent large errors indicate poor value estimates or non-stationarity.

## 3. Experience Replay Mechanics

Random sampling from a buffer decorrelates data, improving gradient stability. Larger buffers remember more history but consume memory and can become stale. Smaller buffers are fresh but have higher correlation between samples. Finding the right buffer size balances recency and diversity.

## 4. Target Network Convergence

Slowly updating target networks enables convergence while fast updates lead to instability. The update rate (soft vs. hard) balances learning speed vs. stability. Too slow and learning stalls; too fast and the algorithm diverges.

## 5. Policy Gradient Variance Reduction

Policy gradients have high variance; each sample is a noisy estimate of the gradient. Baseline functions (like value functions) reduce variance without changing gradient direction. Better variance reduction = better sample efficiency.

## 6. Entropy in Exploration and Learning

High-entropy policies explore thoroughly; low-entropy policies exploit. Entropy regularization softly encourages exploration by penalizing deterministic policies. Different entropy targets appropriate for different problems.

## 7. Why PPO Works

PPO's clipping mechanism constrains policy changes, preventing catastrophic updates. The resulting objective is simpler than trust region methods (TRPO) while maintaining stability. This simplicity combined with robustness explains PPO's popularity.

## 8. Multi-Step Returns and Credit Assignment

One-step TD uses immediate reward plus bootstrapped value (high bias, low variance). Multi-step returns average more future rewards before bootstrapping (lower bias, higher variance). GAE optimally interpolates using the λ parameter.

