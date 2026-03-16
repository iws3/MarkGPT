# Lesson 2: Deep Q-Networks and Policy Gradients

## Table of Contents

1. [Limitations of Tabular Q-Learning](#limitations-of-tabular-q-learning)
2. [Function Approximation in Q-Learning](#function-approximation-in-q-learning)
3. [Deep Q-Networks (DQN) Architecture](#deep-q-networks-dqn-architecture)
4. [Experience Replay](#experience-replay)
5. [Target Networks](#target-networks)
6. [Policy Gradient Methods](#policy-gradient-methods)
7. [Advantage Actor-Critic](#advantage-actor-critic)
8. [Proximal Policy Optimization](#proximal-policy-optimization)
9. [Generalized Advantage Estimation](#generalized-advantage-estimation)
10. [Stability and Convergence Challenges](#stability-and-convergence-challenges)

---

## Limitations of Tabular Q-Learning

While Q-learning is elegant and provably optimal in finite state-action spaces, it faces severe limitations in practical applications. In most real-world domains, the state space is enormous or even continuous—imagine learning to drive a car where the state includes the positions and velocities of all objects in the scene. Creating a lookup table for every possible state-action pair becomes infeasible.

Tabular methods also suffer from poor generalization. If the agent encounters a state it has never seen before, it has no basis for deciding what action to take. In contrast, function approximation allows the agent to generalize knowledge from visited states to similar unvisited states. This generalization is essential for real-world learning where the state space is so large that complete coverage is impossible.

Additionally, tabular methods are not sample efficient. They require repeated visits to the same state-action pair to accurately estimate Q-values. With infinite or very large state spaces, achieving sufficient data coverage would require infeasible amounts of interaction with the environment. These limitations motivated the development of deep Q-learning and other deep RL methods.

---

## Function Approximation in Q-Learning

Function approximation replaces the Q-table with a parameterized function, typically a neural network, that maps states to action values: Q(s, a; θ) ≈ Q*(s, a). The network's parameters θ are adjusted during learning to minimize the prediction error. This approach has several advantages: it generalizes across states, it can handle high-dimensional inputs like images, and it enables exponential reduction in the number of states that must be explicitly encountered.

However, function approximation introduces new challenges. The theoretical guarantees of tabular Q-learning no longer hold—convergence is no longer guaranteed, and the approximation error can propagate and compound. Additionally, the temporal correlation between consecutive transitions violates the independence assumption of supervised learning, causing instability. Despite these challenges, with careful algorithm design, function approximation vastly expands the domains where deep RL can be applied successfully.

The choice of function approximation is crucial. Linear function approximation is stable but limited in expressiveness. Deep neural networks are powerful but prone to instability and high variance in training. Architectural choices like convolutional layers for vision, recurrent connections for temporal dependencies, or attention mechanisms significantly affect learning performance.

---

## Deep Q-Networks (DQN) Architecture

The Deep Q-Network (DQN) was a landmark algorithm that successfully combined Q-learning with deep neural networks to learn from high-dimensional sensory input, particularly images. The network architecture typically includes convolutional layers to process visual information, followed by fully connected layers that output Q-values for each action. For an input image from an Atari game, the network might have three convolutional layers followed by two fully connected layers; the output layer has one unit per possible action.

Key insights of DQN include using separate target and online networks to stabilize learning. The online network θ is used for computing gradients and taking actions, while the target network θ⁻ is frozen and used only to compute target values. This separation breaks the temporal correlation between updates and provides stable training targets. The target network is slowly updated from the online network (either periodically or with a soft update rule), allowing the algorithm to learn from potentially outdated but stable targets.

The architecture also employs experience replay, which stores transitions in a replay buffer and samples minibatches uniformly at random for training. This procedure decorrelates the sequence of updates, improving sample efficiency and stability. Modern extensions of DQN like Dueling networks use separate value and advantage streams to improve learning, and Noisy networks incorporate parametric noise to encourage exploration.

---

## Experience Replay

Experience replay stores past transitions (s, a, r, s') in a buffer and samples them randomly for training. This simple mechanism has profound effects on learning stability and efficiency. By breaking temporal correlations, experience replay reduces variance in gradient estimates, similar to using minibatches in supervised learning. More importantly, randomly sampling old transitions allows the algorithm to learn from experiences multiple times, dramatically improving sample efficiency.

The replay buffer is typically circular with fixed capacity. When the buffer is full, new transitions overwrite old ones. Uniform sampling gives equal probability to all stored transitions, though prioritized experience replay (PER) assigns higher probabilities to transitions with large TD errors, focusing learning on transitions the network finds surprising or incorrect. PER often accelerates learning by ensuring the algorithm spends more effort on important experiences.

Experience replay is not without drawbacks. It requires significant memory to store the buffer, and there's a delay between experience collection and training. The algorithm may overfit to certain transitions if they remain in the buffer for too long, or if the dataset is too small. Despite these limitations, experience replay has become standard in deep RL, appearing in nearly all modern algorithms that combine deep learning with value-based methods.

---

## Target Networks

Target networks are a critical stabilization mechanism in deep Q-learning. In vanilla Q-learning, the update rule uses:

Q(s, a) ← Q(s, a) + α[R + γ max Q(s', a') - Q(s, a)]

When Q is a neural network, updating it with targets that depend on the same network creates a moving target problem. The target max Q(s', a') changes as the network is updated, making the optimization problem non-stationary and unstable. The solution is to maintain two networks: the online network θ provides the learning targets (the gradient step updates this), while a separate target network θ⁻ computes the target values.

The target network θ⁻ is updated infrequently. In the original DQN paper, it was updated every C steps by copying the online network weights. Modern variants use soft updates, gradually blending the online and target networks with coefficient τ:

θ⁻ ← τ × θ + (1 - τ) × θ⁻

With τ ≈ 0.001, this creates a smooth decay where the target network slowly follows the online network's improvements. The use of target networks dramatically stabilizes training, allowing deep Q-learning to converge reliably in domains where it would otherwise diverge or oscillate. Understanding this mechanism is crucial for debugging RL algorithms that rely on bootstrapping.

---

## Policy Gradient Methods

Policy gradient methods take a different approach from value-based methods. Instead of learning a value function and deriving the policy greedily, they directly learn a parameterized policy π(a|s; θ). The policy is typically stochastic, outputting probability distributions over actions. The policy gradient theorem provides the keyway to optimize policies:

∇J(θ) = E[∇ log π(a|s; θ) × Q(s, a)]

This equation states that the gradient of the expected return is the expected value of the policy gradient weighted by the action advantage. The term ∇ log π(a|s; θ) indicates the direction in which we should shift the probability of the action. If the action has high advantage (better-than-average returns), we increase its probability; low advantage decreases it.

Policy gradient methods have several advantages over value-based methods. They can naturally handle stochastic policies and continuous action spaces, they tend to converge to smoother solutions, and they enable direct optimization of non-differentiable objectives. However, they suffer from high variance in gradient estimates because the advantage estimates can be noisy. This variance can slow learning significantly, requiring many samples to get accurate gradient directions. Variance reduction techniques like baseline functions and generalized advantage estimation (GAE) are essential for practical policy gradient algorithms.

---

## Advantage Actor-Critic

Advantage Actor-Critic (A2C) combines policy gradients with value function learning. The "actor" is the policy network that selects actions, and the "critic" is a value network that evaluates state quality. During each step, the actor selects an action according to its policy, the environment responds with a reward and next state, and the critic estimates the advantage:

A(s, a) = R + γV(s') - V(s)

This advantage estimate is then used to update both networks:
- The actor updates to increase the probability of high-advantage actions
- The critic updates to better predict state values

The key innovation of A2C is using the critic's value estimates as a baseline to reduce variance in policy gradient updates. Rather than using raw returns (which have high variance), the algorithm uses advantages (returns minus the value estimate), which have much lower variance while remaining unbiased. This variance reduction dramatically speeds up learning while maintaining theoretical convergence properties.

A2C is particularly attractive because it's relatively simple, stable, and practical. Many modern RL systems, from robotics to game playing, use variants of actor-critic methods. The clear separation between policy learning (actor) and value learning (critic) makes algorithms easier to understand, debug, and extend with additional components like attention mechanisms or hierarchical subpolicies.

---

## Proximal Policy Optimization

Proximal Policy Optimization (PPO) is a state-of-the-art policy gradient algorithm that addresses the instability of naive policy gradients. Large policy updates can lead to poor-performing policies and training collapse, while small updates are inefficient. PPO uses a clipped surrogate objective that prevents the new policy from deviating too far from the old policy:

L^CLIP(θ) = E[min(r(θ) A, clip(r(θ), 1-ε, 1+ε) A)]

where r(θ) = π(a|s; θ) / π(a|s; θ_old) is the probability ratio. The clip function prevents r(θ) from going outside [1-ε, 1+ε], effectively constraining the KL divergence between old and new policies. This constraint ensures each update is small and conservative, improving stability.

PPO has become the standard for many applications because of its robustness and ease of tuning. It performs well across domains with minimal hyperparameter adjustment, unlike some earlier policy gradient methods that required careful tuning. PPO also scales well to high-dimensional continuous control, making it ideal for robotics. Its success in training models at scale (e.g., OpenAI's GPT series uses PPO variants) has made it a go-to choice for practitioners seeking a reliable, stable algorithm.

---

## Generalized Advantage Estimation

Generalized Advantage Estimation (GAE) is a technique for computing advantage estimates that balances bias and variance in temporal difference methods. The standard one-step advantage uses V(s') as a bootstrap target, which has low variance but high bias if the value function is inaccurate. Multi-step returns reduce bias but increase variance. GAE elegantly interpolates between these extremes using:

A_GAE(s) = Σ (γλ)^l δ_V(s+l)

where δ_V(t) = r(t) + γV(s_{t+1}) - V(s_t) is the one-step TD residual. The parameter λ ∈ [0,1] controls the bias-variance trade-off: λ=0 gives the one-step bootstrapped estimate (low variance, high bias), while λ=1 approximates the full trajectory return (low bias, high variance). Intermediate values of λ (typically 0.95-0.99) balance these concerns.

GAE has become the standard way to compute advantages in policy gradient algorithms because it empirically outperforms simpler alternatives. The key insight is that TD residuals δ_V contain useful information—they measure how surprising each state is relative to the value estimate. By aggregating these residuals with exponential weighting, GAE captures the structure of the value learning problem while maintaining good variance properties. Understanding GAE is essential for implementing modern RL algorithms effectively.

---

## Stability and Convergence Challenges

Deep RL algorithms face numerous stability and convergence challenges. Non-stationarity from changing policies or value functions can cause the learning objective to shift, preventing convergence. Function approximation errors can compound, leading to overestimation or underestimation of values. Exploration can be insufficient if the policy is too greedy or if the environment has deceptive reward structures. Additionally, catastrophic interference can occur when learning on a new batch of data erases previously learned knowledge.

Several techniques mitigate these issues. Target networks stabilize value learning by decoupling the data distribution from the update targets. Experience replay provides decorrelated batches and stabilizes the data distribution. Double Q-learning and its extensions reduce overestimation bias in bootstrapping. Entropy regularization encourages exploration by penalizing low-entropy policies. Gradient clipping prevents training instability from large parameter updates. Despite these safeguards, deep RL remains notoriously unstable, and practitioners often implement careful debugging and monitoring practices.

Understanding these stability issues is crucial for successful deep RL work. Bad hyperparameters, architectural choices, or exploration strategies can cause training to diverge silently or converge to poor local optima. Experienced practitioners develop intuition for which components matter most in specific domains and how to diagnose training failures. The field continues to develop new stability improvements, making this an active area of research.

