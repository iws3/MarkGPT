# Common Pitfalls in Deep Q-Networks and Policy Gradients

## 1. Overestimation Bias in Policy Gradients
**The Pitfall**: Policy gradient methods can lead to high variance, which is then misattributed as high values.

**Why It Matters**:
- Overestimation bias accumulates through training
- Leads to unstable and divergent learning
- Makes the policy overly aggressive

**How to Avoid**:
- Use double networks (Double DQN)
- Implement gradient clipping for policy gradient methods
- Monitor max Q-values; they should be reasonable for your environment

## 2. Replay Buffer Contamination
**The Pitfall**: Not properly handling old/stale transitions in experience replay, or using too small buffer.

**Why It Matters**:
- Old transitions don't reflect current policy distribution
- Small buffers lead to high correlation in minibatches
- Can cause catastrophic forgetting

**How to Avoid**:
- Use buffer size at least N transitions per action
- Prioritize sampling: weight recent or surprising transitions higher
- Periodically inspect replay buffer for outdated data

## 3. Wrong Target Network Update Frequency
**The Pitfall**: Updating target network too frequently (making it a copy of online network) or too infrequently (stale targets).

**Why It Matters**:
- Too frequent: loses benefits of stable targets
- Too infrequent: targets become severely stale
- Affects convergence speed and stability

**How to Avoid**:
- Standard: update every 10k steps
- Experiment: try τ ∈ {0.001, 0.01} for soft updates
- Monitor target network divergence from online network

## 4. Policy Gradient Variance Explosion
**The Pitfall**: High variance in policy gradient estimates due to large return estimates or poor baselines.

**Why It Matters**:
- High variance requires huge sample sizes
- Can cause instability and poor efficiency
- Empirically converges to local minima

**How to Avoid**:
- Always use baseline subtraction: ∇ log π * (R - V)
- Properly normalize rewards: (R - mean) / std
- Use advantage normalization in clipped objectives (PPO)

## 5. Incorrect Action Value vs Policy Gradient Separation
**The Pitfall**: Mixing DQN-style discrete value updates with continuous policy gradient methods.

**Why It Matters**:
- Different methods have different convergence guarantees
- Mixing them can break theoretical properties
- Leads to unpredictable behavior

**How to Avoid**:
- Choose one approach for your problem formulation
- If combining (Actor-Critic), ensure both components are appropriate
- Document which algorithm you're implementing

## 6. Inadequate Entropy Regularization
**The Pitfall**: Policy gradients can become too greedy, leading to poor exploration late in training.

**Why It Matters**:
- Without entropy bonus, optimal policy can become degenerate
- Hard to escape local optima
- Adversarial robustness suffers

**How to Avoid**:
- Add entropy regularization: -α * H[π]
- Decay entropy coefficient: starts high (0.01), ends low (0.0)
- Monitor policy entropy; should not collapse to single action

## 7. Network Architecture Incompatibility with DQN
**The Pitfall**: Using shared layers between policy and value networks in DQN-style updates.

**Why It Matters**:
- Separate networks for policy and value help stability
- Shared representations can lead to interference
- Cartpole might work; complex tasks will fail

**How to Avoid**:
- Keep dueling DQN heads separate initially
- If sharing, use residual connections
- Test network replacement policies carefully

## 8. Experience Prioritization Without Importance Sampling
**The Pitfall**: Using prioritized replay but not correcting for bias with importance weights.

**Why It Matters**:
- Prioritized sampling can bias Q-value estimates
- Without importance weights, convergence is incorrect
- Can converge to suboptimal policy

**How to Avoid**:
- Always weight samples: (1/N*P(i))^β in TD update
- Increase β from 0 to 1 over training
- Normalize weights: w_i / max(w)

## 9. Incorrect Advantage Function in Actor-Critic
**The Pitfall**: Computing advantages as A = Q - V instead of A = R + γV' - V.

**Why It Matters**:
- Wrong advantage formula breaks the bias-variance tradeoff
- Can cause divergence in policy updates
- Theoretical convergence fails

**How to Avoid**:
- Use TD-residual: A = R + γV(s') - V(s)
- Or use GAE: combine multiple TD steps for smoother estimates
- Verify formula matches your algorithm specification

## 10. Gym API Friction During Training
**The Pitfall**: Assuming DQN/Policy Gradient code works directly with Gym without proper preprocessing.

**Why It Matters**:
- Image observations need normalization
- Action indices need remapping
- Reset state handling differs between algorithmsHow to Avoid**:
- Explicitly preprocess observations: grayscale, resize, normalize
- Test on small environment first (CartPole)
- Use wrapper functions for observation/action conversion

## Summary
DQN and Policy Gradient pitfalls often stem from the interplay between exploration, value estimation, and policy learning. Master the fundamentals of each component before combining them.
