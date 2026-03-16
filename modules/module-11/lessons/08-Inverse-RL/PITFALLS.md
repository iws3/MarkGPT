# Common Pitfalls in Inverse Reinforcement Learning

## 1. Reward Ambiguity: Multiple Reward Functions Explain Same Behavior
**The Pitfall**: IRL problem is fundamentally under-determined; multiple rewards fit expert demonstrations.

**Why It Matters**:
- Learned reward may be completely different from true reward
- Model fits training data but generalizes poorly
- Hard to validate correctness

**How to Avoid**:
- Add reward prior: simple rewards more likely than complex
- Use regularization: L1/L2 on reward weights
- Validation set: does inferred reward explain new expert behavior?

## 2. Expert Demonstrations Not Optimal
**The Pitfall**: Assuming demonstrations are optimal when expert makes mistakes.

**Why It Matters**:
- IRL assumes E[∑R] highest under expert policy
- Suboptimal demonstrations break this assumption
- Inferred reward is biased

**How to Avoid**:
- Validate expert optimality first
- If suboptimal: implement apprenticeship learning instead
- Document: "Expert is X% optimal" in analysis

## 3. Insufficient Demonstration Diversity
**The Pitfall**: Expert shows only one strategy; reward is specific to that strategy.

**Why It Matters**:
- Inferred reward may not explain other valid strategies
- Learned policy can't adapt
- Overfit to specific expert behavior

**How to Avoid**:
- Collect diverse demonstrations: different approach types
- Check coverage: can inferred reward distinguish good/bad behaviors?
- Validate: does reward generalize to variants?

## 4. Linear Reward Assumption with Nonlinear Dynamics
**The Pitfall**: Using linear rewards: R = w^T φ(s) when features are nonlinear in value.

**Why It Matters**:
- Cannot express some reward preferences
- Inferred weights are biased
- Poor generalization

**How to Avoid**:
- Use nonlinear reward: neural network parameterization
- Or richer features: carefully chosen basis functions
- Measure: can learned reward distinguish expert actions?

## 5. No Regularization Against Reward Hacking
**The Pitfall**: Inferred reward has degenerate solutions (e.g., all actions equally good).

**Why It Matters**:
- IRL solver finds pathological minima
- Reward is meaningless
- Learned policy doesn't reflect expert intent

**How to Avoid**:
- Entropy regularization: prefer simpler reward functions
- Margin constraints: reward should separate expert by margin
- Validate: reward is not near zero everywhere

## 6. Incompatibility Between Expert and Learning Agent
**The Pitfall**: Expert's capabilities (observation, action spaces) differ from learning agent's.

**Why It Matters**:
- Inferred reward is specific to expert's state-action space
- Agent can't use it (can't observe same features)
- Transfer fails

**How to Avoid**:
- Ensure compatibility: same observation/action spaces
- Or translate: map expert observations to agent observations
- Validate: agent can execute expert-inferred behavior

## 7. Long Horizon Accumulation Without Discounting Issues
**The Pitfall**: Ignoring that discount factor affects inferred reward magnitude.

**Why It Matters**:
- Same behavior sequence with different γ implies different rewards
- Comparison across experiments is unfair
- Magnitude of inferred reward is meaningless

**How to Avoid**:
- Fix discount factor: document clearly
- Normalize rewards: divide by (1-γ) to account for horizon
- Report: effective planning horizon implied by reward

## 8. GAIL vs Maximum-Entropy IRL Confusion
**The Pitfall**: Using GAIL (which doesn't invert reward) expecting to get true reward.

**Why It Matters**:
- GAIL is policy imitation, not reward inference
- No meaningful reward vector to extract
- False claims about inferred reward

**How to Avoid**:
- Use IRL for reward: MaxEnt IRL, Deep IRL
- Use GAIL for behavior: if just want to copy expert
- Clear terminology: "We inferred reward" vs "We copied behavior"

## 9. Scalability to High-Dimensional State Spaces
**The Pitfall**: IRL assumes feature-based reward; breaks on raw image observations.

**Why It Matters**:
- Can't hand-engineer features for high-dim problems
- IRL solvers are slow
- Practically infeasible

**How to Avoid**:
- Use neural network reward: end-to-end learning
- Or learn feature representation first, then IRL on features
- Measure: computation time vs state space dimensionality

## 10. Circular Dependency: Inferred Reward Used to Evaluate Policy
**The Pitfall**: Judge learned policy using inferred reward (circularity).

**Why It Matters**:
- Inferred reward may be wrong
- Circular validation reinforces errors
- False confidence in policy quality

**How to Avoid**:
- Validate on ground truth reward (if available) or expert evaluation
- External metrics: separate from inferred reward
- Separate train/test : infer reward on train demos, evaluate on test demos

## Summary
IRL pitfalls stem from under-determinedness and the difficulty of validation. The key challenge: "How do I know if my inferred reward is correct when I don't know ground truth?"
