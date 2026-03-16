# Common Pitfalls in Reinforcement Learning Fundamentals

## 1. Confusing Reward Function with Value Function
**The Pitfall**: Many beginners treat rewards and values as interchangeable concepts.

**Why It Matters**: 
- Rewards are immediate signals from the environment for a single action
- Values represent the cumulative expected future reward from a state/action
- Confusing these leads to poor learning signals

**How to Avoid**:
- Always think: R(s,a) is immediate; V(s) is cumulative discounted future
- Value functions depend on the policy being followed
- Use clear notation: R for rewards, V for values, Q for action-values

## 2. Ignoring the Markov Property
**The Pitfall**: Including unnecessary historical information in the state representation, violating the Markov property.

**Why It Matters**:
- Markov property allows us to make optimal decisions based only on current state
- Violating it makes learning unstable and inefficient
- State representation directly impacts learning capability

**How to Avoid**:
- Verify that your state includes all necessary information for decision-making
- Remove redundant historical features
- Use state stacking when necessary (e.g., page flipping for frames)

## 3. Inappropriate Discount Factor Selection
**The Pitfall**: Using discount factor γ = 0.99 without considering the problem's time horizon.

**Why It Matters**:
- Discount factor represents how much we value immediate vs future rewards
- Wrong selection leads to myopic or over-conservative policies
- Can cause instability in temporal-difference learning

**How to Avoid**:
- Consider the effective planning horizon: 1/(1-γ) steps
- For navigation: higher γ (0.995+), for rapid-response tasks: lower γ (0.99 or less)
- Test sensitivity: try γ ∈ {0.9, 0.95, 0.99, 0.999}

## 4. Failing to Initialize Value Estimates Properly
**The Pitfall**: Initializing all Q-values to 0, or consistently high/low values.

**Why It Matters**:
- Poor initialization can lead to biased exploration or premature convergence
- Can cause overestimation or underestimation of values
- Affects convergence speed significantly

**How to Avoid**:
- Use uniform random initialization for Q-learning
- For more advanced methods, use optimistic initialization (high values) to encourage exploration
- Document your initialization strategy

## 5. Not Understanding Exploration vs Exploitation Trade-off
**The Pitfall**: Using pure greedy policies too early, or exploring uniformly throughout training.

**Why It Matters**:
- Over-exploitation early leads to suboptimal local policies
- Over-exploration late wastes time on known bad actions
- Balance determines convergence speed and final policy quality

**How to Avoid**:
- Implement ε-greedy with annealing schedule: ε decreases over time
- Or use decaying soft-max temperature
- Monitor cumulative regret: should decrease as training progresses

## 6. Ignoring Terminal States
**The Pitfall**: Treating terminal states as regular states with V(terminal) learned through bootstrapping.

**Why It Matters**:
- Terminal states have V = 0 by definition (no future rewards possible)
- Bootstrapping from terminal states introduces bias
- Can cause learning instability

**How to Avoid**:
- Always set V(terminal_state) = 0 explicitly
- Mask out terminal states in TD updates: V(s) ← R(s,a) if s' is terminal
- Document terminal state handling in your code

## 7. Mixing Episodic and Continuing Tasks Without Adjustment
**The Pitfall**: Using same Bellman equations for both task types without proper modifications.

**Why It Matters**:
- Continuing tasks need different value definitions (average reward vs discounted)
- Episodic tasks require explicit terminal state handling
- Algorithm differences are fundamental, not minor tweaks

**How to Avoid**:
- Explicitly declare: "This is an episodic task with horizon H"
- Use appropriate value functions: discounted sum for episodic, average for continuing
- Test both task formulations if unsure

## 8. Poor State Representation Leading to Function Approximation Failure
**The Pitfall**: When moving to function approximation, using feature representations that don't capture problem structure.

**Why It Matters**:
- Linear function approximation relies on features to capture state similarity
- Poor features lead to generalization failures
- Can completely break value function learning

**How to Avoid**:
- Domain analysis: identify what state features matter for decisions
- Use RBF kernels, tile coding, or raw features depending on problem
- Implement feature normalization: zero mean, unit variance
- Visualize feature importance during training

## 9. Incorrect Bellman Equation Application
**The Pitfall**: Writing Bellman equations with wrong sign conventions or missing components.

**Classic Mistake**:
```
Wrong: V(s) = E[R(s,a) + γ*V(s')]  (missing expectation over actions)
Right: V(s) = E_a[R(s,a) + γ*E[V(s')|s,a]]
```

**How to Avoid**:
- Always write the full equation with explicit expectation operators
- Double-check: rewards are immediate, values are future
- Verify signs: should be addition, not subtraction

## 10. Not Validating Core Assumptions
**The Pitfall**: Assuming environment is Markovian/fully observable without verification.

**Why It Matters**:
- Many assumptions break down in practice
- Hidden state leads to poor learning and high variance
- Partial observability requires different algorithms

**How to Avoid**:
- Test Markov property: verify that P(s' | s, a) is well-defined
- Check observability: can current observation disambiguate states?
- Add state history if needed (frame stacking, recurrent networks)
- Document assumptions in problem formulation

## Summary
These pitfalls represent the most common mistakes when learning RL fundamentals. The key pattern: **understand why each equation and design choice exists, don't just apply formulas mechanically**.
