# Lesson 1: Key Takeaways and Summary

## Essential Concepts

1. **MDPs formalize RL**: States, actions, transitions, rewards, and discount factor define the problem.
2. **Markov property**: History-independence simplifies RL dramatically if approximate.
3. **Value functions**: V(s) = expected return from state s, Q(s,a) = expected return from action a in state s.
4. **Bellman equations**: Recursive relationships enabling value computation.
5. **Optimal policies**: Greedy with respect to optimal value functions, can determine policy from value.
6. **Q-learning**: Model-free, off-policy algorithm for learning optimal action-values from experience.
7. **Exploration-exploitation**: Balance between trying new actions and exploiting known good ones.
8. **TD learning**: More efficient than MC methods, bootstrap from learned estimates.

## Critical Insights

- **Bootstrapping is powerful**: Using estimates of future value to improve current estimates dramatically improves sample efficiency.
- **Off-policy learning enables efficient learning**: Q-learning from exploratory trajectories toward optimal policy is highly practical.
- **Value functions generalize**: Similar states have similar values, enabling learning from limited experience.
- **Temporal structure matters**: MDPs with sequential decision structure enable efficient long-horizon planning.

## Common Mistakes

1. **Ignoring exploration**: Insufficient exploration leads to suboptimal convergence.
2. **Wrong reward discounting**: Forgetting the discount factor or using wrong γ causes incorrect value judgments.
3. **Confusing V and Q**: Value functions vs. action-value functions are different; know which applies.
4. **Assuming convergence without tabular settings**: Great convergence guarantees only apply to tabular MDPs.

## When to Apply

- **Q-learning**: Discrete state-action spaces, model-free learning, online learning scenarios, need fast convergence.
- **SARSA**: On-policy safer alternative, real-world applications where mistakes matter.
- **Expected SARSA**: Compromise between SARSA and Q-learning, reduced variance.

## Recommended Further Learning

- Implement Q-learning on Gridworld to internalize the algorithm.
- Study convergence proofs to understand theoretical foundations.
- Explore TD(λ) and eligibility traces for improved efficiency.
- Practice tabular RL on various problem structures to develop intuition.

