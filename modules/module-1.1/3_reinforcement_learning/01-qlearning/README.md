# Q-Learning

## Fundamentals

Q-Learning is a model-free reinforcement learning algorithm that learns the value (Q-value) of state-action pairs without requiring a model of the environment. The agent learns by taking actions, observing rewards, and updating Q-values iteratively. Q-Learning is off-policy (learning from actions different from the policy being followed) and is applicable to both discrete and continuous action spaces with function approximation.

## Key Concepts

- **Q-Value**: Expected cumulative future reward
- **Bellman Equation**: Recursive value update
- **Exploration vs Exploitation**: Epsilon-greedy strategy
- **Learning Rate and Discount Factor**: Convergence control
- **Off-Policy Learning**: Decoupled exploration and exploitation

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Markov Decision Processes and Q-Values

Q-learning operates within the framework of Markov Decision Processes (MDPs), which model sequential decision-making under uncertainty. An MDP consists of states S, actions A, transition probabilities P(s'|s, a), rewards R(s, a, s'), and a discount factor γ ∈ [0, 1]. The goal is to find an optimal policy π* that maximizes expected cumulative discounted reward. The Q-function Q(s, a) represents the expected cumulative reward from taking action a in state s and following the optimal policy thereafter: Q(s, a) = E[R(s, a, s') + γ max_{a'} Q(s', a')]. Q-values satisfy the Bellman optimality equation, providing the foundation for Q-learning. The optimal policy is determined greedily from Q-values: π*(s) = argmax_a Q(s, a). Q-learning directly estimates Q-values from experience without requiring knowledge of transition probabilities or rewards, making it model-free and practical.