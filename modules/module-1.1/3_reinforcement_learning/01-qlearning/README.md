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

### Off-Policy Learning and the Q-Learning Update Rule

Q-learning is off-policy, meaning it learns the optimal policy while following a different behavior policy for exploration. This flexibility allows improving exploration while learning optimal behavior. The Q-learning update rule is: Q(s, a) ← Q(s, a) + α[R(s, a, s') + γ max_{a'} Q(s', a') - Q(s, a)], where α is the learning rate and the bracketed term is the temporal difference (TD) error. This update moves Q-values toward the Bellman target R + γ max Q(s', a'). The learning rate α ∈ (0, 1] balances between incorporating new information (high α) and stability (low α). Decreasing α over time creates robust convergence. The discount factor γ balances immediate and long-term rewards; γ near 1 prioritizes long-term rewards while γ near 0 prioritizes immediate rewards. Q-learning converges to the optimal Q* under the conditions of adequate exploration and decreasing learning rates.