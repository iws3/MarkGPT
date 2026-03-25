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

### Exploration-Exploitation Trade-off and Strategies

During learning, agents must balance exploring unknown actions to discover reward opportunities versus exploiting known good actions. Pure exploitation quickly gets stuck in local optima; pure exploration wastes samples. ε-greedy strategies take random actions with probability ε and greedy actions otherwise. Decaying ε from high to low values provides initial exploration followed by exploitation convergence. Softmax/Boltzmann exploration selects actions with probability proportional to exp(Q(s, a)/τ), providing smooth probabilistic exploration. Temperature τ controls randomness; high τ approaches uniform random exploration while low τ approaches greedy selection. Upper Confidence Bound (UCB) balances exploration and exploitation by selecting actions with highest uncertainty-adjusted Q-values. Optimistic initialization of Q-values encourages initial exploration. The choice of exploration strategy significantly impacts learning efficiency; sophisticated strategies substantially reduce sample complexity compared to simple ε-greedy.

### Convergence, Limitations, and Deep Q-Learning

Q-learning theoretically converges to the optimal Q* under mild conditions, but convergence in practice depends on proper hyperparameter settings and sufficient exploration. Q-learning faces challenges with large state spaces: the tabular representation requires O(|S||A|) memory and computation. For continuous states, function approximation (neural networks) approximates Q-values: Q(s, a; θ) ≈ Q(s, a). However, neural network approximation can destabilize learning due to non-stationarity of targets (network parameters change during learning). Deep Q-Networks (DQN) address this through experience replay (storing and sampling past transitions) and target networks (slowly updating copies of the main network). These techniques substantially improve convergence and sample efficiency. DQN achieved superhuman performance on Atari games, demonstrating Q-learning's power when combined with modern neural networks. Despite successes, Q-learning and its extensions remain sensitive to hyperparameters and may suffer from overestimation bias or divergence without careful tuning.

### The Bellman Equation and Dynamic Programming

The Bellman equation V(s) = E[R(s) + γV(s')] expresses the value of a state as immediate reward plus discounted future value. This recursive relationship is the foundation of dynamic programming in RL. If transition probabilities and rewards are known (model-based), iterative application of Bellman updates computes optimal values. Value iteration: V_{t+1}(s) = E[R(s) + γ max_a V_t(s')] computes values by repeatedly updating. Policy iteration alternates between evaluating policy (computing values for current policy) and improving policy (greedily selecting best actions). Both converge to optimal values. However, both require knowing model (transition probabilities); Q-learning doesn't, learning from experience. Q-learning directly estimates Q-values satisfying Bellman optimality: Q*(s,a) = E[R + γ max_a' Q*(s',a')]. The connection is deep: Q-learning in tabular form is model-free dynamic programming.

### ε-Greedy, Softmax, and Exploration Strategies

ε-Greedy exploration: take greedy action (highest estimated Q) with probability 1-ε, random action with probability ε. Simple and effective; ε-decay (decreasing ε over time) starts exploratory, becomes exploitative. Decaying ε from 1 to 0 over thousands of episodes balances exploration and exploitation. Softmax exploration: select actions probabilistically proportional to exp(Q(s,a)/τ). Temperature τ controls randomness; τ → 0 approaches greedy, τ → ∞ approaches uniform random. Softmax is smooth unlike ε-greedy's discontinuity. Upper Confidence Bound (UCB) balances exploration and exploitation through uncertainty: prioritize actions with high uncertainty (not yet explored) and high estimated value. UCB: a = arg max [Q(a) + c * √(ln(t) / N(a))], where N(a) is times action a taken, t is total timesteps. This naturally decays exploration as actions are well-understood. Optimistic initialization: initialize Q-values optimistically (high values); incentivizes exploring to confirm or revise. Each strategy has trade-offs: ε-greedy is simple, softmax is smooth, UCB is principled, optimistic initialization is elegant.

### Experience Replay and Target Networks in Deep RL

Applying neural networks directly to Q-learning's temporal difference learning is unstable: targets constantly change as network updates. Experience replay addresses this: store (s, a, r, s') in memory buffer, sample random minibatches for training. Random sampling breaks temporal correlations; IID samples satisfy gradient descent assumptions. Minibatch updates (averaging over multiple samples) reduce variance. Target networks: maintain a separate network for computing targets, updated infrequently (every C steps or slowly with weight decay). During training, main network updates to match targets from fixed target network, providing stable targets. Combining experience replay and target networks stabilizes training substantially; DQN (Deep Q-Network) applies these techniques to Atari games, achieving superhuman performance. Memory requirements are non-trivial: storing millions of transitions requires gigabytes. Prioritized experience replay samples high temporal-difference error transitions more frequently, improving sample efficiency.

### Double Q-Learning and Overestimation

Q-learning tends to overestimate action values: selecting max_{a'} Q(s', a') uses same network for both action selection and evaluation. Double Q-learning decouples selection and evaluation: select action via a* = argmax_a Q(s', a'; θ), evaluate via Q(s', a*; θ_target). This reduces overestimation by ~50% empirically. The improvement is subtle but consistent: overestimation causes agents to prefer actually-bad actions incorrectly. Double Q-learning is simple (one line code change) and significantly improves performance and stability. Dueling Q-networks decompose Q-values into state value V(s) and advantage A(s,a): Q(s,a) = V(s) + A(s,a) - mean_a A(s,a). Advantages (action advantages) are centered; this representation is more stable. Dueling architecture enables networks to focus on value estimates and advantage estimates separately, improving learning. Combining double and dueling Q-learning with prioritized experience replay creates Rainbow DQN achieving near-superhuman Atari performance.