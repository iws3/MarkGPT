# Lesson 1: Reinforcement Learning Fundamentals

## Table of Contents

1. [Introduction to Reinforcement Learning](#introduction-to-reinforcement-learning)
2. [Markov Decision Processes (MDPs)](#markov-decision-processes-mdps)
3. [States, Actions, and Rewards](#states-actions-and-rewards)
4. [Value Functions and Bellman Equations](#value-functions-and-bellman-equations)
5. [Policies and Optimal Policies](#policies-and-optimal-policies)
6. [Exploration vs. Exploitation](#exploration-vs-exploitation)
7. [Temporal Difference Learning](#temporal-difference-learning)
8. [Q-Learning Fundamentals](#q-learning-fundamentals)

---

## Introduction to Reinforcement Learning

Reinforcement Learning (RL) is a paradigm in machine learning where an agent learns to make sequential decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties, allowing it to gradually improve its decision-making strategy. Unlike supervised learning, which relies on labeled data, RL enables learning through trial-and-error and environmental feedback. This approach has proven remarkably effective in domains ranging from game playing (Atari, Go) to robotics and autonomous systems.

The core idea of RL is simple: an agent observes the current state of the environment, takes an action, receives a reward, and transitions to a new state. Through repeated interactions, the agent learns which actions lead to better long-term outcomes. This trial-and-error approach mirrors how humans and animals learn from their environment, making RL particularly intuitive and powerful for real-world problems where explicit reward signals are available.

---

## Markov Decision Processes (MDPs)

An MDP is the mathematical framework used to formalize RL problems. It consists of five key components: a set of states (S), a set of actions (A), a transition function (P), a reward function (R), and a discount factor (γ). The fundamental property of MDPs is the **Markov Property**: the future state depends only on the current state and action, not on the entire history of past states.

Formally, an MDP is defined as:
- **States (S)**: The set of all possible configurations the environment can be in.
- **Actions (A)**: The set of all possible actions the agent can take.
- **Transition probability (P)**: P(s'|s,a) gives the probability of reaching state s' from state s by taking action a.
- **Reward function (R)**: R(s,a) gives the immediate reward for taking action a in state s.
- **Discount factor (γ)**: A value between 0 and 1 that determines the importance of future rewards relative to immediate rewards.

The Markov Property is crucial because it allows us to make decisions based only on the current state, greatly simplifying the problem. Without it, the agent would need to track the entire history of states, making the problem computationally intractable for most real-world scenarios.

---

## States, Actions, and Rewards

The state space represents all possible configurations or observations the agent can make about its environment. States can be discrete (finite number of states like chess board positions) or continuous (infinite-dimensional spaces like robotic arm configurations). The representation of states is critical to RL performance—a well-designed state representation helps the agent learn efficient policies, while a poor one can make learning impossible.

Actions are the decisions the agent can make at each step. Like states, actions can be discrete (e.g., moving up, down, left, right in a grid world) or continuous (e.g., motor commands for a robot). The size of the action space affects the learning difficulty; larger action spaces require more exploration to find good actions.

Rewards provide the signal that guides learning. The reward function R(s, a) or R(s, a, s') specifies the immediate numerical feedback the agent receives for its actions. Importantly, rewards should be designed to reflect the agent's true objectives. Poorly designed rewards can lead to pathological learning behaviors where the agent optimizes the stated reward but fails to achieve the actual goal—a phenomenon known as reward hacking.

---

## Value Functions and Bellman Equations

The value function V(s) represents the long-term expected return (cumulative discounted reward) the agent can achieve starting from state s, assuming it follows a given policy. The value function decomposes according to the **Bellman Equation**, which recursively relates the value of a state to the values of its successor states:

V(s) = E[R(s, a) + γV(s')]

This elegant recursive relationship is the foundation of many RL algorithms. The Bellman equation states that the value of a state is the immediate reward plus the discounted value of the next state. By iteratively applying this equation, we can compute accurate value estimates even for very long horizons.

The **action-value function** Q(s, a) extends this idea to represent the expected return from taking action a in state s and then following the optimal policy:

Q(s, a) = E[R(s, a) + γ max Q(s', a')]

The relationship between V and Q is simple: V(s) = max_a Q(s, a), meaning the value of a state is the maximum action-value across all available actions. Understanding this relationship is crucial for grasping how Q-learning and related algorithms work.

---

## Policies and Optimal Policies

A policy π is a mapping from states to actions that determines which action the agent takes in each situation. Policies can be deterministic (π(s) → a) or stochastic (π(a|s) → probability distribution over actions). A stochastic policy is often written as π(a|s), representing the probability of taking action a given state s.

An **optimal policy** π* is one that maximizes the expected cumulative reward. In MDPs, there always exists at least one optimal policy, and all optimal policies achieve the same value function V*(s) for every state. The goal of RL algorithms is typically to find or approximate an optimal policy.

The relationship between policies and value functions is bidirectional:
- **Policy Evaluation**: Given a policy, compute its value function.
- **Policy Improvement**: Given a value function, find a better policy by acting greedily with respect to the value function.

This policy iteration framework forms the basis of many classical RL algorithms like Value Iteration and Policy Iteration, which guarantee convergence to optimal policies in finite state spaces with known dynamics.

---

## Exploration vs. Exploitation

Exploration vs. Exploitation (EE) is a fundamental dilemma in RL. Exploitation means using the current knowledge to maximize immediate reward, selecting actions known to be good. Exploration means trying new, untested actions to potentially discover better strategies. Over-emphasizing exploitation leads to suboptimal solutions if the agent gets stuck in local optima; over-emphasizing exploration wastes time on clearly bad actions.

Common strategies for managing the EE trade-off include:
- **ε-greedy**: With probability ε, take a random action (explore); otherwise, take the greedy action (exploit).
- **Upper Confidence Bound (UCB)**: Balance optimism in the face of uncertainty by maintaining confidence intervals around action values.
- **Thompson Sampling**: Use Bayesian posterior distributions to sample plausible action values and act optimally with respect to samples.
- **Entropy Regularization**: Encourage the policy to maintain high entropy (act randomly) initially, then decay this bonus over time.

The exploration challenge is especially pronounced in sparse reward environments where good learning trajectories are rare. Without sufficient exploration, agents may never discover that rare sequence of actions leading to high reward states.

---

## Temporal Difference Learning

Temporal Difference (TD) Learning is a powerful framework that combines ideas from Dynamic Programming and Monte Carlo methods. TD methods update value estimates based on other learned value estimates rather than waiting until the end of an episode. This enables learning from incomplete episodes and makes TD methods particularly sample-efficient.

The simplest TD algorithm is **TD(0)**, which updates the value of a state using the observed reward and the bootstrapped estimate of the next state:

V(s) ← V(s) + α[R(s, a) + γV(s') - V(s)]

The term [R(s, a) + γV(s') - V(s)] is called the **TD error**. When the TD error is positive, it means the observed reward was better than expected, so we increase our value estimate. When it's negative, we decrease the estimate.

The key insight of TD learning is **bootstrapping**: we use our current estimate of V(s') to improve our estimate of V(s). This is both an advantage and a disadvantage. The advantage is sample efficiency—we learn from partial episodes. The disadvantage is that we can propagate errors if our bootstrap estimates are poor. Despite this theoretical concern, TD methods empirically outperform Monte Carlo approaches in most practical scenarios.

---

## Q-Learning Fundamentals

Q-learning is one of the most influential RL algorithms. It learns the action-value function Q(s, a) directly, which can be used to derive an optimal policy by always selecting the action with maximum Q-value in any state.

The Q-learning update rule is:

Q(s, a) ← Q(s, a) + α[R(s, a) + γ max_a' Q(s', a') - Q(s, a)]

This is an off-policy algorithm, meaning it can learn an optimal policy while following a different (exploratory) policy. This property makes Q-learning incredibly flexible and practical. For example, the agent might follow an ε-greedy exploration policy while learning the optimal deterministic policy.

Q-learning learns using "bootstrapping"—it updates Q estimates using other Q estimates. The algorithm is guaranteed to converge to optimal Q-values in tabular settings (finite state and action spaces) under mild conditions. However, Q-learning can be unstable when combined with function approximation (e.g., neural networks), leading to the development of more sophisticated variants like Double Q-learning and Dueling Q-networks.

The beauty of Q-learning lies in its simplicity and effectiveness. It requires minimal assumptions about the environment (only that it's an MDP) and can be applied to a wide range of problems. This generality, combined with its strong theoretical backing, makes Q-learning a cornerstone algorithm that every RL practitioner must understand.

