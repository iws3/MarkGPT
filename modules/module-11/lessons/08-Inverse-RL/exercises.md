# Lesson 8: Inverse Reinforcement Learning - Exercises

## Questions

1. **What is Inverse Reinforcement Learning (IRL) and how does it differ from forward reinforcement learning?**
   - Explain the goal of learning reward functions from observed behavior and its applications.

2. **Define the "IRL Problem" formally and discuss why it is inherently ambiguous or ill-posed.**
   - Explain the concept of "reward ambiguity" where multiple reward functions explain the same behavior equally well.

3. **Describe "Maximum Entropy IRL" and how it resolves reward ambiguity.**
   - Explain entropy maximization over policies consistent with expert behavior and its theoretical properties.

4. **How does "Generative Adversarial Imitation Learning" (GAIL) use adversarial training for IRL?**
   - Compare GAIL to maximum entropy IRL in terms of optimization and sample efficiency.

5. **Explain "Apprenticeship Learning" and the Algorithms for Inverse RL (AIRE) framework.**
   - How can an agent iteratively match expert behavior by improving its policy based on learned rewards?

6. **What are the computational challenges of IRL, and how can deep learning improve scalability?**
   - Discuss how neural networks parameterize reward functions and address high-dimensional feature spaces.

7. **How can IRL handle sparse or limited expert demonstrations?**
   - Discuss approaches to handle ambiguity and learn robust reward functions from small datasets.

8. **Explain the difference between "Preference-Based IRL" and trajectory-based IRL.**
   - When might preference learning be more practical than demonstrating full trajectories?

9. **Discuss the application of IRL to "Safe RL" where the reward function encodes safety objectives.**
   - How can IRL help align AI systems with human values and intentions?

10. **Design an IRL system that learns complex human preferences for autonomous vehicle decision-making.**
    - Specify the state/action representation, how expert demonstrations or preferences are collected, and how the learned reward function ensures safe behavior.
