# Lesson 7: Model-Based Reinforcement Learning - Exercises

## Questions

1. **Define Model-Based Reinforcement Learning and explain its advantages over model-free approaches.**
   - Discuss how learning an environment model enables sample efficiency and planning.

2. **What is a "World Model" or "Environment Model" in MBRL?**
   - Explain the components (state transition dynamics, reward model, termination prediction) and how they are typically learned.

3. **Describe "Dyna" and how it combines model-based and model-free learning.**
   - Explain the interaction between real experience, planning with the learned model, and policy improvement.

4. **What are the challenges of using learned models for planning over long horizons?**
   - Discuss issues like model error accumulation, distribution shift, and pessimism in learned models.

5. **Explain "Model Predictive Control" (MPC) and how it uses forward models for decision-making.**
   - Compare MPC to other planning algorithms like Monte Carlo Tree Search (MCTS) in the context of RL.

6. **How can you quantify and mitigate model uncertainty in MBRL?**
   - Discuss approaches like model ensembles, Bayesian neural networks, and epistemic vs. aleatoric uncertainty.

7. **What is "Latent Space Planning" and how does it enable learning in high-dimensional observation spaces?**
   - Explain how learning compact latent representations supports efficient planning.

8. **Describe "Imagination-Based Learning" where agents imagine rollouts to improve their policies.**
   - How do algorithms like PlaNet and Dreamer use world models for both planning and learning?

9. **Discuss the trade-offs between model learning, value function learning, and policy learning in MBRL.**
   - When should computational budget be allocated to each component?

10. **Design a model-based RL system for a continuous control task (e.g., quadruped locomotion).**
    - Specify the model representation, planning algorithm, exploration strategy, and how you would handle model uncertainty.
