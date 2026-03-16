# Lesson 7: Model-Based Reinforcement Learning

## Table of Contents

1. [Introduction to Model-Based RL](#introduction-to-model-based-rl)
2. [World Models and Environment Dynamics](#world-models-and-environment-dynamics)
3. [Reward Prediction and Termination](#reward-prediction-and-termination)
4. [Learning Models from Interaction](#learning-models-from-interaction)
5. [Planning with Learned Models](#planning-with-learned-models)
6. [Dyna: Integrating Planning and Learning](#dyna-integrating-planning-and-learning)
7. [Model Predictive Control](#model-predictive-control)
8. [Latent Space Models and Imagination](#latent-space-models-and-imagination)
9. [Model Uncertainty and Exploration](#model-uncertainty-and-exploration)
10. [Challenges and State-of-the-Art MBRL](#challenges-and-state-of-the-art-mbrl)

---

## Introduction to Model-Based RL

Model-based RL explicitly builds a model of the environment's dynamics and uses this model to plan optimal actions. Rather than learning what to do through experience in the actual environment, the agent learns what the environment does (how it responds to actions), then reasons about that model to find good actions. This approach can be dramatically more sample-efficient than model-free methods because agents can learn from simulated rollouts using their learned model.

The sample efficiency advantage is profound. An agent might run thousands of simulated rollouts through its learned model at minimal computational cost, whereas each real environment interaction takes time. If the model is accurate, learning from simulated data is nearly as valuable as learning from real data. This became transformative in domains like robotics where real-world interaction is expensive, and game playing where simulation is available but computational.

However, model-based RL has counterbalancing challenges. Building accurate models is itself a learning problem, and model errors compound during planning over long horizons. Additionally, planning in large spaces is computationally expensive. Model-based methods work best in domains where accurate models are learnable and computational budget is available. The field has matured to the point where model-based and model-free methods are often combined for best results.

---

## World Models and Environment Dynamics

A world model predicts future states given current state and actions. The transition model p(s'|s, a) captures environment dynamics, outputting the probability distribution of next states. In deterministic environments, deterministic models suffice (s' = f(s, a)); in stochastic environments, probabilistic models are needed to capture the range of possible outcomes.

Learning transition models from experience is a supervised learning problem. Collect transitions (s, a, s'), then train a function approximator (typically a neural network) to predict s' given (s, a). The challenge is learning accurate models in high-dimensional spaces. For image-based control, learning pixel-space models is inefficient (the model must learn low-level stochasticity like shadows that doesn't matter for control). Learning models in learned latent spaces is more effective.

Transition models come in multiple forms: (1) **deterministic models** that predict a single next state, (2) **stochastic models** that predict a distribution over next states, (3) **latent models** that work in compressed latent state spaces, and (4) **ensemble models** that maintain multiple model hypotheses. Ensemble models are particularly valuable for quantifying model uncertainty—if multiple models make different predictions, that indicates high uncertainty at that state.

---

## Reward Prediction and Termination

Beyond dynamics, models must predict rewards and termination conditions. The reward function r(s, a) or r(s, a, s') predicts immediate rewards from actions. Learning this is again supervised learning—but reward signals are often sparse (only at episode end) making learning challenging.

Termination prediction is subtle but important. In finite-horizon or episodic tasks, the model must predict when episodes end. This is often learned as a binary classification (terminal vs. non-terminal state) from episode boundaries. For continuing tasks, termination is not relevant. Inaccurate termination predictions cause planning to fail because the model believes trajectories extend longer (or end sooner) than they actually do.

The three components—dynamics, reward, termination—together form a complete model enabling full trajectory simulation. These are often learned jointly or separately depending on the domain. Learning a single unified model is more efficient but harder; learning separate models for each component is modular and enables easier debugging but requires more overall modeling effort.

---

## Learning Models from Interaction

Learning accurate models requires diverse data. If the agent only interacts in a narrow region of the state space, the learned model will be accurate there but fail in new regions. This creates a version of the exploration problem—the agent must explore to collect data for model learning, even if exploration doesn't directly maximize reward.

Exploration strategies for model-learning include:
- **Random exploration**: Take random actions to cover state space broadly.
- **Uncertainty-driven exploration**: Prioritize exploring states where the model is uncertain, improving model accuracy in those regions.
- **Curiosity-driven exploration**: Take actions that lead to states the model finds surprising, again targeting model uncertainty.

Model learning must be interleaved with planning—the agent plans using its current model, acts in the environment, collects data, updates the model, then replans. Early in learning, the model is inaccurate so planning should be conservative; as the model improves, planning can be more aggressive. This interleaving is the foundation of practical model-based RL.

---

## Planning with Learned Models

Planning uses the learned model to find good action sequences. Several approaches exist:

- **Trajectory optimization**: Formulate planning as optimization over action sequences, finding actions that maximize predicted cumulative reward. Uses gradient descent or other optimization techniques.

- **Tree search**: Build a search tree where nodes represent states and edges represent actions, exploring the tree to depth equal to the planning horizon using techniques like Monte Carlo Tree Search.

- **Shooting methods**: Randomly sample action sequences, evaluate them using the model, and select the best sample (or iteratively improve).

- **Cross-entropy method**: Maintain a distribution over promising action sequences, iteratively focusing the distribution on high-reward sequences.

The choice of planning algorithm affects performance. Tree search scales poorly to large action spaces but works well with small spaces. Trajectory optimization scales better but requires differentiable models. Shooting methods are simple and work with any model.

---

## Dyna: Integrating Planning and Learning

Dyna elegantly integrates model-free learning and planning. The agent simultaneously (1) experiences the real environment, (2) learns a model from this experience, and (3) performs planning using the learned model. After each real experience:

1. Execute experience (s, a, r, s') in the environment.
2. Update Q-values using this real experience (Q-learning update).
3. Update the learned model with this experience.
4. Perform multiple planning steps using the model:
   - Randomly sample previously visited state-action pairs.
   - Use the model to predict next states and rewards.
   - Update Q-values as if these imagined experiences were real.

Dyna benefits from both real and imagined experience. Real experience improves both the value function and the model. Imagined experience exploits the model, efficiently extracting knowledge from limited real data. This combination is dramatically more sample-efficient than pure Q-learning.

The elegance of Dyna is its simplicity. The algorithm just alternates between Q-learning and using Q-learning on imagined data. Dyna-style planning has been extended to modern algorithms with good results—planning steps using learned models consistently improve sample efficiency of deep RL algorithms.

---

## Model Predictive Control

Model Predictive Control (MPC) is a planning method that directly optimizes action sequences online. At each timestep, MPC optimizes the next H steps to maximize predicted return, executes the first action, then at the next timestep re-optimizes. This receding horizon approach naturally handles short-term inaccuracies in the model because replanning continuously corrects for errors.

MPC has several nice properties: (1) it  only requires a model (not a value function), (2) it's very flexible about action sequences, (3) constraint satisfaction is natural (just add constraints to the optimization), and (4) receding horizon replanning handles model errors. However, MPC is computationally expensive—optimizing action sequences every timestep can be prohibitive in low-compute environments.

In robotics, MPC has proven highly successful for continuous control. Many robotic systems use MPC combined with learned models to achieve impressive performance. The computational requirements are mitigated by using learned models (faster than physics simulators) and approximate optimization (not finding globally optimal action sequences, just good ones). MPC remains a practical choice for systems where computation is available and accuracy is important.

---

## Latent Space Models and Imagination

Learning pixel-space models (predicting next pixels from current pixels) is inefficient because pixels contain nuances (shadows, reflections) that don't matter for control. Latent space models learn a compact representation and model dynamics in this representation. The model learns:

1. An encoder e(x) mapping observations to latent states z.
2. A transition model p(z'|z, a) predicting latent state evolution.
3. A decoder d(z) reconstructing observations from latents.

Planning happens in latent space (computational efficient), and decoded observations provide visual feedback. World Models and Dreamer both use this approach successfully. By working in learned low-dimensional spaces, models become much more accurate and sampling is more efficient.

Additionally, latent space models enable "imagination-augmented" learning: agents can imagine trajectories in latent space and learn directly from imagined experience. This is the core of Dreamer—use a latent model to imagine trajectories, then optimize a policy and value function using imagined rollouts. The resulting algorithms are sample-efficient and can learn from high-dimensional observations.

---

## Model Uncertainty and Exploration

Model errors are inevitable—no learned model perfectly predicts the true environment. Over long planning horizons, errors compound: incorrect predictions lead to unforeseen states, which accumulate more errors. Understanding and managing model uncertainty is crucial for MBRL.

Approaches to uncertainty quantification:
- **Ensemble models**: Train multiple models, use their disagreement as uncertainty estimate.
- **Bayesian models**: Use probabilistic models that provide confidence intervals.
- **Epistemic vs. aleatoric uncertainty**: Distinguish uncertainty from true randomness (aleatoric) from uncertainty about the model (epistemic).

Uncertainty drives exploration: explore regions where the model is uncertain to improve the model. This creates a virtuous cycle: better model → better planning → reaching more diverse states → better model learning. Balancing exploration for model improvement vs. exploitation using the current model determines overall learning efficiency.

Pessimistic planning—when model uncertainty is high, optimize conservatively—improves robustness. Rather than assuming the best-case model predictions, assume worst-case predictions, leading to conservative policies. This improves reliability when models are uncertain but can be overly conservative.

---

## Challenges and State-of-the-Art MBRL

Model-based RL faces several key challenges. Compounding errors make long-horizon planning unreliable; solutions include short-horizon planning, conservative planning, and frequent replanning. High-dimensional spaces make models harder to learn; solutions include latent space models and modular models. Computational cost of planning can be high; solutions include approximate optimization and learned optimization. Model learning is itself expensive; solutions include active learning targetingimportant regions.

State-of-the-art MBRL systems often combine multiple approaches. Dreamer demonstrates that latent space models + imagination-augmented policy learning + value learning can achieve impressive results. PETS (Probabilistic Ensembles with Trajectory Sampling) shows that ensemble models + trajectory sampling creates a simple yet effective system. PlaNet uses a learned world model for both planning and learning abstract representations.

The field continues advancing. Recent trends include: combining model-based and model-free learning for better sample efficiency and final performance, using transformers for world models, learning causal world models that support intervention, and meta-learning of model priors that improve model learning efficiency. MBRL is increasingly important as computational budgets for real-world learning grow and sample efficiency becomes paramount.

