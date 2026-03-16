# Lesson 8: Inverse Reinforcement Learning

## Table of Contents

1. [The Inverse RL Problem](#the-inverse-rl-problem)
2. [Reward Ambiguity and Ill-Posedness](#reward-ambiguity-and-ill-posedness)
3. [Maximum Entropy IRL](#maximum-entropy-irl)
4. [Apprenticeship Learning](#apprenticeship-learning)
5. [Generative Adversarial Imitation Learning (GAIL)](#generative-adversarial-imitation-learning-gail)
6. [IRL with Deep Learning](#irl-with-deep-learning)
7. [Preference-Based Learning](#preference-based-learning)
8. [IRL in Complex Domains](#irl-in-complex-domains)
9. [Safety and Alignment via IRL](#safety-and-alignment-via-irl)
10. [IRL Applications and Challenges](#irl-applications-and-challenges)

---

## The Inverse RL Problem

Inverse Reinforcement Learning (IRL) addresses the reverse problem of standard RL. Given a policy (expert demonstrations), infer the reward function that explains it. This is motivated by the observation that directly specifying reward functions is difficult—how would you mathematically encode the objective "drive safely and comfortably"? By observing expert demonstrations, one can infer these implicit objectives.

Formally, given demonstrations τ* = {(s,a) trajectories}, find reward function R such that π* is optimal for R. This is fundamentally different from RL's goal of finding optimal π given R. IRL's goal is finding R given π.

Applications of IRL include: (1) extracting human values/preferences from behavior, (2) training agents by first inferring rewards then optimizing them, and (3) understanding complex behaviors by inferring their objectives. IRL is valuable whenever we want to understand why agents (humans or AI) behave as they do rather than just predict what they'll do.

---

## Reward Ambiguity and Ill-Posedness

The IRL problem is fundamentally ambiguous. Multiple reward functions could explain the same behavior. An expert demonstrating safe driving behavior could be motivated by safety, comfort, compliance with laws, or combinations thereof. All these reward functions could produce identical behavior. This ambiguity means IRL is ill-posed without additional structure.

Consider simple examples: a policy that always stays in a safe state could be motivated by (1) maximizing safety, (2) minimizing cost, or (3) any combination of individual state values that assigns high value to safe states. Without additional information, these are indistinguishable. This non-uniqueness is not a bug but a feature—it recognizes that behavior underdetermines objectives.

Resolving ambiguity requires additional assumptions. Maximum entropy IRL chooses the highest-entropy reward function consistent with demonstrations. This selection principle prefers simpler, more general reward functions over complex ones that perfectly fit the data. Other approaches use Bayesian priors over reward functions or explicitly ask for user guidance (e.g., "is reward A or B more likely?"). The choice of disambiguation strategy significantly impacts learned rewards.

---

## Maximum Entropy IRL

Maximum entropy IRL elegantly resolves reward ambiguity. Among all reward functions that explain expert behavior, choose the one with maximum entropy—the one that constrains behavior as little as possible while remaining consistent with demonstrations.

The formulation: expert demonstrations maximize entropy subject to constraints that expert policy is optimal:

R* = argmax H(π) subject to V(π) = V(expert)

where H(π) is the entropy of policy π. Equivalently, constrain that expected feature counts under the learned policy match those of experts:

E[φ(τ)] = E_expert[φ(τ)]

Feature counts φ(τ) are representations of trajectories (e.g., sum of features φ(s) along trajectory). This constraint ensures the learned policy visits the same kinds of states as the expert.

Maximum entropy IRL has nice properties: (1) the result is a distribution over policies rather than a single policy, capturing behavioral variation in the expert, (2) the choice of maximum entropy is principled, and (3) algorithms have convergence guarantees. However, maximum entropy IRL is computationally expensive—each iteration of the algorithm requires solving an RL problem to find the optimal policy for the current reward.

---

## Apprenticeship Learning

Apprenticeship learning frames IRL as a learning problem: simultaneously learn reward and policy such that the policy's feature counts approach the expert's. The algorithm alternates between: (1) modifying the reward function to increase the gap between expert and current features, and (2) retraining the policy to optimize the new reward.

The algorithm iteratively finds rewards that make the current policy worse than the expert at feature matching. This drives the policy to change behaviors, eventually converging when no such discriminative reward exists (i.e., feature counts match). This is slower than maximum entropy IRL in  theory but often faster in practice because it doesn't require solving full RL problems.

Apprenticeship learning reveals an insight: IRL is really about matching statistics/features of expert behavior rather than recovering exact reward functions. By focusing on features rather than individual state rewards, the problem becomes more tractable. The learned reward simply must assign appropriate weights to features; the specific values of individual state rewards matter only insofar as they affect overall feature count matching.

---

## Generative Adversarial Imitation Learning (GAIL)

GAIL learns a discriminator D to distinguish expert trajectories from learned policy trajectories, simultaneously learning a policy that fools the discriminator. This is elegant: if the learned policy generates indistinguishable trajectories from the expert, it must share the same objective. GAIL implicitly learns both an underlying reward function (the discriminator) and the policy optimizing it.

The min-max objective: the discriminator tries to distinguish expert from learned policy; the policy tries to fool the discriminator. The discriminator provides a differentiable reward signal (discriminator output) to the policy, enabling direct policy optimization. When the discriminator can no longer distinguish trajectories, the learned and expert policy distributions match.

GAIL significantly outperforms behavioral cloning for many tasks. The key advantages: (1) handles distribution shift better than BC by training policy through RL, (2) works with small expert datasets because adversarial training extracts structure, and (3) learned policies have smoother, less erratic behavior than BC policies. GAIL has become a canonical approach combining ideas from IRL and generative modeling.

---

## IRL with Deep Learning

Scaling IRL to complex domains requires deep learning. Deep Maximum Entropy IRL learns reward functions parameterized by neural networks. Deep GAIL scales GAIL to high-dimensional observations by using neural networks for both discriminator and policy.

Deep learning enables learning from high-dimensional inputs like images but introduces new challenges. Neural network reward functions have high capacity—they can fit any behavior but don't generalize well to new states. Regularization and inductive biases are crucial to prevent overfitting. Additionally, training jointly learns feature representations while learning rewards; the reward function is only as meaningful as the learned representation allows.

Deep IRL methods have successfully extracted human preferences from complex behaviors. Applications include autonomous driving (inferring human driving objectives), robotics (inferring task objectives from demonstrations), and recommendation systems (inferring user preferences from engagement). In each case, deep learning enables scale to realistic complexity while IRL enables learning objectives directly from behavior.

---

## Preference-Based Learning

Preference learning generalizes IRL by learning from comparisons between trajectories rather than full demonstrations. Rather than collecting full trajectories, ask humans which of two trajectories they prefer. This is often easier than demonstrating or labeling—humans naturally compare alternatives.

Preference-based IRL learns reward functions such that complete trajectories with high cumulative reward are preferred over low-reward trajectories. A neural network learns to predict preferences from trajectory pairs, and the preference model induces a reward function. This approach has become central to making AI systems aligned with human values—by learning human preferences through comparison rather than demonstration.

Preference learning is less sample-intensive than full demonstration collection and more natural than specifying explicit rewards. Humans readily compare trajectories ("I prefer this driving style to that one") without needing to verbalize objectives. This has been particularly impactful in training large language models, where preference learning from human comparisons enabled learning complex, nuanced human alignment preferences.

---

## IRL in Complex Domains

Applying IRL to complex domains requires handling partial observability, multimodal behavior, and high-dimensional spaces. Partial observability (agent can't see full state) makes demonst behavior harder to interpret—the expert might appear irrational when in fact they're responding to unobserved state. Methods that handle partial observability explicitly model hidden state and infer it from demonstration sequences.

Multimodal behavior (multiple ways to accomplish objectives) is challenging for IRL. Maximum entropy approaches naturally handle multimodality by learning distributions over policies. Ensemble methods maintain multiple possibly-correct reward hypotheses. Directly modeling behavioral modes helps—explicitly learning that one mode represents active behavior, another passive—enables better generalization across modes.

High-dimensional spaces require architectural and algorithmic innovations. Hierarchical IRL decomposes complex tasks into subtasks, learning reward functions for each level of the hierarchy. Causal IRL learns causal graphs explaining behavior, enabling more interpretable and transferable reward functions. These advances move IRL toward scalability comparable with standard deep learning.

---

## Safety and Alignment via IRL

IRL offers a principled approach to AI alignment: learn human objectives through their behavior, then optimize AI systems according to these learned objectives. Rather than engineers speculating about human values, let AI systems learn these values observationally. The hope is this leads to AI systems more closely aligned with actual human intentions.

One key approach: collect diverse human behavior (driving, conversation, decision-making in ethical dilemmas), use IRL to infer underlying objective functions, then use these learned objectives to constrain or guide AI system behavior. This sidesteps the difficult problem of explicit value specification.

However, there are caveats. Behavior underdetermines objectives—learning from biased human data (e.g., discriminatory driving patterns) could infer discriminatory objective functions. IRL only recovers objectives implicit in behavior; humans might not truly intend the behaviors they exhibit (norms, accidents, or external pressures). Despite limitations, IRL-based alignment is an active research direction aiming to make AI systems more trustworthy and aligned with human values.

---

## IRL Applications and Challenges

IRL has been applied successfully to understanding human driving, learning robot manipulation from human demonstration, inferring user preferences in recommendation systems, and large-scale language model alignment. In each case, learning objectives from behavior enabled better generalization and more robust systems than hand-engineered rewards.

Challenges remain. Collected behavior is often biased (doesn't represent true preferences), multimodal (multiple valid objectives), or irrational (humans sometimes behave irrationally). Scaling IRL to even higher dimensions and more complex domains remains difficult. Interpretability of learned rewards is limited—knowing a neural network's output but not what it represents is only partially useful. Theoretical understanding of when IRL is well-posed remains incomplete.

Future directions include: combining IRL with active learning (asking targeted questions), causal inference for learning causal reward structures, multi-agent IRL for inferring objectives in multi-agent settings, and more robust methods for handling distribution shift. The field continues advancing as practitioners tackle increasingly complex real-world applications and theory improves our understanding of identifiability and inference in inverse RL.

