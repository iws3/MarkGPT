# Lesson 6: Hierarchical Reinforcement Learning

## Table of Contents

1. [Motivation for Hierarchical Reinforcement Learning](#motivation-for-hierarchical-reinforcement-learning)
2. [Temporal Abstraction and Options](#temporal-abstraction-and-options)
3. [The Options Framework Formalism](#the-options-framework-formalism)
4. [Learning Options and Option Discovery](#learning-options-and-option-discovery)
5. [Hierarchical Abstract Machines (HAMs)](#hierarchical-abstract-machines-hams)
6. [Feudal Reinforcement Learning](#feudal-reinforcement-learning)
7. [Skill Learning and Reuse](#skill-learning-and-reuse)
8. [Multi-Level Hierarchies](#multi-level-hierarchies)
9. [Credit Assignment in Hierarchies](#credit-assignment-in-hierarchies)
10. [Hierarchical RL Applications](#hierarchical-rl-applications)

---

## Motivation for Hierarchical Reinforcement Learning

Complex sequential decision problems have inherent temporal structure. A robot grasping an object doesn't think about every motor command continuously; it decomposes the task into subtasks (reach toward object, close gripper, lift object). Similarly, a video game player doesn't issue actions at every frame; they think in terms of higher-level goals (get power-up, defeat enemy). Hierarchical RL formalizes this insight.

Hierarchical RL has multiple benefits. First, it enables learning of long-horizon behaviors by decomposing into shorter-horizon subtasks, each solving a simpler problem. Second, learned subtasks (skills) can be reused across different problems, dramatically accelerating learning. Third, hierarchical representations are more interpretable—we can examine what skills the agent learned, understanding its behavior at multiple levels of abstraction. Fourth, hierarchy enables transfer learning—if we learn useful skills in one domain, those skills might transfer to related domains.

The key insight is that not all temporal scales are equally important. High-level decisions (what skill to execute) change slowly while low-level decisions (which motor command) change rapidly. By creating separate hierarchical levels, each operating at an appropriate timescale, learning becomes more tractable.

---

## Temporal Abstraction and Options

The Options Framework formalizes temporal abstraction. An option is an abstraction of a skill or behavior. Like primitive actions, options take states as input and execute until a termination condition is satisfied. But unlike primitive actions which always last one timestep, options extend over multiple timesteps, enabling hierarchical planning and learning.

Temporal abstraction is valuable because it reduces the horizon of decision problems. If primitive actions are taken every timestep but options last on average 10 timesteps, decision problems are 10x shorter, dramatically reducing sample complexity. Additionally, options can exploit the structure of tasks—a "reach-object" option encapsulates the complexity of arm manipulation, allowing higher-level planners to think simply about object relations.

The mathematics of options extends MDPs to have multiple levels. At the top level, an agent chooses which option to execute; while executing an option, the option's internal policy is active. Options naturally decompose long problems into manageable subproblems, enabling sophisticated hierarchical learning and planning.

---

## The Options Framework Formalism

An option ω is formally defined by three components:

1. **Initiation Set I_ω**: The set of states from which the option can be initiated. Some options may only apply in specific contexts.

2. **Policy π_ω(a|s)**: The intra-option policy executed while the option is active. This policy selects primitive actions.

3. **Termination Condition β_ω(s)**: A function specifying the probability of terminating in each state.

Using options, the MDP is augmented to form a Semi-MDP (SMDP). States and options replace primitive states and actions, extending decision timescales. Value functions decompose into option values Q(s, ω) and within-option values Q_ω(s, a). Learning algorithms like Q-learning extend naturally to options, with temporal differences computed over option-steps rather than primitive steps.

The power of this formalism is its generality. Options can be deterministic or stochastic, can have long or short expected durations, and can be specialized to specific regions of the state space. The same learning algorithms used for primitive MDPs extend to option-based hierarchies, providing a unified framework for hierarchical and non-hierarchical learning.

---

## Learning Options and Option Discovery

Learning which options to use is the challenge of option discovery. One approach is to manually specify options based on domain knowledge, but this is brittle and task-specific. Automatic option discovery learns useful options from experience.

Several approaches exist:

- **Skill-based learning**: Simultaneously learn primitive policies and a higher-level policy that coordinates them. Use auxiliary rewards or intrinsic motivation to encourage learning of diverse skills.

- **Graph-based discovery**: Build a state-similarity graph and identify option candidates as behaviors that effectively traverse clusters in this graph.

- **Empowerment-based discovery**: Learn options that maximize agent empowerment—enabling maximum future flexibility in action space.

- **Goal-conditioned discovery**: Learn options conditioned on subgoals, where options are behaviors that reach particular goal states.

Option discovery remains an open research problem. Ideally, the learner would automatically identify useful skill abstractions without domain knowledge, but current methods require careful tuning or significant prior structure. Some approaches combine learned options with manually specified options to get benefits of both.

---

## Hierarchical Abstract Machines (HAMs)

Hierarchical Abstract Machines (HAMs) provide an alternative formalization of hierarchies to options. Rather than options existing in a flat space and being called by a higher-level policy, HAMs embed hierarchy in the policy structure. The policy is defined as a hierarchy of state machines; high-level machines call lower-level machines, forming a stack.

HAMs explicitly constrain which actions are available in each hierarchical state, enforcing task structure. This differs from options which make available all options at each state. The HAM framework enables expressing domain knowledge about appropriate hierarchical structures and has proven useful for encoding task-specific invariants.

Learning in HAMs uses similar techniques to options-based learning but respects the explicit hierarchical structure. HAMs are particularly useful when domain structure is well-understood and an engineer can encode it as a hierarchy of machines. However, HAMs are less flexible and harder to automatically discover than options.

---

## Feudal Reinforcement Learning

Feudal RL uses a hierarchical structure inspired by feudal governance: a manager directs a worker through abstract goals, the worker executes primitive actions, and reward flows through the hierarchy. The manager learns what subgoals to pursue (high-level policy), while the worker learns to achieve subgoals (low-level policy).

The manager outputs goals/subgoals, the worker tries to achieve them by executing primitive actions, and the manager observes worker behavior and adjusts its goal-setting policy. Importantly, the worker receives augmented rewards for achieving manager-set goals in addition to environment rewards. This intrinsic motivation encourages the worker to achieve manager-set subgoals even if they don't directly maximize environment rewards.

Feudal learning naturally handles the credit assignment problem in hierarchies. The manager is credited for setting useful subgoals while the worker is credited for achieving them. This decomposition simplifies learning at each level. Modern variants like hierarchical Q-learning and options-based methods build on feudal intuitions with improved algorithms and guarantees.

---

## Skill Learning and Reuse

A primary motivation for hierarchies is that learned skills can be reused across tasks. If an agent learns to "move to location X" as a general skill during training on one task, this skill can be deployed when learning related tasks. Skill reuse dramatically accelerates learning because the learner doesn't need to rediscover basic behaviors for each new task.

For skill reuse to work, skills must generalize across task variations. A "grasp object" skill needs to generalize to different object shapes, sizes, and weights. This requires skills to be learned in ways that capture underlying principles rather than memorizing task-specific variants. Meta-learning combined with hierarchies learns skills that rapidly adapt to new variations, improving generalization.

Transfer learning through skill reuse is advancing practical hierarchical RL. A robot can be pre-trained to learn useful manipulation skills, then quickly adapt these skills to new assembly tasks. Similarly, video game players can learn to string together learned strategies to handle new game scenarios. As skill libraries accumulate, new tasks become learnable from minimal interaction, moving toward more general and efficient AI systems.

---

## Multi-Level Hierarchies

Some problems have natural multi-level hierarchies. A robot learning manufacturing tasks might have levels: primitive actions (move motor), mid-level skills (reach position, grasp object), and high-level skills (assemble part, move assembly). Decisions at each level operate at different timescales: primitive actions at millisecond scale, mid-level skills at second scale, high-level skills at minute scale.

Multi-level hierarchies are more expressive than two-level hierarchies but harder to learn. Options theory scales naturally to multiple levels, with options at each level calling options at the next lower level. Learning becomes a challenge because credit must flow through multiple levels, requiring careful algorithm design. Some approaches learn bottom-up (learning primitive skills first, then higher levels), while others learn jointly across all levels.

Multi-level hierarchical RL remains largely unsolved compared to two-level hierarchies. However, successes like locomotion hierarchies (high-level: navigation targets; mid-level: gaits; low-level: motor control) suggest that multi-level learning is achievable with careful design.

---

## Credit Assignment in Hierarchies

Credit assignment becomes complex in hierarchies. When a high-level action (executing an option) leads to good outcomes, should credit go to the option or to the primitive actions taken? Should the option be credited for choosing good primitive actions, or should it be credited for the long-term consequences of its choices? Incorrect credit assignment leads to learning failures where agents learn useless options or fail to improve.

Solutions include:

- **Temporal difference learning**: Use TD learning to compute returns at each hierarchical level, naturally decomposing credit through the hierarchy.

- **Option-specific value functions**: Learn separate value functions for each option, enabling precise credit assignment within options.

- **Eligibility traces**: Use eligibility traces to propagate credit through time and hierarchy, similar to how traces work in standard RL.

- **Intrinsic motivation**: Augment rewards to provide immediate credit at each level, encouraging learning of useful options.

Sophisticated option learning algorithms spend significant effort on credit assignment, recognizing it as critical to successful hierarchical learning. Modern hierarchical methods handle credit assignment better than early approaches, enabling deeper hierarchies.

---

## Hierarchical RL Applications

Hierarchical RL has achieved impressive results in robotics, game playing, and autonomous systems. In robotics, hierarchy enables learning of complex manipulation tasks by decomposing into subtasks. In Atari games, hierarchical learning discovers strategies at multiple timescales. In navigation, hierarchy natural separates high-level path planning from low-level obstacle avoidance.

Real-world applications benefit from this structure. A robotic system learning manufacturing tasks can reuse learned subtasks across different assembles. A game-playing AI can reuse learned tactics in new game variants. An autonomous vehicle can reuse learned driving skills across different cities. Hierarchy's ability to decompose complex problems and enable skill reuse makes it especially valuable for practical systems.

However, hand-engineering hierarchies is brittle and task-specific. Moving toward automatic hierarchy discovery remains a key frontier. Additionally, deploying hierarchical systems requires careful system design to handle safety (ensuring subtasks don't cause dangerous behaviors) and robustness (ensuring learned hierarchies don't fail when deployed in new variation). Success in hierarchical RL depends on combining learning advances with careful engineering.

