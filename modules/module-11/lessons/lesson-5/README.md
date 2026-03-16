# Lesson 5: Meta-Learning and Few-Shot Learning

## Table of Contents

1. [Fundamentals of Meta-Learning](#fundamentals-of-meta-learning)
2. [Learning to Learn Framework](#learning-to-learn-framework)
3. [Few-Shot Learning Problem Definition](#few-shot-learning-problem-definition)
4. [Model-Agnostic Meta-Learning (MAML)](#model-agnostic-meta-learning-maml)
5. [Optimization-Based Meta-Learning](#optimization-based-meta-learning)
6. [Prototypical Networks and Metric Learning](#prototypical-networks-and-metric-learning)
7. [Matching Networks](#matching-networks)
8. [Task Distributions and Domain Shift](#task-distributions-and-domain-shift)
9. [Meta-Learning for Reinforcement Learning](#meta-learning-for-reinforcement-learning)
10. [Continual and Online Meta-Learning](#continual-and-online-meta-learning)

---

## Fundamentals of Meta-Learning

Meta-learning, or "learning to learn," is the ability to improve learning itself through experience across multiple tasks. Rather than training on a single task, an agent encounters many related tasks and learns to quickly adapt to new members of that task family. The classical ML paradigm is single-task learning; meta-learning extends this to multi-task learning where this generalization across tasks is explicitly optimized.

The motivation for meta-learning is profound. Humans quickly learn new tasks by leveraging prior experience—a child who has learned to identify dogs can quickly learn to distinguish dog breeds from a few examples. Similarly, meta-learning aims to enable agents that improve their learning ability through experience, eventually learning new tasks from minimal data. This is particularly valuable in domains where collecting large labeled datasets is expensive.

Meta-learning has two timescales of adaptation: (1) **outer loop** where the meta-learner improves across tasks through experience, and (2) **inner loop** where on each task the learner adapts quickly using task-specific data. By optimizing the outer loop to enable fast inner loop adaptation, meta-learning learns learning algorithms rather than just task solutions. This perspective shift from learning tasks to learning learning algorithms is profound.

---

## Learning to Learn Framework

The meta-learning framework formalizes this idea. A learner receives a meta-training set of tasks, each with a small amount of labeled data. For each task, the learner attempts to solve it quickly (inner loop adaptation), then the meta-learner evaluates performance and updates to improve future adaptation (outer loop optimization).

Formally, tasks come from a task distribution p(T). Each task T consists of a training set D^train and test set D^test. During meta-training, the learner samples a task, sees samples from D^train, makes predictions on D^test, and the metalearner observes performance. The meta-objective is to minimize average test error across tasks:

min E_T[L(learner(D^train_T), D^test_T)]

This outer-loop objective directly optimizes for fast learning. A learner that can solve tasks from small training sets will have low meta-loss. The key insight is that by experiencing diverse tasks during meta-training, the learner acquires inductive biases that generalize to new tasks from the same distribution.

Different meta-learning approaches vary in (1) how the learner adapts to new tasks (optimization, metric learning, or hypernetworks), and (2) how the meta-learner optimizes across tasks (gradient-based or evolutionary). The framework is general and has been applied to supervised learning, reinforcement learning, and robotics with impressive results.

---

## Few-Shot Learning Problem Definition

Few-shot learning is the setting where a learner must achieve good performance on a new task from a small number of labeled examples—typically 1 to 10 examples (called shots). In the "N-way K-shot" classification setting, the agent faces N classes with K labeled examples each and must classify new test examples. Common benchmarks include 5-way 1-shot (learn distinguishing 5 classes from 1 example each) or 5-way 5-shot.

Few-shot learning is motivated by human learning. Humans can learn new object categories from just one or two examples, a capability that standard machine learning finds extremely challenging. Enabling machines to achieve this would be transformative for applications where data is scarce. Few-shot learning has become a hallmark capability sought in modern machine learning systems.

The few-shot setting constrains the amount of task-specific data, forcing learners to rely on prior knowledge acquired during meta-training. A model trained on ImageNet classification might be fine-tuned on a few-shot animal classification task by observing just a handful of examples per species. Similarly, in NLP, a language model might be adapted to a new task style from a few examples. Few-shot learning naturally emerges from meta-learning when the inner-loop adaptation phase uses minimal data.

---

## Model-Agnostic Meta-Learning (MAML)

MAML is an elegant optimization-based meta-learning algorithm. The core idea is to learn an initialization θ such that after one or a few gradient steps on a task, the resulting parameters achieve low loss. This can be achieved by optimizing θ to minimize the task loss after a gradient step:

θ' = θ - α∇L(θ)
min E_T[L(θ')] = min E_T[L(θ - α∇L(θ))]

This meta-objective is optimized through bi-level optimization. For each meta-training task, we compute the gradient ∇L(θ) at the current parameters, simulate the inner-loop update to get θ', compute the test loss at θ', then backpropagate through the simulation to update θ.

The brilliance of MAML is its model-agnosticism. MAML doesn't care what learning algorithm or model architecture is used—MAML only sees gradients. This enables MAML to work with any differentiable learner and makes it practical for diverse applications. MAML learns good initializations for new task learning; the resulting initialized parameters are already well-positioned to solve tasks from the metatask distribution.

MAML has shown remarkable results in few-shot learning, metareinforcement learning, and domain adaptation. However, MAML requires computing second-order derivatives (hessians), which is computationally expensive. First-order variants like FOMAML approximate the meta-gradient without computing hessians, dramatically improving computational efficiency while maintaining most of MAML's performance.

---

## Optimization-Based Meta-Learning

Beyond MAML, optimization-based methods learn more structured adaptation procedures. Some methods learn the optimization algorithm itself (e.g., learning the learning rate or update direction as a neural network output). Others learn low-dimensional adaptation parameters—while the main model remains fixed, a small set of parameters are adapted per task, enabling rapid task-specific fine-tuning.

Hypernetworks exemplify this approach: a small meta-network generates task-specific parameters for a primary network based on the task's training data. This enables rich task-specific adaptation while maintaining a shared representation. Adapter modules take a different approach: small learnable components are inserted into a pre-trained network, and only these small adapters are optimized per task while the main network remains frozen.

The common thread in optimization-based meta-learning is that the approach learns not just what knowledge to transfer, but how to transfer it. Rather than discovering good initial parameters (like MAML), these methods learn adaptation mechanisms that enable efficient transformation of the base model into a task-specific variant. This flexibility often outperforms fixed fine-tuning strategies, especially when adaptation data is very limited.

---

## Prototypical Networks and Metric Learning

Prototypical Networks take a different approach based on learning a good metric for comparing examples. The key idea: a class is modeled by a "prototype" (typically the mean of class examples in embedding space). To classify a test example, compute its distance to each prototype and assign it to the nearest class.

During meta-training, the network learns an embedding space where examples from the same class are close and examples from different classes are far. This metric learning objective naturally transfers to new classes. At test time, compute new class prototypes from the few available examples and classify test examples using learned distance. Prototypical Networks require only forward passes for adaptation—no gradient computation—making them efficient.

The elegance of Prototypical Networks is that classification naturally decomposes into two stages: (1) learn good representations (during meta-training), (2) classify test examples based on distances to class prototypes (at test time). This decomposition is different from optimization-based methods which learn adaptation procedures. Metric-learning approaches tend to be faster at test time but sometimes achieve lower accuracy than optimization-based methods because they use simpler adaptation mechanisms.

---

## Matching Networks

Matching Networks learn to solve few-shot classification through attention mechanisms. Rather than computing distance to prototypes, Matching Networks use an attention-based matching process. Test examples attend to training examples, computing weighted combinations of their labels. The architecture learns what aspects of examples are important for matching.

The key innovation is that matching is learned rather than fixed. While Prototypical Networks use Euclidean distance, Matching Networks learn a learned attention mechanism that can capture more complex similarities. This flexibility often improves performance, especially when data characteristics don't align with Euclidean distance assumptions.

Matching Networks can be viewed as learning a task-conditioned metric optimized for matching—the metric learned for one task is different from that learned for another. This task-awareness enables better generalization than fixed metrics. Matching Networks and Prototypical Networks represent the metric-learning branch of few-shot learning, emphasizing learned similarity functions as the core mechanism for rapid adaptation.

---

## Task Distributions and Domain Shift

The distribution of meta-training tasks dramatically affects few-shot learning performance. A model trained on ImageNet few-shot tasks (dog breeds, flower species) may perform poorly on medical image few-shot tasks if the transition involves different visual characteristics and task semantics. Domain shift in meta-learning encapsulates how characteristics of meta-test tasks differ from meta-training tasks.

Successful meta-learning requires that meta-training and meta-test task distributions be closely aligned. If meta-training tasks are quite different from deployment tasks, the learned learning algorithm will be poorly optimized for the deployment setting. This motivated curriculum learning in meta-learning contexts: organize meta-training tasks from easy to hard, allowing the meta-learner to incrementally adjust to diverse task types.

Domain randomization and task augmentation extend the meta-training distribution, improving robustness to domain shift. By including diverse visual augmentations, procedural variation, and synthetic data during meta-training, learned algorithms become more adaptable to domain-shifted test tasks. Understanding domain shift in meta-learning is crucial for practical applications where deployment and training task distributions mismatch.

---

## Meta-Learning for Reinforcement Learning

Meta-RL aims to learn policies that quickly adapt to new RL tasks. Standard RL requires extensive environment interaction to learn a good policy. Meta-RL pretrains on a distribution of tasks, enabling rapid policy adaptation to new tasks from minimal interaction. This is valuable in robotic settings where real-world interaction is expensive and changing task scenarios are common.

Applying optimization-based meta-learning (MAML) to RL is natural: learn a policy initialization that becomes well-adapted to new tasks after a few policy gradient steps. Similarly, metric-learning approaches define task-conditioned policy embeddings where similar tasks result in similar policies. Some methods learn to select among a library of learned behaviors, effectively choosing which skill to deploy in a new task.

Meta-RL has achieved impressive results in simulated robotic manipulation and locomotion. Policies learned through meta-RL adapt to new environments, morphologies, or task variations significantly faster than standard RL. However, challenges remain in sim-to-real transfer and in scaling meta-RL to complex high-dimensional environments. Meta-RL remains an active research frontier as roboticists seek algorithms that learn efficiently from limited real-world data.

---

## Continual and Online Meta-Learning

Continual (or lifelong) meta-learning extends the basic meta-learning setting to sequential task learning. Rather than receiving all meta-training tasks at once, the learner encounters tasks sequentially over time. The challenge is learning continuously from new tasks while maintaining (or improving) performance on previously encountered tasks—avoiding catastrophic forgetting.

Online meta-learning is minimally more constrained than standard meta-learning: the learner receives tasks sequentially and must perform well immediately without revisiting earlier tasks. This mirrors more realistic learning scenarios than batch meta-learning, where all tasks are available simultaneously. Online meta-learning requires careful memory management, regularization to prevent forgetting, and mechanisms for detecting when to adapt versus maintain current knowledge.

Continual meta-learning is advancing toward more realistic AI systems that learn throughout their operational lifetime. A deployed robot might encounter new environments and tasks continuously, requiring simultaneous optimization for immediate performance and future learning efficiency. This setting is intrinsically harder than standard meta-learning but represents a more realistic goal for practical AI systems.

