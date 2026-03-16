# Lesson 4: Key Concepts Deep Dive

## 1. Compounding Error in Behavioral Cloning

BC's core issue: small deviations from expert trajectory lead to unfamiliar states, causing more errors. Error compounds temporally—after T steps of small ε errors per step, the total deviation can be huge. This explains catastrophic BC failures in long-horizon tasks.

## 2. Multimodal Behavior Learning

Humans have multiple ways to accomplish tasks—different driving styles, manipulation approaches. BC averages across modes, creating mediocre average policy. GAIL and ensemble methods handle multimodality better by learning distributions over behaviors.

## 3. Expert Data Bias

If expert data reflects biases (e.g., discriminatory driving patterns), learned policies inherit these biases. Pre-filtering, importance weighting, or augmentation can help, but fundamentally requires quality expert data.

## 4. Demonstration Sufficiency

How much expert data is enough? It depends on task complexity and state space size. Empirically, collect diverse demonstrations covering task variety. Testing on held-out demo data indicates when sufficient data exists.

## 5. IRL Interpretability Problem

GAIL learns an implicit reward function (the discriminator). This function is a neural network—interpreting what it rewards is difficult. Black-box reward functions limit understanding and trust.

## 6. One-Shot Learning Requirements

For one-shot learning to work, significant pre-training on related tasks is necessary. Meta-learning primes the system to rapidly adapt using minimal data.

