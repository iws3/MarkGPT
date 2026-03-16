# Lesson 9: Key Concepts Deep Dive

## 1. Gradient Alignment and Source-Target Task Similarity

How similar must source and target be? Complete alignment (source task = target task) guarantees transfer. Large differences risk negative transfer. Measuring task similarity helps predict transfer success.

## 2. Feature Learning Dynamics

Different layers of neural networks learn different features. Shallow layers learn low-level features (edges, textures); deep layers learn abstract features (object categories). Transfer benefits from shared abstract feature representations even when low-level variations differ.

## 3. Fine-Tuning Stability

Standard fine-tuning with full networks can catastrophically forget source knowledge. Regularization approaches: reduce learning rate, freeze early layers, or penalize weight changes. Careful tuning of the fine-tuning learning rate is critical.

## 4. Multi-Task Learning Synergy

When do multiple tasks help vs. hurt each other? Tasks with aligned objectives (physics domains) often help each other. Tasks with conflicting objectives (fast vs. safe driving) can hurt. Task clustering—grouping compatible tasks—improves synergy.

## 5. Curriculum Progression Difficulty

Curriculum sequences matter: too easy initial tasks provide little learning signal; too hard initial tasks prevent progress. Gradual difficulty increase enables efficient learning. Automated curriculum methods avoid manual difficulty specification.

## 6. Randomization Domain Variability

How much randomization is enough? Too little: policies overfit to training variations. Too much: randomization includes unrealistic scenarios. Finding the right balance requires domain expertise and empirical validation.

## 7. Sim-to-Real Domain Gap

Common gaps: sensor noise, actuator delays, friction, air resistance. Randomizing all these sources leads to robust policies, but extremely broad randomization can prevent learning. Targeted randomization addressing known sim2real gaps works better.

