# Lesson 5: Key Concepts Deep Dive

## 1. Bi-Level Optimization in MAML

MAML's bi-level structure: inner loop adapts to tasks (forward pass + gradient step), outer loop optimizes for fast adaptation. Computing gradients through gradient steps (second-order derivatives) is computationally expensive, motivating first-order variants.

## 2. Overfitting in Few-Shot Learning

With only a few examples, memorization is easy and generalization is hard. Techniques: data augmentation, early stopping, regularization. Different from standard supervised learning where overfitting means optimizing training data—meta-learning overfitting means poor adaptation to new tasks.

## 3. Task Distribution Design

The meta-training task distribution ultimately limits generalization. If distribution is too narrow, learned algorithms generalize poorly to new task types. Too broad and algorithms might not specialize sufficiently to succeed on any task.

## 4. Metric Learning Geometry

Metric learning learns embeddings where similar examples are close. The geometry of learned embedding spaces determines performance. Linear separability in embedding space = easy classification; complex curved decision boundaries = harder learning.

## 5. Task-Conditioned Adaptation

Some meta-learning methods condition on task information (task embeddings). This enables more flexible adaptation than general meta-learners. Conditioning information might be learned from data, specified by users, or derived from problem structure.

