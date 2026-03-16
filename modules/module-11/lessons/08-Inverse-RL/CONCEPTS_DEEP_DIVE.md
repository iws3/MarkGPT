# Lesson 8: Key Concepts Deep Dive

## 1. Reward Unidentifiability

Even with perfect expert data, multiple reward functions perfectly explain the behavior. Without additional assumptions, we cannot uniquely determine THE reward. Maximum entropy IRL resolves this by choosing the simplest (highest entropy) consistent reward.

## 2. Feature Representation in IRL

IRL learns reward as weighted combination of features φ(s). Features must be chosen (hand-designed) or learned. Using wrong features makes learned rewards uninformative. Feature selection is often more important than algorithm choice.

## 3. Large Demonstrations Versus Small

Large expert datasets improve IRL but become computationally expensive. Small datasets risk overfitting to specific expert idiosyncrasies. Medium-sized diverse datasets often work best—enough data to learn general principles, small enough for efficient computation.

## 4. Inter-Task Consistency

When learning from human behavior on multiple tasks, should each task have its own reward or should There be shared reward structure? Shared rewards encourage generalization; separate rewards allow task specialization.

## 5. Preference Learning Efficiency

Learning from pairwise comparisons rather than full demonstrations can be 10x more efficient. Humans naturally compare alternatives; this aligns better with human cognition. Comparison-based IRL is emerging as practical preference learning paradigm.

