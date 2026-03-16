# Common Pitfalls in Meta-Learning and Adaptation

## 1. Task Distribution Mismatch
**The Pitfall**: Meta-training and meta-testing tasks come from different distributions.

**Why It Matters**:
- Meta-learning optimizes for training distribution
- Poor generalization to new tasks
- Can't evaluate fairness across task types

**How to Avoid**:
- Explicitly define task family: "all variations of CartPole with friction"
- Check: are test tasks from same generative process?
- Use data splits: clearly separate train/test task distributions

## 2. Insufficient Adaptation Steps During Meta-Test
**The Pitfall**: One gradient step not enough to adapt to new task; also using too many.

**Why It Matters**:
- Insufficient adaptation: can't use new task information
- Too many steps: overfits to specific test instances
- Tradeoff is task-dependent

**How to Avoid**:
- Grid search: k ∈ {1, 3, 5, 10} adaptation steps
- Monitor: validation loss should decrease then plateau
- Use early stopping on validation set

## 3. Inner Loop Learning Rate Tuning Neglect
**The Pitfall**: Using same learning rate for both meta and inner loop.

**Why It Matters**:
- Inner loop needs different rate: faster for quick adaptation
- Outer loop needs different rate: slower for stable meta-gradient
- Poor performance if misaligned

**How to Avoid**:
- Inner loop: higher LR (0.01-0.1)
- Outer loop: lower LR (0.0001-0.001)
- Learn LR: use meta-optimizer for per-layer rates

## 4. Forgetting Problem: Catastrophic Task Interference
**The Pitfall**: Learning new task causes degradation on previously learned tasks.

**Why It Matters**:
- Sequential adaptation breaks meta-knowledge
- Network weights overwritten by new task
- Continual learning becomes impossible

**How to Avoid**:
- Use context variables to distinguish tasks
- Implement replay: revisit old tasks during training
- Adapter networks: keep base model fixed, adapt small modules

## 5. Evaluation Metric Confusion for Meta-Learning
**The Pitfall**: Measuring only within-task accuracy, not between-task generalization.

**Why It Matters**:
- High accuracy on training tasks ≠ good generalization
- Meta-learning success is generalization
- Easy to fool yourself with biased metrics

**How to Avoid**:
- Primary metric: average performance on held-out tasks
- Secondary: performance on in-distribution vs out-of-distribution
- Report both

## 6. Insufficient Task Diversity During Meta-Training
**The Pitfall**: Only using 2-3 task variants; meta-learning needs wide distribution.

**Why It Matters**:
- Meta-learner overfits to specific task structures
- Poor transfer to novel tasks
- Apparent success on similar tasks

**How to Avoid**:
- Generate 100+ unique training tasks
- Vary: reward functions, dynamics, state dimensions
- Measure: performance on unseen task characteristics

## 7. Hyperparameter Meta-Learning Instability
**The Pitfall**: Learning hyperparameters through meta-gradient is unstable.

**Why It Matters**:
- Second-order gradients can explode
- Hyperparameters oscillate wildly
- Training becomes non-convergent

**How to Avoid**:
- Add gradient clipping to meta-gradient updates
- Use small initial hyperparameter ranges
- Monitor: plot hyperparameter trajectories over time

## 8. Mode Collapse in Task Adaptation
**The Pitfall**: Meta-learner learns single "universal" adaptation strategy.

**Why It Matters**:
- Only works for task types resembling training distribution
- Can't handle task diversity
- False confidence in generalization

**How to Avoid**:
- Implement task discrimination: "What kind of task am I?"
- Use mixture of experts: different experts for different tasks
- Measure: performance variance across tasks should be low but nonzero

## 9. Intra-Task vs Inter-Task Performance Imbalance
**The Pitfall**: Good performance on few tasks, poor on others; not balanced.

**Why It Matters**:
- Indicates overfitting to specific task structure
- Unfair evaluation across task distribution
- Poor generalization

**How to Avoid**:
- Report per-task performance distribution
- Use weighted losses: higher weight for harder tasks
- Stratified sampling: equal representation of task types

## 10. Not Validating Few-Shot Performance Assumption
**The Pitfall**: Claiming few-shot learning without actually doing adaptation on few examples.

**Why It Matters**:
- Meta-learning premise is fast adaptation from few examples
- If using 1000 examples, not really few-shot
- Misleading performance claims

**How to Avoid**:
- Operational definition: "Few-shot means ≤ 5 examples"
- Report explicitly: exactly how many examples used
- Compare against supervised baseline with same examples

## Summary
Meta-learning pitfalls center on task distribution generalization and adaptation-test time tradeoffs. The core question: "Is my meta-learner actually adapting quickly, or just memorizing task distribution?"
