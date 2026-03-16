# Debugging Guide for Meta-Learning

## 1. Meta-Learning Not Generalizing
**Symptoms**: Works on training tasks, fails on test tasks

**Diagnose**:
- Measure: train task vs test task performance gap
- Check: are test tasks actually novel?

**Fix**:
- Increase training task diversity
- Use more shot examples: few-shot may need more data
- Simplify model: may be overfitting to task distribution

## 2. Adaptation Not Happening
**Symptoms**: Model's behavior doesn't change on new task

**Diagnose**:
- Measure: inner loop loss decrease (should drop)
- Check: are gradients flowing?

**Fix**:
- Increase inner loop learning rate: make bigger steps
- Increase inner loop steps: give more time to adapt
- Verify: is the model actually updating?

## 3. Gradient Explosion in Meta-Gradient
**Symptoms**: Meta-learning diverges; loss becomes NaN

**Diagnose**:
- Check gradient norms: should be < 10
- Look for: very large second derivatives

**Fix**:
- Add gradient clipping: clip to [-1, 1]
- Use first-order approximation: ignore 2nd derivatives
- Reduce meta-learning rate: use 0.001

## 4. Catastrophic Forgetting
**Symptoms**: Learning new task ruins old task performance

**Diagnose**:
- Evaluate on old tasks: should maintain performance
- Check: weight changes from new task learning

**Fix**:
- Use replay: occasionally revisit old tasks
- Or context vectors: condition on task identity
- Elastic weight consolidation: regularize important weights

## 5. Debugging Few-Shot Learning
**Tools**:
- analyze_adaptation.py: plot loss during inner loop
- visualize_features.py: t-SNE of learned representations
- benchmark_generalization.py: measure test performance
