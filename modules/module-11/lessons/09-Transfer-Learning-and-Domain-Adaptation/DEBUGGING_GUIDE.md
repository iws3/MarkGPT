# Debugging Guide for Transfer Learning

## 1. Negative Transfer: Transfer Makes Things Worse
**Symptoms**: Learning from scratch > learning with transfer

**Diagnose**:
- Compare learning curves: transfer vs from scratch
- Check: is transfer actually helping?

**Fix**:
- Measure domain similarity: sources too different
- Reduce reliance on transfer: freeze fewer layers
- Or: don't transfer, learn from scratch faster

## 2. Sim-to-Real Transfer Breaks
**Symptoms**: Works in simulation, fails in real world

**Diagnose**:
- Identify failure modes: specific scenarios?
- Check: is domain gap too large?

**Fix**:
- Domain randomization: vary simulator parameters significantly
- Add real data: retrain on real-world experiences
- Robustness testing: make policy more conservative

## 3. Catastrophic Forgetting on New Task
**Symptoms**: Learning new task ruins performance on old

**Diagnose**:
- Test on old domain: performance drops?
- Check: weight changes from new training

**Fix**:
- Rehearsal: mix old and new task data
- Elastic weight consolidation: protect important weights
- Task-specific modules: separate network parts per task

## 4. Feature Mismatch Between Domains
**Symptoms**: Source features don't work in target domain

**Diagnose**:
- Visualize source vs target features (t-SNE)
- Check: do features capture target task structure?

**Fix**:
- Learn feature mapping: bridge representation
- Adversarial domain adaptation: make features domain-invariant
- Or: don't transfer, learn fresh features

## 5. Debugging Domain Adaptation
**Tools**:
- feature_visualizer.py: t-SNE of representations
- transfer_effectiveness.py: measure improvement
- domain_similarity_analyzer.py: quantify domain gap
