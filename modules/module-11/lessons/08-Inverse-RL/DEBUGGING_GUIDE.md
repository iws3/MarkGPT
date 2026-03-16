# Debugging Guide for Inverse RL

## 1. Inferred Reward Doesn't Match Expert Behavior
**Symptoms**: IRL converges but learned policy doesn't match expert

**Diagnose**:
- Compare rewards: expert trajectory vs learned policy trajectory
- Check: is inferred reward increasing along expert path?

**Fix**:
- Verify: expert is actually optimal under inferred reward
- Maybe: expert not optimal, use apprenticeship learning
- Debug: intermediate reward functions during IRL training

## 2. IRL Converges to Degenerate Solutions
**Symptoms**: Converges but reward is meaningless (all zeros, etc)

**Diagnose**:
- Visualize reward function: plot R(s) values
- Check: do rewards distinguish expert actions?

**Fix**:
- Add regularization: penalize large weights
- Use margin constraints: expert reward should be high
- Validate: reward separates expert from random

## 3. Insufficient Training Data for IRL
**Symptoms**: Multiple different rewards explain demonstrations

**Diagnose**:
- Generate policies from inferred rewards
- Check: do they match expert behavior?

**Fix**:
- Collect more diverse demonstrations
- Add human-defined reward basis: start from known features
- Use maximum margin: make expert reward higher

## 4. Scaling to High-Dimensional Observations
**Symptoms**: IRL infeasible on images; too slow

**Diagnose**:
- Measure: IRL solver runtime
- Check: does feature learning converge?

**Fix**:
- Learn features first: autoencoder or self-supervised
- Then IRL on learned features
- Or use deep IRL: end-to-end neural reward

## 5. Validation of Inferred Rewards
**Tools**:
- compare_policies.py: inferred vs expert behavior
- reward_visualization.py: heatmaps of reward function
- generalization_test.py: test on new expert demos
