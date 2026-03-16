# Debugging Guide for Model-Based RL

## 1. Model Predictions Diverge from Reality
**Symptoms**: Model works initially, then compounds errors

**Diagnose**:
- Collect rollouts: compare model predictions vs observed
- Measure: MSE of model predictions on test data
- Check: how far do predictions stay accurate? (horizon)

**Fix**:
- Ensemble models: average 5-10 models
- Learn uncertainty: train model to predict error bounds
- Short planning horizons: 5-20 steps max

## 2. Planning Quality Doesn't Improve with Better Model
**Symptoms**: Better model ≠ better policy

**Diagnose**:
- Separate analysis: model error vs policy reward
- Check: does plan execution succeed?

**Fix**:
- Validate: plans are actually being executed
- Measure: does model improvement help planning?
- Or: focus on policy learning instead

## 3. Real-Time Planning Too Slow
**Symptoms**: Can't plan fast enough for real-time control

**Diagnose**:
- Profile: where is computation spent?
- Measure: steps/second achieved

**Fix**:
- Use simpler models: distill to smaller network
- Cache trajectories: reuse past plans
- Or use learned policy without explicit planning

## 4. Reward Misspecified in Planning
**Symptoms**: Model predicts well but plans go wrong

**Diagnose**:
- Validate: does reward function work during rollout?
- Check: reward values seem reasonable?

**Fix**:
- Verify rewards in imagined trajectories match real
- Use learned value function instead of reward model
- Audit reward function carefully

## 5. Debugging Model-Based Agents
**Tools**:
- visualize_predictions.py: plot model outputs vs real
- trajectory_analysis.py: analyze rollout quality
- planning_horizon_analysis.py: find optimal horizon
