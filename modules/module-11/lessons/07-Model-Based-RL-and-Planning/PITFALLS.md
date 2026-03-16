# Common Pitfalls in Model-Based RL and Planning

## 1. Model Overconfidence on Out-of-Distribution States
**The Pitfall**: Trusting learned dynamics model on states it was never trained on.

**Why It Matters**:
- Compounding errors in planning
- Agent execute actions model predicts well but are disastrous
- Catastrophic failures in deployment

**How to Avoid**:
- Implement uncertainty quantification: epistemic + aleatoric
- Reject planning on high-uncertainty states
- Fall back to real environment or conservative policy

## 2. Model Bias Accumulation During Planning
**The Pitfall**: Small systematic biases in dynamics model compound over planning horizon.

**Why It Matters**:
- Plans diverge from true trajectories
- Agent learns to exploit model errors, not solve task
- Negative transfer

**How to Avoid**:
- Measure bias: compare model predictions vs actual
- Short planning horizons: H < 1/(1-discount)
- Use ensemble models: average out systematic biases

## 3. Environment Interaction During Planning Without Real Resets
**The Pitfall**: Assuming you can reset to any state during planning (model-based), but can't in real env.

**Why It Matters**:
- Plans are infeasible: require resetting mid-trajectory
- No actual behavioral policy for deployment
- Gap between planning and execution

**How to Avoid**:
- Enforce continuity: plans must follow from previous trajectory
- Or plan only forward: given current state, multiple futures
- Test: execute planned trajectory without resets

## 4. Insufficient Model Training Before Planning
**The Pitfall**: Using model for planning before it's reasonably accurate.

**Why It Matters**:
- Poor model → poor plans → suboptimal behavior
- Can be worse than no planning
- Wastes computation on bad plans

**How to Avoid**:
- Validate model first: MSE/MAE on holdout set
- Threshold: use planning only if model error < target
- Monitor: planning performance vs random action

## 5. Horizon Length Not Tuned to Model Accuracy
**The Pitfall**: Using fixed planning horizon regardless of model quality.

**Why It Matters**:
- Long horizon on inaccurate model: compounded errors
- Short horizon on accurate model: suboptimal plans
- Mismatch hurts performance

**How to Avoid**:
- Adaptive horizon: inversely proportional to model uncertainty
- Or schedule: increase horizon as model improves
- Experiment: measure performance vs horizon length

## 6. Ignoring Model Latency in Real-Time Planning
**The Pitfall**: Model inference is slow; can't replan fast enough.

**Why It Matters**:
- Planning computation exceeds action execution time
- Stale plans from outdated observations
- Real-time control impossible

**How to Avoid**:
- Profile: measure model inference time
- Plan while executing: generate next plan while current executing
- Use simpler models if needed

## 7. Reward Speculation Without Validation
**The Pitfall**: Assuming reward function is accurate for planning without checking.

**Why It Matters**:
- Wrong reward → wrong plans, even with perfect model
- Easy to miss: joint model + reward that's internally consistent but wrong
- Hard to debug

**How to Avoid**:
- Separately validate reward: compare to actual rewards received
- Ablation study: does better model improve planning?
- Or better reward? Or both?

## 8. Plan Execution Gap: Planning in Imagination vs Reality
**The Pitfall**: Behavior during planning must exist in real environment.

**Why It Matters**:
- Plan requires states that are impossible to reach
- Or actions that aren't actually executable
- Real world rejects the plan

**How to Avoid**:
- Constrain planning to feasible region
- Model includes action constraints
- Test: plans are realizable by actual agent

## 9. No Computational Budget Allocation Between Model and Policy Learning
**The Pitfall**: No explicit tradeoff between learning better model vs better policy.

**Why It Matters**:
- Sometimes better to learn policy directly
- Sometimes better to improve model accuracy
- Inefficient allocation of computation

**How to Avoid**:
- Measure: value of improving model vs policy
- Budget allocation: (x% learning, (100-x)% planning)
- Dynamic budget: adjust based on model accuracy trends

## 10. Stochasticity Handling in Planning
**The Pitfall**: Planning algorithm assumes deterministic dynamics when environment is stochastic.

**Why It Matters**:
- Single trajectory insufficient
- Need to handle outcome distribution
- Plans collapse to unlikely futures

**How to Avoid**:
- Use tree search: sample multiple future rollouts
- Cross-entropy method: importance sampling of futures
- Measure: plan robustness across stochasticity samples

## Summary
Model-based planning pitfalls revolve around model uncertainty, compounding errors, and the reality gap. The golden rule: always validate that planned behaviors work in the real environment.
