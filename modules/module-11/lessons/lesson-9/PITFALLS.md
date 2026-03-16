# Common Pitfalls in Transfer Learning and Domain Adaptation

## 1. Source and Target Domain Too Different
**The Pitfall**: Assuming policies learned in wildly different domains transfer.

**Why It Matters**:
- Large sim-to-real gaps: physics differs fundamentally
- Agent learns source-specific quirks
- Policy worse than learning from scratch

**How to Avoid**:
- Measure domain distance: feature distribution divergence
- Threshold: only transfer if domain distance small
- Else: use as initialization only, extensive fine-tuning needed

## 2. Catastrophic Forgetting When Fine-Tuning
**The Pitfall**: Fine-tuning on new task destroys performance on source task.

**Why It Matters**:
- Weights optimized for new task overwrite old knowledge
- Can't solve source task anymore
- Negative backward transfer

**How to Avoid**:
- Use rehearsal: replay source domain data during fine-tuning
- Or elastic weight consolidation: constrain weight changes
- Measure: performance on both source and target

## 3. Feature Reuse Without Validation
**The Pitfall**: Assuming learned representation transfers when it's domain-specific.

**Why It Matters**:
- Features optimized for source may not help target
- Slower learning or convergence to worse policy
- No actual benefit from transfer

**How to Avoid**:
- Ablation: freeze features vs fine-tune
- Measure: learning curves with/without transfer
- If no improvement: don't transfer that component

## 4. Sim-to-Real Gap Underestimation
**The Pitfall**: Training in simulation but deploying in reality without accounting for differences.

**Why It Matters**:
- Simulation has ideal physics, noise-free observations
- Real world: friction, delays, sensor noise
- Policy fails immediately

**How to Avoid**:
- Domain randomization: vary simulator parameters
- Noise injection: add realistic sensor/actuator noise
- Test in sim first: if policy is brittle in sim, will fail in reality

## 5. State Space Mismatch Between Source and Target
**The Pitfall**: Source has different observation space than target.

**Why It Matters**:
- Policy expects different input format
- Can't directly apply weights
- Requires careful adaptation

**How to Avoid**:
- Feature alignment: learn mapping between state spaces
- Or padding: if target has more observations, pad source
- Validate: policy works with target observations

## 6. Action Space Transfer Without Proper Mapping
**The Pitfall**: Actions have different semantics; policy transfer fails.

**Why It Matters**:
- Action index order differs between domains
- Or action ranges are different
- Policy selects wrong actions

**How to Avoid**:
- Explicit action mapping: define correspondence
- Test: verify mapped actions produce expected behavior
- If not feasible: action adaptation layer

## 7. Reward Function Transfer Unjustified
**The Pitfall**: Assuming reward structure is the same between domains.

**Why It Matters**:
- Rewards measure different things
- Agent optimizes for wrong objective
- Policy doesn't solve target problem

**How to Avoid**:
- Operator definition of reward in each domain
- Or derive target reward from source
- Validate: does inferred reward make sense in target?

## 8. Insufficient Target Domain Data Collection
**The Pitfall**: Transfer learning assumed to reduce data needs; then not collecting enough target data.

**Why It Matters**:
- Transfer helps but isn't free
- Still need reasonable target domain coverage
- Overfitting to small target dataset

**How to Avoid**:
- Measure: sample complexity with vs without transfer
- Rule of thumb: collect 10-20% of what needed without transfer
- Monitor: validation performance on target

## 9. Distribution Shift in Data Collection Process
**The Pitfall**: No domain shift in policy but domain shift in data collection method.

**Why It Matters**:
- Policy exploits artifacts of data collection
- Fails when collection method changes
- Brittleness disguised as generalization

**How to Avoid**:
- Log data collection metadata: how was data gathered?
- Test on different collection procedures
- Validate: performance is robust to collection method

## 10. Negative Transfer Undetected
**The Pitfall**: Transfer learning hurts performance but you don't realize it.

**Why It Matters**:
- Policy worse than learning from scratch
- Wasted computational resources
- False belief in transfer scheme

**How to Avoid**:
- Always establish baseline: learning from scratch on target
- Measure: compare transfer vs scratch learning curves
- If transfer worse: don't use it; investigate why

## Summary
Transfer learning pitfalls center on domain mismatch and feature/reward misalignment. The principle: transfer is useful only when domains are sufficiently similar and transfer mechanism is valid.
