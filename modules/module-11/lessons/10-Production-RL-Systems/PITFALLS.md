# Common Pitfalls in Production RL Systems

## 1. Reward Specification Ambiguity in Production
**The Pitfall**: Rewards seem clear in research but become intricate in deployment.

**Why It Matters**:
- Agent exploits reward loopholes
- Real-world consequences from unintended behavior
- Costly corrections

**How to Avoid**:
- Specification review: engineers + domain experts
- Adversarial examples: brainstorm reward exploits
- Progressive deployment: gradual rollout, monitor for issues

## 2. Distribution Shift in Production Data
**The Pitfall**: Training data distribution differs from production data distribution.

**Why It Matters**:
- Policy makes mistakes on production states
- Performance degradation over time
- Unexpected failures

**How to Avoid**:
- Monitor: track state distribution changes
- Retraining schedule: periodic updates with production data
- Outlier detection: flag anomalous states for manual review

## 3. Stale Model and Real-Time Constraints
**The Pitfall**: Model update frequency inadequate for production SLA.

**Why It Matters**:
- Outdated policy continues suboptimal behavior
- Can't respond quickly to environment changes
- Service level agreement violations

**How to Avoid**:
- Measure: required policy update frequency
- Implement fast retraining pipeline
- Use multi-armed bandits: rapid response to reward changes

## 4. Insufficient Rollback Capability
**The Pitfall**: New policy deployed but no way to quickly revert if it fails.

**Why It Matters**:
- Bad policy contaminates real user data
- Cannot quickly restore service
- Loss of trust

**How to Avoid**:
- Version control: save all policy checkpoints
- Automated rollback: triggers if performance drops
- Canary deployment: test on subset before full rollout

## 5. Offline RL without Proper Logging
**The Pitfall**: Training offline policy but historical logs were not recorded systematically.

**Why It Matters**:
- Offline RL needs high-quality historical data
- Missing actions or rewards breaks training
- Policy quality is unpredictable

**How to Avoid**:
- Instrumentation: log all state/action/reward/next_state
- Data validation: check for missing/corrupted entries
- Retention policy: keep sufficient historical data

## 6. Exploration Neglect in Logged Data
**The Pitfall**: Offline RL policy learns to exploit logged data distribution, not explore effectively.

**Why It Matters**:
- Policy constrained to logged behavior
- Can't improve beyond historical performance
- No discovery of better strategies

**How to Avoid**:
- Logged data should include exploratory actions
- Or use batch constrained learning: limit deviation from logged
- Estimate value of exploration: accept suboptimality for potential improvement

## 7. Off-Policy Corrections Incorrectly Applied
**The Pitfall**: Using on-policy RL methods on production data (off-policy data).

**Why It Matters**:
- Biased gradient estimates
- Convergence to wrong policy
- Poor production performance

**How to Avoid**:
- Use off-policy methods: DQN, SAC, etc.
- Or importance sampling correction
- Validate: are you actually doing off-policy learning?

## 8. Safety Constraints Become Ad-Hoc
**The Pitfall**: Safety constraints added after model training, not integrated during learning.

**Why It Matters**:
- Constraints don't guide learning
- Agent finds loopholes
- Safety is fragile

**How to Avoid**:
- Constraints during training: hard constraints on actions
- Or in reward: penalize constraint violations heavily
- Formal verification: prove policy never violates constraints

## 9. Multiobjective Tradeoffs Implicit Not Explicit
**The Pitfall**: Production RL has multiple objectives (revenue, user engagement, cost) without clear tradeoff.

**Why It Matters**:
- Algorithm optimizes first objective, ignores others
- Unbalanced outcomes
- Stakeholder dissatisfaction

**How to Avoid**:
- Explicit multi-objective formulation: weighted sum or Pareto frontier
- Stakeholder alignment: agree on objective weights
- Monitor: report all objectives regularly

## 10. Model Degradation Monitoring Absent
**The Pitfall**: Policy deployed without monitoring performance degradation.

**Why It Matters**:
- Silent failures: policy gets worse, nobody notices
- Cascading problems: small degradation leads to larger issues
- Recovery difficult after extended period

**How to Avoid**:
- Automated monitoring: key metrics tracked continuously
- Alerting: anomalies trigger human review
- Degradation response: automatic triggering of retraining

## Summary
Production RL pitfalls arise from the gap between research and deployment. Key principle: production systems require explicit handling of safety, monitoring, and distribution shift that research often takes for granted.
