# Debugging Guide for Production RL

## 1. Model Performance Degrades Over Time
**Symptoms**: Policy works initially, then performance drops

**Diagnose**:
- Monitor: key metrics over last week/month
- Check: has data distribution changed?

**Fix**:
- Retraining: trigger model update on performance drop
- Monitor: trend alerts for early warning
- Rollback: keep older policies, revert if needed

## 2. Offline RL from Logged Data Fails
**Symptoms**: Offline policy performs worse than behavior policy

**Diagnose**:
- Check: is reward signal present in logs?
- Measure: coverage of logged data

**Fix**:
- Verify logging pipeline: all data captured?
- Importance sampling: weight samples by logging probability
- Conservative policy: constrain to logged behavior

## 3. Safety Violations in Production
**Symptoms**: Policy violates safety constraints

**Diagnose**:
- Log violations: when/why they occur
- Check: are constraints being enforced?

**Fix**:
- Hard constraints: don't allow invalid actions
- Or: massive penalty for violations
- Formal verification: prove safety guarantee

## 4. Multi-Objective Conflicts
**Symptoms**: Optimizing one objective hurts another

**Diagnose**:
- Monitor all objectives separately
- Check: are tradeoffs being made correctly?

**Fix**:
- Scalarize explicitly: define objective weights
- Pareto frontier: explore tradeoffs
- Stakeholder alignment: agree on weights

## 5. Production Monitoring
**Tools**:
- metrics_dashboard.py: real-time KPI tracking
- anomaly_detector.py: flag unusual behavior
- performance_audit.py: regular evaluation
- safety_monitor.py: constraint violation logging
