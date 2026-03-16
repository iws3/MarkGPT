# Debugging Guide for Hierarchical RL

## 1. Skills Not Learning
**Symptoms**: Skill agents don't improve despite training

**Diagnose**:
- Check reward signal: are skills getting positive feedback?
- Measure: skill loss should decrease over time

**Fix**:
- Verify goal reachability: some goals may be impossible
- Use curriculum: easy goals first, gradually increase
- Add intrinsic motivation: bonus for reaching any goal

## 2. Hierarchical Decomposition Fails
**Symptoms**: High-level planner selects unreachable sub-goals

**Diagnose**:
- Record: which sub-goals are attempted?
- Check: how often does skill success rate drop to 0%?

**Fix**:
- Ensure skills cover state space fully
- Learn option termination: when is skill truly "done"?
- Use safety constraints: only reachable sub-goals

## 3. Option Termination Misbehaves
**Symptoms**: Options terminate too early or too late

**Diagnose**:
- Measure: average option duration vs task horizon
- Check: is termination learned or fixed?

**Fix**:
- Learn termination: β(s) = P(terminate | s)
- Incentivize: higher reward if option terminates correctly
- Validate: duration distribution sensible

## 4. Upper-Lower Level Credit Assignment
**Symptoms**: Hard to debug which level caused failure

**Diagnose**:
- Log: which option was selected, did it succeed?
- Break down rewards per level

**Fix**:
- Separate advantage functions for each level
- Monitor: intrinsic vs extrinsic reward contribution
- Debug each level independently first

## 5. Scalability Issues
**Tools**:
- measure_complexity.py: plot steps vs hierarchy depth
- profile_hierarchical.py: wall-clock time analysis
- compare_flat_vs_hierarchical.py: performance comparison
