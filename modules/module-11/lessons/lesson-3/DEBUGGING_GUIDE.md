# Debugging Guide for Multi-Agent RL

## 1. Non-Stationary Training
**Symptoms**: Policies oscillate; training unstable

**Diagnose**:
- Plot individual agent performance: should increase
- Check opponent strength: is it adapting?

**Fix**:
- Reduce learning rate: slows policy changes
- Use population-based training: diverse opponent pool
- Add replay: store old opponent policies

## 2. Credit Assignment Fails
**Symptoms**: Free-rider problem; some agents don't learn

**Diagnose**:
- Monitor individual rewards: are they proportional to contrib?
- Check: agent rewards correlate with its actions?

**Fix**:
- Switch to value decomposition: QMIX
- Or use intrinsic motivation: individual reward signal
- Validate: reward signal makes sense

## 3. Communication Collapse
**Symptoms**: Agents learn but don't actually communicate

**Diagnose**:
- Measure: do messages vary by state?
- Check: performance same with/without communication?

**Fix**:
- Bottleneck communication: limit bandwidth
- Add regularization: discourage trivial messages
- Validate: removing communication hurts performance

## 4. Scalability Breaks
**Symptoms**: Works with 2 agents, breaks with 10+

**Diagnose**:
- Measure wall-clock time per update
- Profile: where is computation spent?

**Fix**:
- Use hierarchical policies: groups of agents
- Sparse communication: agents only talk locally
- Measure again: ensure still correct

## 5. Debugging Multi-Agent Games
**Tools**:
- record_games.py: save gameplay for analysis
- analyze_communication.py: visualize message flow
- audit_rewards.py: verify fairness
