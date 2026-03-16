# Debugging Guide for Imitation Learning

## 1. Behavioral Cloning Diverges From Expert
**Symptoms**: Agent performs well initially, then fails

**Diagnose**:
- Measure distribution shift: how different is rollout vs expert?
- Plot: state visitation from agent vs expert

**Fix**:
- Use DAgger: collect corrections on agent-generated states
- Or reward augmentation: small RL bonus on top of BC
- Validate: agent now follows expert better

## 2. Insufficient Demonstration Data
**Symptoms**: BC overfits; high train loss but poor test

**Diagnose**:
- Plot train vs test loss: gap indicates overfitting
- Measure state coverage: does data cover full state space?

**Fix**:
- Collect more diverse demonstrations
- Use data augmentation: small perturbations
- Early stopping: don't train to zero loss

## 3. GAIL Discriminator Dominates
**Symptoms**: Discriminator converges but policy doesn't improve

**Diagnose**:
- Monitor discriminator accuracy: should stay ~50%
- Check: does generated behavior match expert?

**Fix**:
- Alternate training: more policy updates before discriminator
- Gradient penalty: prevent discriminator overfitting
- Spectral normalization: stabilize discriminator

## 4. IRL Reward Unrealistic
**Symptoms**: Reward function learned but doesn't make sense

**Diagnose**:
- Visualize reward: plot R(s) for key states
- Check: is reward structure human-interpretable?

**Fix**:
- Add regularization: prefer simpler rewards
- Start from hand-designed basis: then adjust
- Validate: does reward separate expert from random?

## 5. Multimodal Expert Behavior
**Symptoms**: BC averages expert actions; learns weird behavior

**Diagnose**:
- Check action distribution: is it multi-peaked?
- Visualize: group expert trajectories by strategy

**Fix**:
- Use mixture models: BC with latent variables
- Or behavior cloning + RL: learn from both
- Detect modes: cluster trajectories first
