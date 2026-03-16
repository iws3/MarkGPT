# Debugging Guide for Deep Q-Networks and Policy Gradients

## 1. Experience Replay Not Working
**Symptoms**: DQN learns slowly or unstable

**Diagnose**:
- Check replay buffer size: should be at least 10k transitions
- Verify buffer diversity: first action != latest action
- Monitor: average advantage in sampled batch

**Fix**:
- Increase buffer size: 100k+ for Atari
- Sample uniformly: not just recent transitions
- Check for bugs: is buffer actually being sampled?

## 2. Target Network Not Stabilizing Learning
**Symptoms**: Q-values diverge despite target network

**Diagnose**:
- Compare online vs target network outputs: should differ gradually
- Check update frequency: if too high, target becomes copy

**Fix**:
- Standard: update target every 10k steps
- Or soft update: θ_target = 0.99*θ_target + 0.01*θ_online

## 3. Policy Gradient Variance Too High
**Symptoms**: Policy updates are inconsistent; high variance

**Diagnose**:
- Compute gradient std: should be < 1.0
- Check returns: min/max should be reasonable

**Fix**:
- Always use baseline subtraction
- Normalize returns: (R - mean)/std
- Use GAE for smoother estimates

## 4. Entropy Regularization
**Symptoms**: Policy becomes too greedy; gets stuck

**Diagnose**:
- Measure policy entropy: should be > 0.5 for discrete
- Check: does policy visit all actions?

**Fix**:
- Add entropy bonus: -0.01 * H[π]
- Decay entropy: starts high, decreases over time

## 5. Network Architecture Issues
**Symptoms**: Policy doesn't learn despite correct algorithm

**Diagnose**:
- Test on toy problem: CartPole
- Check network output ranges

**Fix**:
- Verify dimensions: input matches observation space
- Add batch normalization: standardize activations
- Use ReLU + final layer without activation
