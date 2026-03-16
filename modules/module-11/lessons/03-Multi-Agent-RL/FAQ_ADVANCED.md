# Advanced FAQ for Lesson 3

## Q1: How do I know if my implementation is correct?

You can validate your implementation by:
1. Testing on toy problems first (CartPole, MountainCar)
2. Comparing against published benchmarks
3. Running multiple random seeds and checking variance
4. Visualizing learned policies
5. Checking that performance improves monotonically

## Q2: What hyperparameters should I tune first?

Priority order for tuning:
1. Learning rate (biggest impact, try 1e-4, 1e-3, 1e-2)
2. Batch size (32, 64, 128, 256)
3. Network architecture (hidden layer sizes)
4. Discount factor gamma (0.99, 0.999)
5. Exploration rate (epsilon decay schedule)

## Q3: How long should training take?

For benchmark environments:
- CartPole: <1 hour
- Atari: 1-7 days (depending on algorithm)
- MuJoCo: Hours to days

If significantly slower, check:
- GPU utilization
- Batch size optimization
- Replay buffer efficiency

## Q4: Should I use distributed training?

Use distributed training if:
- Single GPU training takes >48 hours
- You need results quickly
- You have compute resources available

Don't use if:
- Single GPU training finishes in <24 hours
- Complex distributed setup required
- Marginal speedup not worth complexity

## Q5: How do I compare algorithms fairly?

Ensure fair comparison by:
1. Same hyperparameter tuning budget
2. Same computational resources
3. Same evaluation procedure
4. Multiple random seeds
5. Statistical significance tests

## Q6: What are common failure modes?

Common failures:
1. NaN/Inf gradients - check learning rate
2. Not converging - check reward signal
3. High variance - check batch size
4. Slow training - check network size
5. Memory issues - reduce batch size

## Q7: How do I debug if training stalls?

Debug stalled training:
1. Check if states are changing
2. Verify rewards are being received
3. Monitor network gradient statistics
4. Log intermediate values
5. Visualize policy behavior

## Q8: Should I normalize data differently?

Data normalization is critical:
1. Observations: zero mean, unit variance
2. Rewards: scale to [-1, 1] or [0, 1]
3. Advantages: for policy gradients
4. Inputs to layers: batch normalization

## Q9: How do I handle continuous vs discrete actions?

Key differences:
- Discrete: use softmax policy, cross-entropy loss
- Continuous: use Gaussian policy, MSE loss
- Mixed: separate heads per action type

## Q10: Can I combine multiple algorithms?

Yes, considered advanced techniques:
1. Actor-Critic: policy + value
2. Hierarchical: options + primitives
3. Multi-task: shared features + task-specific
4. Ensemble: multiple models, voting
