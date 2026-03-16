# Troubleshooting Guide for Lesson 2

## Problem: Training crashes with NaN loss

Symptoms: Loss becomes NaN after few iterations

Root Causes:
1. Exploding gradients: too high learning rate
2. Invalid computations: log of negative number
3. Numerical instability: very large intermediate values

Solutions:
- Reduce learning rate by 10x
- Add gradient clipping: clip to [-1, 1]
- Check input normalization
- Use more numerically stable ops

## Problem: Convergence is too slow

Symptoms: Training takes 100+ episodes to converge

Root Causes:
1. Learning rate too low
2. Feature representation inadequate
3. Reward signal too sparse

Solutions:
- Increase learning rate gradually
- Improve feature engineering
- Add shaped rewards or intrinsic motivation

## Problem: Performance plateaus

Symptoms: Stops improving despite training

Root Causes:
1. Local optimum reached
2. Exploration insufficient
3. Model capacity limited

Solutions:
- Increase exploration: higher epsilon
- Larger network capacity
- Different initialization

## Quick Diagnosis

Is loss NaN? -> Reduce learning rate
Is agent learning? -> Check reward signal
Is training unstable? -> Increase batch size
Does it converge? -> SUCCESS
