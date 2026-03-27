# Vanishing and Exploding Gradients
## Comprehensive Learning Guide

## Gradient Flow in RNNs

Gradients flow backward through time unfold graph.

Chain rule multiplication creates products of Jacobians.

Repeated multiplication can cause exponential growth or decay.

Vanishing gradients prevent early timesteps from learning.

Exploding gradients cause instability and NaN values.

Proper initialization mitigates gradient flow problems.

## Vanishing Gradient Problem

Activation derivatives less than one cause decay.

Long sequences accumulate many small multiplications.

Early time steps receive negligible gradient signal.

Long-term dependencies become unlearnable.

Sigmoid activation particularly prone to vanishing.

ReLU activation partially alleviates problem.

## Solutions and Mitigations

Gradient clipping caps maximum gradient magnitude.

