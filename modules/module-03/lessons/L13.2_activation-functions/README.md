# Activation Functions for Neural Networks
## Comprehensive Learning Guide

## Linear Functions

Linear activation f(x) = x produces outputs proportional to inputs.

Linear activations fail to introduce non-linearity to networks.

Stacking linear layers is mathematically equivalent to a single layer.

Linear functions limit the network to learning linear relationships.

Deep networks with only linear activation reduce to shallow networks.

This motivates using non-linear activation functions.

## Sigmoid and Tanh Functions

The sigmoid function squashes values to range (0, 1).

Sigmoid was historically popular but suffers from vanishing gradients.

The tanh function maps values to range (-1, 1).

Tanh is zero-centered improving optimization over sigmoid.

Both suffer from gradient saturation at extreme values.

Modern networks prefer ReLU-based activations.

## ReLU and Variants

ReLU (Rectified Linear Unit) is f(x) = max(0, x).

ReLU enables efficient computation with no exponential calculations.

ReLU helps avoid vanishing gradient problems in deep networks.

Leaky ReLU allows small negative gradients to prevent dead neurons.

ELU (Exponential Linear Unit) provides smooth negative values.

