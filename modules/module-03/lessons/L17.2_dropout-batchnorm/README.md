# Dropout and Batch Normalization
## Comprehensive Learning Guide

## Dropout Mechanism

Dropout randomly disables neurons during training.

Each neuron kept with probability p during forward pass.

Prevents co-adaptation of features.

Acts as ensemble of thinned networks.

No dropout applied during inference.

Scaling ensures same expected output at inference.

## Dropout Variants

Standard dropout drops neurons independently.

Spatial dropout drops feature maps in convolutional layers.

Variational dropout shares dropout mask across timesteps.

DropConnect drops weights instead of activations.

Monte Carlo dropout enables uncertainty estimation.

Variants optimize dropout for different architectures.

## Batch Normalization Details

Normalization performed per-feature across minibatch.

Learnable scale and shift parameters restore expressiveness.

Running statistics tracked for inference.

