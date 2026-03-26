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

Different behavior during training vs. inference.

Improves gradient flow enabling faster training.

Reduces need for careful weight initialization.

## Normalization Variants

Layer normalization normalizes across features per sample.

Instance normalization per sample per feature.

Group normalization groups features for normalization.

Layer norm doesn't depend on batch size.

Each normalization suits different architectures.

Normalization critical for stable deep learning.


## Dropout Mechanisms

Co-adaptation prevention forces independent feature learning.

Thinned networks at test time approximate ensemble averaging.

Stochastic regularization improves generalization and robustness.

Dropout variants apply to different layer types.


## Batch Normalization

Internal covariate shift reduction stabilizes activations.

Whitening transformation normalizes each layer's inputs.

Scale and shift parameters restore representation capacity.

