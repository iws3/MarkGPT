# RNN Architecture and Variants
## Comprehensive Learning Guide

## Basic RNN Structure

Recurrent Neural Networks contain loops enabling memory.

Time unfolding creates deep feedforward graph.

Parameters shared across time steps enable generalization.

Hidden state updates are primarily recurrent not forward.

Output at each timestep can be read independently.

Sequence processing enables variable length inputs.

## RNN Variations

One-to-one networks process fixed inputs producing fixed outputs.

One-to-many networks generate sequences from single input.

Many-to-one networks encode sequences to single output.

Many-to-many networks transform input sequences to output sequences.

Encoder-decoder separates sequence encoding from decoding.

Attention-based variants weight input relevance.

## Training Recurrent Networks

Backpropagation Through Time extends backprop to sequences.

Truncated BPTT limits computation by cutting gradients.

Teacher forcing uses ground truth during training.

Scheduled sampling gradually reduces ground truth reliance.

Gradient clipping prevents explosion in recurrent networks.

Different learning rates per layer improves convergence.

## Advanced RNN Designs

Hierarchical RNNs process multi-level temporal structures.

Clockwork RNNs operate at multiple time scales simultaneously.

Dilated RNNs enable larger receptive fields per layer.

Parametric RNN variants adapt behavior to input characteristics.

Probabilistic RNNs model uncertainty in predictions.

Residual RNNs improve gradient flow through layers.

