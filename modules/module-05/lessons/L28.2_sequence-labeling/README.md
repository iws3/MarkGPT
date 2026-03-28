# Sequence Labeling and Structured Prediction
## Comprehensive Learning Guide

## Sequence Models

Structured prediction outputs sequences of labels.

Sequential dependencies affect valid predictions.

First-order Markov models capture label pairs.

Higher-order models capture longer patterns.

Constraints enforce valid label combinations.

Inference algorithms find best sequence.

Decoding complexity depends on feature scope.

## CRF and Structured Learning

Conditional Random Fields model label dependencies.

Potential functions score label sequences.

Global normalization enables correct probability.

CRF training maximizes sequence likelihood.

Feature templates define potential functions.

Exact inference through dynamic programming.

Approximate inference for complex models.

## Model Combinations

RNN captures sequential context.

CRF decoding ensures label validity.

LSTM-CRF combines neural and structured approaches.

BiLSTM-CRF uses bidirectional context.

Self-attention captures non-local dependencies.

Transformer-CRF leverages modern architectures.

Multi-task learning improves generalization.

