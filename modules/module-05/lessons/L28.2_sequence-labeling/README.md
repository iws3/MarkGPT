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

## Advanced Sequence Models

Pointer networks learn attention-based selection.

## POS Tagging

### Universal Dependencies

NOUN, VERB, ADJ, ADP, ...
Language-independent tags
100+ languages annotated
Standard benchmark
Simplifies cross-lingual

### Language-Specific Tags

Penn Treebank: English specific
47 tags (vs 17 UD)
More fine-grained
Better for downstream
Less portable

### Evaluation Metrics

Accuracy: % correct
Per-tag F1: Class-wise
Confusion matrix: Where errors
Simple task: 97%+ accuracy
Mostly solved

### Morphological Analysis

Fine-grained: POS + morphology
"books" = NOUN + Numb:Plur
Universal Features schema
Richer representation
Harder annotation

### Tagging with Context

"saw" could be NOUN or VERB
Context disambiguates
RNN/attention exploit context
Much better than dict-based
Neural models: 97%+ accuracy

## Advanced Sequence Modeling

### Transformer for Tagging

Self-attention over sequence
No recurrence needed
Parallel computation
Better long-range dependencies
State-of-the-art

### Conditional Random Fields Deep Dive

Global normalization
Consider tag transitions
P(sequence) not independent
Viterbi decoding optimal
Better than independent labels

