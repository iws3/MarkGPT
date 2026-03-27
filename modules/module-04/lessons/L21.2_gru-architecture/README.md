# GRU Architecture and Simplifications
## Comprehensive Learning Guide

## GRU Design

Gated Recurrent Units simplify LSTM with fewer gates.

Single hidden state replaces separate cell and hidden states.

Reset gate determines relevance of previous hidden state.

Update gate controls how much hidden state changes.

Simpler architecture reduces parameters and computation.

Smaller memory footprint enables larger batches.

## GRU vs LSTM

GRU has two gates versus LSTM's three gates.

GRU lacks separate cell state simplifying updates.

LSTM explicitly maintains memory enables stronger control.

