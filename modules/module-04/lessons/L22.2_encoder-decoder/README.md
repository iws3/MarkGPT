# Encoder-Decoder with Attention
## Comprehensive Learning Guide

## Attention Mechanism

Attention weights enable decoder to focus on relevant inputs.

Attention queries decoder state to find relevant input positions.

Alignment scores measure similarity between query and keys.

Softmax normalizes alignment scores to probability distribution.

Attention output is weighted sum of input values.

Context vector augmented with attention-weighted input.

## Implementation Details

Query is decoder hidden state at current timestep.

Keys and values derived from encoder outputs.

Dot-product attention measures query-key similarity efficiently.

