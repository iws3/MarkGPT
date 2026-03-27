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

Additive attention uses learned scoring function.

Multi-head attention models different aspects simultaneously.

Attention weights reveal which inputs influenced output.

## Advantages

Access to full input sequence mitigates context bottleneck.

Direct paths enable gradient flow to early inputs.

Interpretability from attention weights.

Variable length input handling improved.

Particularly effective for long sequences.

