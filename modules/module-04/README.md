# Module 04 — Recurrent Networks & Sequence Modeling
## Days 19–24 | Intermediate

---

## Module Overview

Sequences are everywhere in language: a sentence is a sequence of words, a word is a sequence of characters. This module teaches how to build networks that remember, using RNNs, LSTMs, and the first forms of attention.

By the end of Module 04, you will:
- Understand how RNNs maintain hidden state over time
- Implement the vanishing gradient problem and see it firsthand
- Build an LSTM that actually remembers long sequences
- Use attention to look back selectively through a sequence

## Learning Objectives

- Understand core ML concepts
- Implement algorithms from scratch
- Relate theory to MarkGPT architecture
- Complete hands-on exercises

## Structure

```
lessons/       - Conceptual explanations with code examples
exercises/     - Practical implementation exercises
projects/      - Larger projects (optional)
resources/     - Additional readings and links
```

## Time Estimate

- Lessons: 4-6 hours
- Exercises: 4-6 hours
- **Total: 8-12 hours per module**

## Key Concepts

[See lesson files for detailed content]

## Completion Checklist

- [ ] Read all lessons (L*_*.md files)
- [ ] Complete all exercises (day*_*.md files)
- [ ] Pass the module quiz (if provided)
- [ ] Understand connections to MarkGPT

## Resources

- Lesson references contain links to papers and tutorials
- http://markgpt-docs.com (forthcoming)
- GitHub discussions: https://github.com/yourusername/MarkGPT-LLM-Curriculum/discussions

## Next Module

See ../module-0$((i+1))/README.md for the next module.
## Recurrent Neural Networks - Fundamentals

### What is a Recurrent Network?

Standard feedforward: x → h → output
Recurrent: x_t → h_t → output_t
h_t depends on x_t AND h_{t-1}
Hidden state carries information from past.

Processing sequences: One element at a time
Weights shared across time steps (parameter efficiency).
### Hidden State and Time Unrolling

h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
o_t = W_o @ h_t + b_o

Time unrolling: Unfold RNN for T time steps
Creates deep feedforward network (depth = sequence length)
Backprop through time (BPTT): Chain rule across time.

## Sequence to Output

### Many-to-One (e.g., sentiment)

Input: Sequence of 100 words
Process: Apply RNN at each step
Output: Only use final h_T for classification
Loss computes only on last output.
Gradient flows backward through all steps.

### One-to-Many (e.g., image captioning)

Input: Single image
Process: Encode to h_0
Output: Generate caption word by word
h_0 from CNN → fed to RNN
Each RNN step outputs word token.

### Many-to-Many (e.g., NER, machine translation)

Input: Sequence of N tokens
Process: RNN processes all
Output: Sequence of N predictions
Each time step has input and output.
Examples: Sequence labeling, translation.

## The Vanishing Gradient Problem

### The Issue

BPTT: Chain rule multiplies gradients
∂L/∂h_0 = (∂L/∂h_T) * (∂h_T/∂h_{T-1}) * ... * (∂h_1/∂h_0)
Each ∂h_t/∂h_{t-1} < 1 typically
Product of T < 1 terms → exponentially small
Gradient for h_0 becomes nearly 0.

### Why This Matters

Early inputs get negligible gradients.
Model forgets distant past (effective window ~5-20 steps).
Long-range dependencies can't be learned.
Example: Pronoun in position 1, reference at position 50.
RNN unlikely to learn this dependency.

### Exploding Gradients (Opposite Problem)

If ∂h_t/∂h_{t-1} > 1:
Product of T > 1 terms → exponentially large
Gradients overflow to NaN/Inf
Training becomes unstable.
Less common than vanishing but worse when it happens.

## Solutions: Gradient Clipping

### The Fix

Gradient norm clipping:
if ||∇|| > threshold:
  ∇ = ∇ * (threshold / ||∇||)
Rescales large gradients.
Prevents explosion.
Threshold: 1.0 or 5.0 typical.

### Implementation

Compute gradients as usual.
Compute L2 norm: sqrt(sum of g^2).
If norm > max_norm: rescale.
Apply update.
Handles both explosion and (partially) vanishing.

## Solutions: Better Activation Functions

### ReLU in RNNs

tanh: Saturates, derivative → 0
ReLU: Linear on positive side, derivative = 1
Helps gradients flow better.
But can have dying ReLU problem.
ELU/GELU: Smooth, no saturation.

## Simple RNN Implementation

### Core Loop

```python
class SimpleRNN:
  def forward(self, X):  # X: (T, batch, input_dim)
    h = zeros((batch, hidden_dim))
    outputs = []
    for t in range(T):
      h = tanh(X[t] @ Wx + h @ Wh + bh)
      out = h @ Wo + bo
      outputs.append(out)
    return stack(outputs)
```

### Backpropagation Through Time

```python
def backward(self, grad_output):  # (T, batch, out_dim)
  dWx, dWh, dWo = 0, 0, 0
  dh_next = 0
  for t in reversed(range(T)):
    dh = (grad_output[t] @ Wo.T + dh_next)
    dWo += h[t].T @ grad_output[t]
    dh = dh * (1 - h[t]**2)  # tanh derivative
    dWx += X[t].T @ dh
    dWh += h[t-1].T @ dh
    dh_next = dh @ Wh.T
```

## Truncated Backpropagation

### Motivation

Full BPTT through entire sequence → slow
Backprop only through last k steps
Practical compromise: Efficient + reasonably good
k values: 20-50 steps typical
Still captures local temporal dependencies.

## Weight Initialization

### Why It Matters

Poor init: Gradients vanish/explode from start
Good init: Preserve signal variance across layers
Key: Keep ||h_t|| roughly constant
Var(h_t) ≈ Var(h_{t-1})

### Orthogonal Initialization

Initialize Wh as orthogonal matrix
Properties: Preserves vector norm
Eigenvalues = 1 (no growth/decay)
Prevents gradient explosion/vanishing initially.
Recommended for RNNs.

## Bidirectional RNNs

### Motivation

Forward RNN: Process left to right
Backward RNN: Process right to left
Concatenate outputs: [h_fwd; h_bwd]
Access context from both directions.
Improves performance on tagging tasks.

### Architecture

Input sequence: [w1, w2, w3, w4]
Forward pass: → → → →
Backward pass: ← ← ← ←
Output at t: [fwd_h_t; bwd_h_t]
Dimension: 2 * hidden_dim

## Peephole Connections

### With RNN

Standard: h_t = f(W @ [x_t; h_{t-1}])
No dependency on cell state (in basic RNN).
Gradient flow during forward pass constrained.

## Sequence Padding and Masking

### Variable Length Sequences

Real sequences: Different lengths
Batch processing: Need same length
Solution: Pad short sequences
Padding token: 0 (special index)
Sequence lengths: Store actual lengths

### Masking

During forward: Process padded positions
During loss: Ignore padded positions
Loss = sum(loss[i] * mask[i]) / sum(mask)
Prevents gradients from padding tokens.
Attention:  Mask with -inf (softmax → 0).

## Common RNN Patterns

### Encoder-Decoder (No Attention)

Encoder: Process input → final h_T
h_T: Summary of entire input
Decoder: Initialize with h_T, generate output
Limitation: All info in single vector
Better approach: Use attention (module-05).

### Autoregressive Generation

At test time: Generate one token at a time
Use own output as next input
Temperature: Control randomness
Sampling vs beam search tradeoffs
Exposure bias: Train vs test mismatch.

## Practical Considerations

### Sequence Length

Very long sequences: Truncated BPTT
Typical: 50-512 tokens
Maximum: GPU memory constraint
Tradeoff: Longer = more context, slower training

### Batch Size

Standard: 32-128
Affects gradient estimate quality
Memory per sequence * batch_size
Typical GPU: batch_size=64 for seq_len=512
Larger batch = noisier gradients

## Long Short-Term Memory (LSTM)

### The Cell State Innovation

Key insight: Separate cell state from hidden state
c_t: Cell state (internal memory)
h_t: Hidden state (external output)
Cell state acts like "conveyor belt"
Gradient can flow without vanishing.

### Gates: Forget, Input, Output

Three gating mechanisms:
1. Forget gate: f_t = sigmoid(W_f @ [h_{t-1}; x_t] + b_f)
   Controls what to discard from c_{t-1}
2. Input gate: i_t = sigmoid(W_i @ [h_{t-1}; x_t] + b_i)
   Controls what new info to add
3. Output gate: o_t = sigmoid(W_o @ [h_{t-1}; x_t] + b_o)
   Controls what to expose from c_t

### Cell State Update

Candidate cell state:
c̃_t = tanh(W_c @ [h_{t-1}; x_t] + b_c)

Cell state update:
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
⊙ denotes element-wise multiplication

Hidden state:
h_t = o_t ⊙ tanh(c_t)

### Gradient Flow Through Cell State

∂c_t / ∂c_{t-1} = f_t (Hadamard product)
f_t values in (0, 1) but not multiplication of many terms
Much better gradient flow than standard RNN
Allows gradients to propagate 100+ steps
Solves vanishing gradient problem!

### LSTM Advantages

Long-range dependencies: Can learn 100-200 step dependencies
Forget gate: Can selectively discard info
Input gate: Can control what to remember
Output gate: Can control what to reveal
Trade-off: 4x parameters vs standard RNN

## GRU: A Simpler Alternative

### Motivation

LSTM: 4 gates, complex, many parameters
Can we simplify?
GRU: 2 gates, simpler, 3x fewer parameters
Similar performance on most tasks

### GRU Gates

Reset gate: r_t = sigmoid(W_r @ [h_{t-1}; x_t] + b_r)
Controls how much of h_{t-1} to use

Update gate: z_t = sigmoid(W_z @ [h_{t-1}; x_t] + b_z)
Controls how much of new info vs old

Candidate state:
h̃_t = tanh(W @ [r_t ⊙ h_{t-1}; x_t] + b)

Final state:
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

### LSTM vs GRU

LSTM: Better with complex patterns
GRU: Faster, fewer parameters
Empirically: Often similar performance
Use GRU: When compute is limited
Use LSTM: When data is abundant
Modern trend: Transformers replace both

## Stacked RNNs

### Multiple Layers

1D: Single RNN layer
2D: Stack 2 RNN layers
Output of layer 1 → input of layer 2

Deep encoders: 2-4 layers beneficial
Each layer computes higher-level features
Example: Word embeddings → syntax → semantics

Parameters: L * layers
Training time: ~L * slower

### Residual Connections in RNNs

Very deep RNNs: Training becomes hard
Add skip connections: x_{l+2} = f(x_{l+1}) + x_l
Enables training 4+ layer RNNs
Helps gradient flow
Used in cutting-edge models.

## Attention in RNNs

### Problem: Bottleneck

Encoder outputs h_T (single vector)
Must contain all input information
Problematic for long sequences (100+ tokens)
Solution: Use all h_1, h_2, ..., h_T

