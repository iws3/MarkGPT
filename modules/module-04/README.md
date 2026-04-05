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

### Attention Mechanism

Query: Decoder state s_t
Keys: Encoder states h_1, ..., h_T
Values: Encoder states h_1, ..., h_T

Score: e_t,j = v^T @ tanh(W_s @ [s_t; h_j])
Weights: α_t,j = softmax(e_t,j)
Context: c_t = Σ α_t,j @ h_j

Output: decoder processes [s_t; c_t]

### Multiplicative Attention

Simpler form (used in transformers):
Score: e_t,j = (s_t @ h_j) / sqrt(d)
No learned parameters in scoring
Just dot product + softmax
Scale by 1/sqrt(d) for stability
Very efficient!

## Bidirectional LSTM

### Design

Forward LSTM: Process left to right
Backward LSTM: Process right to left
Outputs: [fwd_h_t; bwd_h_t]
Cannot be used for generation (needs input sequence end)
Great for tagging, classification

## PyTorch/TensorFlow LSTM Usage

### PyTorch

```python
lstm = nn.LSTM(input_size, hidden_size, num_layers,
               batch_first=True, bidirectional=True)
outputs, (h_n, c_n) = lstm(x)  # x: (batch, T, input_size)
# outputs: (batch, T, 2*hidden if bidirectional)
# h_n: (num_layers*2, batch, hidden) if bidirectional
# c_n: (num_layers*2, batch, hidden)
```

### TensorFlow

```python
lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
outputs = lstm(x)  # x: (batch, T, input_size)
# outputs: (batch, T, hidden_size)
# For last output: outputs[:, -1, :]
# Bidirectional: Bidirectional(LSTM(...))
```

## Encoder-Decoder with Attention

### Architecture

Encoder: Bi-LSTM reads input sequence
Outputs: h_1, h_2, ..., h_T
Decoder: LSTM generates output
Each step: Attends to encoder outputs
Completely parallelizable (replaced by Transformers)

### Context Vector

Each decoder step:
1. Query from decoder state
2. Compute attention weights over all encoder outputs
3. Weighted sum of encoder outputs
4. Concatenate with decoder input for next step
Very powerful (translation baseline ~30 BLEU).

## Common Practices

### Dropout

Apply to:
- Input x_t
- Between LSTM layers
- NOT between time steps (breaks temporal coherence)
Typical rate: 0.3-0.5
Prevents overfitting on small datasets

### Learning Rate

RNNs very sensitive to learning rate
Start with 1e-3
If diverges: Lower to 1e-4
Use gradient clipping (max_norm=5.0)
Warmup beneficial: Linear increase first 5% steps

## Practical Tips

- Start with 2 layers, expand if needed
- Hidden size: 128-512 typical
- Sequence length: 32-256 for NLP
- Longer = more memory, smaller batches
- Check gradient flow: norms should be O(0.1-1.0)
- Monitor validation loss during training
- Save checkpoint with best validation

## Attention Variants

### Additive (Bahdanau) Attention

Score: e = v^T @ tanh(W @ concat([query, key]))
Learnable combining function
More expressive than dot-product
Higher computational cost
Earlier method (2014), still effective

### Scaled Dot-Product Attention

Score: e = query @ key / sqrt(d_k)
No learned parameters
O(n^2) complexity (acceptable for n<512)
Scaling prevents saturation in softmax
Foundation of Transformer architecture
Preferred in modern systems

### Multi-Head Attention

Concatenate multiple attention heads
Head i: Attend to different feature subsets
Benefits: Attend to different positions, semantic meanings
Typical: 8-16 heads
Total params same as single head
h(Q,K,V) = concat(head_1, ..., head_h) @ W^O

### Sparse Attention

Full attention: O(n^2) memory
Sparse: Attend only to local window
Block-sparse: Attend to blocks
Enables longer sequences
Trade-off: Some long-range dependencies lost

### Self-Attention

Query, Key, Value all from same sequence
Each position attends to all positions
Captures positional relationships
Foundation for Transformers
Used in BERT, GPT, etc.

## Transformers: The Game Changer

### Motivation

RNN/LSTM: Sequential, hard to parallelize
Transformers: Fully parallelizable
2017: "Attention is All You Need" paper
Became foundation of modern NLP
100x more efficient training

### Encoder Architecture

1. Multi-head self-attention
2. Feed-forward network
3. Layer normalization
4. Residual connections
Repeat 12-96 times (layers)

Each layer increases semantic understanding
Lower: Syntax, Upper: Semantics

### Postional Encoding

Problem: Attention doesn't know position
Solution: Add positional pattern
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
Encodes both absolute and relative position
Allows extrapolation to longer sequences

### Feed-Forward Network

2-layer MLP between attention
Expands: d_model → d_ff → d_model
d_ff = 4 * d_model typical
Applied position-wise (all tokens independently)
ReLU activation
Adds non-linearity, capacity

### Layer Normalization

Before 2020: Pre-LN (norm before sublayer)
After 2020: Post-LN (norm after)
Pre-LN: Easier to train deep networks
Post-LN: Slightly better final performance
Modern: Usually Pre-LN
Normalizes each sample independently

### Decoder with Cross-Attention

Self-attention: On target sequence
Cross-attention: On encoder outputs
Causality: Can't attend to future tokens
Masked attention: Mask future positions to -inf
Enables generation one token at a time

## Sequence-to-Sequence with Transformers

### Training

Encoder: Process full source
Decoder: Process full target (during training)
Loss: Cross-entropy on target tokens
Efficiency: Fully parallel, train in hours not days
MarkGPT: Uses this architecture

### Inference

Encoder: Process source (once)
Decoder: Generate token by token
Feed own output as next input
Stop: Until EOS token
Beam search: Track multiple hypotheses
Temperature: Control randomness

## Attention Analysis

### Visualization

Heatmap: Attention weights
X-axis: Keys (what to attend to)
Y-axis: Queries (who's attending)
Color: Weight magnitude
Reveals learned patterns
Early layers: Local patterns
Late layers: Long-range dependencies

### Interpretability

What does attention attend to?
Head 1: Syntactic relationships
Head 2: Semantic relationships
Head 3: Position information
Useful but not ground truth
Single value doesn't explain decision

## Language Modeling

### Task Definition

Predict next token given context
P(w_t | w_1, ..., w_{t-1})
Unsupervised pretraining
Foundation for fine-tuning
GPT trained this way

### Perplexity

Measures model uncertainty
PP = 2^(cross-entropy)
Lower is better
PP=100: Like choosing from ~100 equally likely tokens
MarkGPT: Achieves 8-15 on different domains
Baseline: 50-100 typical

## Machine Translation: Case Study

### Dataset

WMT14 English-German
4.5M sentence pairs
English: 80K vocab, German: 80K vocab
Average length: 25 tokens
Test: 3000 sentences

### Model Architecture

Encoder-Decoder Transformer
6 layers each
512 hidden dimension
8 attention heads
2048 FFN dimension
Total: 65M parameters

### Training

Batch size: 4096 tokens
Learning rate: 0.0001 (with warmup)
Optimizer: Adam (β1=0.9, β2=0.98)
Training: 5 days on 8 GPUs
Dropout: 0.1
Label smoothing: 0.1

### Results

BLEU: 28.4 (very competitive)
Inference speed: 100 tokens/sec GPU
Inference speed: 2 tokens/sec CPU
vs phrase-based SMT: 23 BLEU
5 point improvement!
Transformers surpassed SMT in 2017

## Question Answering: Case Study

### SQuAD Dataset

100K questions on Wikipedia passages
Answer: Span in passage
Train: 80K, Test: 10K
Avg passage: 150 tokens, Avg question: 10 tokens
Avg answer span: 3 tokens

### Model Architecture

BERT: 12 layers, 768 hidden
Encoder only (no decoder)
Add task-specific layers:
- Start span prediction
- End span prediction
Span = argmax(start) to argmax(end)

### Results

EM (Exact Match): 85.1%
F1 Score: 91.8%
vs human performance: 91.2% F1
Model actually beats humans slightly!
SQuAD v2: Adds unanswerable questions
Model performance: 83%

## Sentiment Classification: Case Study

### Dataset Setup

40K movie reviews (binary)
Positive: 20K, Negative: 20K
Average length: 250 tokens
Train: 25K, Test: 5K, Val: 10K
Very imbalanced words (stop words common)

### Fine-tuning Approach

Start with pre-trained BERT
Remove language modeling head
Add classification head: [CLS] → dense → 2 classes
Fine-tune 2-5 epochs
Learning rate: 2e-5 (small!)
Batch size: 32

### Results

Accuracy: 91.3% (vs LSTM: 87%)
Training: 2 hours (vs LSTM: 6 hours)
Inference: 500 samples/sec GPU
Transfer learning wins!
Pre-training on 3B tokens helps

## Named Entity Recognition: Case Study

### Task Definition

Tag each token with entity type
Categories: PERSON, LOCATION, ORG, O (other)
Sequence labeling task
Dataset: 15K training sentences
Average: 15 tokens per sentence

### Model Architecture

BERT encoder: 12 layers
Token classification head:
Output for each token → 4 classes
CRF layer (optional): Enforce valid tag sequences
Total params: 110M

### Results

F1 score: 92.4% (strong)
vs baseline BiLSTM-CRF: 90.2%
vs rule-based NLP tools: 87%
Transformers great for structured output
Context modeling crucial

## Common Training Tips

### Batch Size

Larger batch: Better gradient estimate
Smaller batch: Faster feedback loop
Typical: 16-64 for classification
Typical: 32-256 for LM
Memory: Scales with batch_size * seq_len

### Learning Rate Schedules

Linear warmup: 0 → LR over 10% steps
Then constant: Stay at LR
Or decay: cos(...) decreasing
Warmup prevents divergence
Decay helps convergence
Half life: 50K steps typical

### Regularization

Dropout: 0.1 in attention, FFN
Weight decay: 0.01 typical
Label smoothing: 0.1 (soften targets)
Data augmentation: Back-translation
Early stopping: Monitor validation

### Checkpointing

Save model every N steps
Keep best (by validation metric)
Enable recovery from failure
A/B testing different configs
Typical: Save every 1000 steps
Keep 3-5 recent checkpoints

## Inference Optimization

### Batching

Process multiple samples simultaneously
GPU utilization: 5% → 90%
Batch size: Trade-off latency vs throughput
Typical: Batch 32-64 for low latency
Batch 256+ for throughput

### KV-Cache

During generation, recompute all steps
KV-cache: Store K, V from previous
Compute only for latest token
Trade-off: Memory for compute
7B model: ~30GB cache for batch=32
Worth it: 10x speedup

### Quantization

FP32 → INT8 or INT4
Model size: 4x smaller
Inference: 2-3x faster
Accuracy: 1-5% loss (varies)
INT4: More aggressive, ~10% loss
Worth exploring for deployment

## Multi-GPU Training

### Data Parallelism

Same model on N GPUs
Each GPU: Different batch
Sync gradients after backward
Linear speedup (mostly)
Easy to implement

### Distributed Training

Multiple nodes, multiple GPUs
NCCL: Efficient GPU communication
NVLink: Fast inter-GPU bandwidth
Scaling: 8 GPUs → 8x speedup (95%)
Scaling: 64 GPUs → 50x speedup (78%)
Communication overhead increases

## Debugging Guide

### Check Data Loading

Print sample batch
Verify shapes
Check for NaN/Inf
Verify tokenization
Look for label issues

### Check Model Forward Pass

Pass single batch
Print all activation shapes
Check for NaN/Inf
Monitor gradient flow
Gradient should be O(0.001 - 0.1)

### Training Diagnostics

Loss not decreasing: LR too small
Loss diverges: LR too large
NaN in loss: Overflow, reduce LR
Train >> val: Overfitting
Train ≈ val: Good generalization

## Comparison: RNN vs Transformer

### RNN Advantages
- Constant memory (no KV cache)
- Streaming (process online)
- Good for very long sequences
- Simpler to understand

### Transformer Advantages
- Parallel computation
- Better scalability
- Attention interpretable
- Empirically stronger
- Easy to scale to billions parameters

Winner: Transformers for 2017-2026
## State-of-Art Timeline

2016: Seq2seq + Attention
2017: Transformers published
2018: BERT pre-training
2019: GPT-2 shows scaling laws
2020: GPT-3 (175B parameters)
2022: ChatGPT (fine-tuned GPT-3)
2023: GPT-4, Claude, Gemini
2024: Open weights: LLaMA, Mistral
2025: Trillion parameter models emerging

Module-04 foundation for all modern NLP!
## Hyperparameter Reference

**Model Size**
Hidden: 256-4096
Layers: 2-96
Heads: 4-32
FFN: 4*hidden typical

**Training**
Learning rate: 1e-5 to 1e-3
Batch size: 8-4096
Warmup: 5-10% of total steps
Weight decay: 0.0-0.1
Dropout: 0.0-0.3

## Common Pitfalls

1. **Learning rate too high**
   → loss diverges to NaN
   → reduce by 10x

2. **Learning rate too low**
   → loss barely decreases
   → increase, use warmup

3. **Sequence length too long**
   → OOM (out of memory)
   → reduce, use gradient checkpointing

4. **Batch size too small**
   → noisy gradients, unstable
   → increase if memory allows

## ResNets in Transformers

### Residual Connections

x → block_1 → + → LayerNorm → block_2 → +
Enables very deep networks (96 layers)
Gradient flows through skip connection
Each layer: Additive update
Without residuals: Very hard to train deep

## Layer Normalization Details

### Why LN not BatchNorm

BatchNorm: Normalize across batch
LayerNorm: Normalize across features
Transformers use LayerNorm
Independent of batch size
Works better for variable lengths
More stable numerically

## Vocabulary and Tokenization

### Byte-Pair Encoding (BPE)

Start: All bytes (256 tokens)
Merge: Most frequent pair
Repeat: Until vocab size reached
Result: Subword tokens
Typical vocab: 50K tokens
MarkGPT: Uses custom BPE

### Sentence Piece

Similar to BPE
Works directly on text
No need for initial word split
Better for non-Latin scripts
Used in many models
Example: spaCy, BERT

## Memory Efficiency

### Model Size

7B params in FP32: 28GB
7B params in FP16: 14GB
7B params in INT8: 7GB
Typical GPU: 24GB
So must use quantization or LoRA

## Practical Deployment

### Production Considerations

1. Inference latency: <100ms typical
2. Throughput: 100+ req/sec
3. Cost: GPU hours, model storage
4. Reliability: 99.9% uptime
5. Monitoring: Accuracy, latency, errors
6. Updates: New model versions
7. Compliance: Data privacy, bias

## Testing and Evaluation

### Unit Tests

```python
def test_model_shapes():
  x = torch.randn(2, 10, 768)
  model = Transformer()
  y = model(x)
  assert y.shape == (2, 10, 4)
```

### Integration Tests

Test end-to-end pipeline
From raw text to predictions
Verify post-processing

## Module 04 Capstone Project

**Build a Chatbot**

1. Fine-tune seq2seq model on dialogue data
2. Implement beam search decoding
3. Add context management (remember past)
4. Evaluate with human raters
5. Deploy as API
6. Write report analyzing errors

Challenge: Make it coherent and engaging!
## Next Steps: Module 05

Module-04 covers:
- RNNs, LSTMs, GRUs
- Attention mechanisms
- Transformers
- Case studies
- Production techniques

Module-05 continues:
- Advanced architectures
- Vision transformers
- Multimodal models
- Research frontiers
- Your own research ideas!

## Resources and References

**Papers**
- Sequence to Sequence Learning (Sutskever et al, 2014)
- Neural Machine Translation (Bahdanau et al, 2014)
- Attention is All You Need (Vaswani et al, 2017)
- BERT (Devlin et al, 2018)
- GPT-2 (Radford et al, 2019)
- GPT-3 (Brown et al, 2020)

**Implementations**
- HuggingFace Transformers
- PyTorch Lightning
- TensorFlow Hub

## Module 04 Summary

**What You Learned**
- RNN fundamentals and limitations
- LSTMs and GRUs for long-term memory
- Attention mechanisms and their power
- Transformers: The breakthrough architecture
- How to train and deploy sequence models
- Real-world case studies
- Production best practices

**What You Can Now Build**
- Machine translation systems
- Question answering
- Chatbots with context
- Text generation
- Named entity recognition
- Any NLP task!

**How This Connects**
Module-03: Neural networks
Module-04: Sequence models (this module)
Module-05: Advanced topics
Module-06: Capstone (put it all together)

You're ready for production NLP work!
## Advanced Optimization Techniques

### Gradient Accumulation

Simulate larger batch without OOM
Process N small batches, accumulate gradients
Update every N steps
Effect: Same as batch_size * N
Ideal for sequences with large token count

### Mixed Precision Training

Master weights in FP32 (stability)
Computation in FP16 (speed)
Loss scaling: Multiply by 2^16 (prevent underflow)
Result: 2-3x speedup, <1% accuracy loss
Essential for large models

### Gradient Checkpointing

Trade: Compute for memory
Forward: Don't save activations
Backward: Recompute as needed
Memory: O(√N) instead of O(N)
Speed: ~30% slower
Worth it for batch_size doubling

### Lookahead Optimizer

Keep K slow weights, N fast weights
Fast updates: Normal gradient descent
Slow updates: Every K fast steps
Benefits: More stable, better generalization
Less sensitive to learning rate

### Layer-wise Learning Rates

Lower layers learn slower (foundation)
Higher layers learn faster (task-specific)
BERT fine-tuning: 0.1-0.5 ratio
Example: Lower LR 1e-5, upper LR 1e-4
Better transfer learning

### Warmup Strategies

Linear warmup: 0 → LR over fraction of steps
Helps optimization stability
10% of total steps typical
Alternatives: Square root, exponential
Important for transformers especially

### Learning Rate Scheduling

Constant: Simple baseline
Linear decay: Decrease linearly
Cosine: cos(π * t / T) shaped
Step: Decay every N steps
Exponential: Exponential decay
Empirically: Cosine ≈ linear, both good

### Weight Decay & L2 Regularization

Standard L2: Add 0.5 * λ * ||w||^2 to loss
AdamW: Decouple weight decay from gradient
Weight decay ≠ L2 with adaptive optimizers!
Typical λ: 0.01-0.1
Prevents overfitting

### Label Smoothing

One-hot target: [0, 1, 0, 0]
Smoothed: [0.01, 0.91, 0.01, 0.01] (with ε=0.1)
Prevents overconfidence
Regularization effect
Typical ε: 0.1
Improves generalization

## Interpretability Deep Dive

### Probing Tasks

Train classifier on hidden states
If high accuracy: Layer encodes feature
Example: Predict POS from h_t
Result: Earlier layers = syntax, later = semantics
Reveals learned representations

### Saliency Maps

∇ L / ∇ x: Input gradient
Magnitude: How much input affects output
Visualization: Color heat map
Interpretation: Which tokens matter
Caveat: Not always meaningful for text

### Attention Head Analysis

Query-key dot products
Visualize as heatmap
Pattern 1: Attending to next token
Pattern 2: Attending to same position
Pattern 3: Positional patterns
Interpretable but not complete story

### SHAP Values

Game theory approach
Shapley value: Fair feature contribution
Expensive to compute (combinatorial)
Approximations exist (SHAPATTY)
Gold standard for interpretability

### Influence Functions

Which training examples help/hurt?
Trace gradient backward in Hessian
Computationally expensive
Useful for data debugging
Find adversarial examples

## Domain Adaptation

### Covariate Shift

Train and test: Different input distribution
Example: Medical text differs from typical
Solution: Importance reweighting
Or: Continued pre-training on target

### Domain-Adaptive Pre-training

DAPT: Further pre-train on target domain
10K-100K steps on unlabeled target
Then fine-tune on task
Improves +5-10% on small fine-tuning sets

### Task-Adaptive Pre-training

TAPT: Pre-train further on task data
Only 100 steps sufficient
Very effective for low-resource tasks
Can beat full fine-tuning of generic model

### Few-Shot Learning

Meta-learning: Learn to learn
MAML: Model-agnostic meta-learning
Update parameters for few examples
Learn good initialization
Enables fast adaptation

## Multilingual Models

### mBERT Design

104 languages in single model
Shared vocabulary across languages
WordPiece tokenization
110K tokens total (vs 30K monolingual)
Trade-off: More tokens, covers more

### Cross-lingual Transfer

Train on English, test on Hindi
Transfer quality: 80%+ on many pairs
Magic: Shared embedding space
Better with similar languages
Enables low-resource NLP

### Language-specific Fine-tuning

Start: mBERT (multilingual)
Fine-tune: On target language data
Effect: Specialize to language
Performance: Often better than monolingual
Due to multilingual pre-training signal

## Continual Learning

### Catastrophic Forgetting

Train on task A
Fine-tune on task B
Performance on A: Drops to 10%
Weights: Optimized away from A
Challenge: Maintain both

### Elastic Weight Consolidation

Compute parameter importance
Fisher information matrix
Penalize changing important params
Loss = task_loss + λ * Σ F_i * (θ_i - θ_old)^2
Achieves 80%+ on both tasks

### Replay Methods

Keep some examples from task A
Mix with task B during training
Simple: Just replay
Effective: Prevents catastrophic forgetting
Memory-efficient: Store embeddings not raw data

### Parameter Isolation

Different tasks: Different parameters
Adapters: Task-specific modules
Sparse masks: Select per task
Complete isolation: No interference
Trade-off: More storage (LoRA helps)

## Adversarial Training

### Adversarial Examples

Small input perturbation
Flips model prediction
Example: Change 1-2 words
Model thinks should change sentiment
Shows model brittleness

### Defense: Adversarial Training

Generate adversarial examples
Augment training data
Train on both natural + adversarial
Result: Robust to perturbations
Cost: ~3x slower training

## Uncertainty Quantification

### Model Confidence

Max softmax: Simple but miscalibrated
Better use: Temperature scaling
Output prob / T where T ≈ 1.5
Helps calibration
On OOD data: Confidence no good

### Bayesian Deep Learning

Uncertainty over weights
Variational inference
Approximation: MC Dropout
Forward pass N times, average
Get uncertainty from variance
Expensive but principled

### Out-of-Distribution Detection

Detect when input unusual
Methods: Max probability, energy-based
Mahalanobis distance: Distance from training
Useful for safety-critical systems
Important for production deployment

## Efficiency: Model Distillation

### Knowledge Distillation

Large teacher → small student
Student learns soft targets from teacher
Temperature: Soften distribution
T=3-20 typical
Result: 100M student = 90% of 7B teacher

### Architecture Distillation

Distill BERT-large → BERT-small
6 layers instead of 12
Result: 40% smaller, 90% accuracy
Student matches teacher logits + attention
ALBERT: Parameter sharing in layers

### Task-specific Distillation

Distill-BERT: Fine-tune then distill
Results: 2-4x faster inference
Minimal accuracy loss
Great for production
Can stack multiple distillations

## Efficiency: Quantization

### Post-training Quantization

Train: FP32
Quantize: INT8/INT4 after
Fastest: No retraining
Accuracy: 0.5-2% loss (INT8), 10%+ (INT4)
Easy to implement

### Quantization-Aware Training

Simulate quantization during training
Model learns robust representations
INT4 with QAT: 5% loss vs 25% PTQ
Training time: 2x
Worth it for deployment

### Mixed-bit Quantization

Different bits per layer
Attention: INT8
FFN: INT4
Balance: (int4 gradual decrement)
Result: Optimal size-accuracy trade-off

## Efficiency: Pruning

### Magnitude Pruning

Remove weights with small magnitude
Simple to implement
Result: 30-50% sparsity
Inference: Hard to accelerate (unstructured)
Better with hardware support

### Structured Pruning

Remove entire neurons/channels
Hard to determine importance
Methods: Lottery ticket, fisher pruning
Result: 40-60% speed improvement
Can achieve 10:1 compression

### Lottery Ticket Hypothesis

Initialize, train, prune, reset
Pruned network (lottery ticket) ≈ trained!
Key: Right initialization + pruning mask
Explains why networks over-parameterized
Can prune 90%+ and still work

## Multimodal Architecture

### Vision Transformers

Split image into patches
Linear projection to embeddings
Apply standard transformer
Comparable to CNNs for classification
Better for downstream tasks

### CLIP: Image-Text

Contrastive learning
Image encoder and text encoder
Maximize: sim(image, caption)
Result: Zero-shot classification
Foundation for image search

### Speech-to-Text

Wav2Vec: Self-supervised speech
HuBERT: Discrete speech units
Transformers for speech: Conformer
Combined with text: Encoder-decoder
End-to-end speech recognition

## MarkGPT Implementation Details

### Tokenizer

Custom BPE (Byte-Pair Encoding)
100K vocabulary
Trained on entire corpus
Preserves rare words
Balanced subword lengths

### Pre-training Data

CommonCrawl: 400B tokens
Books: 100B tokens
Wikipedia: 50B tokens
Code: 50B tokens (improves math)
Total: 600B tokens (MarkGPT-7B)
Ratio: 60% CC, 17% books, 8% wiki, 8% code, 7% other

### Training Infrastructure

8 GPUs (V100): ~1 month
16 GPUs (A100): ~1 week
128 GPUs (TPU): ~1 day
Batch size: 512 tokens * 64
Learning rate: 0.0001 with warmup
Gradient accumulation: Steps=8

### Model Architecture

32 transformer layers
4096 hidden dimension
32 attention heads
16384 FFN dimension
RoPE positional encoding
Pre-normalization (LayerNorm before)
Total: 7B parameters

### Inference System

vLLM: Efficient batching
FlashAttention: Fast attention kernel
KV-cache: Avoid recomputation
Continuous batching: Remove finished sequences
Result: 100x speedup vs naive

## Production Deployment

### API Design

POST /generate
Input: prompt, max_tokens, temperature
Output: text, tokens, metadata
Rate limiting: Prevent abuse
Auth: API key validation
Versioning: Multiple model versions

### Monitoring

Latency: p50, p95, p99
Throughput: tokens/sec
Cost: $/request
Quality: User feedback
Errors: Failure rate
Bias: Check for outputs

### Scaling

Load balancer: Distribute requests
Caching: Cache frequent prompts
Batching: Combine requests
Sharding: Split model across GPUs
Replication: Multiple copies
Auto-scaling: Add GPUs under load

### Cost Optimization

GPU hours: $1-2 per hour
Quantization: 4x faster, 4x cheaper
Batching: Better utilization
Spot instances: 70% cheaper (interruptible)
Caching: Reduce redundant computation
Distillation: Smaller model for simple queries

## Safety and Bias

### Bias Detection

Stereotype tests: Bias-in-Bios
Template-based: Check output variance
Manual review: Human evaluation
Demographic parity: Similar across groups
Equalized odds: Similar errors across groups

### Bias Mitigation

Data: Balance training set
Sampling: Over/under-sample groups
Constraints: Fairness-aware objectives
Post-processing: Adjust predictions
Better: Combine approaches

### Harmful Content

Filter: Block known harmful patterns
Classifiers: Detect unsafe outputs
Human review: High-risk cases
Rate limiting: Prevent attack sequences
Logging: Track for improvement

### Privacy

Data anonymization: Remove identifiers
Differential privacy: Probabilistic guarantee
Federated learning: Train on device
Secure enclaves: Trusted execution
User consent: Transparent about data

## Research Directions

### Emerging Topics

1. Mixture of Experts: Sparse scaling
2. Retrieval Augmentation: Fact checking
3. Multimodal: Text + Vision + Audio
4. Fast inference: Sub-millisecond
5. Continual learning: Learn new tasks
6. Steering: Control model behavior
7. Hardware: Custom chips for LLMs

## Capstone: End-to-End System

**Project Requirements**

1. Data collection: Scrape or download
2. Preprocessing: Tokenization, cleaning
3. Training: Fine-tune on domain
4. Evaluation: Metrics + human eval
5. Optimization: Quantize, distill, or prune
6. Deployment: API with monitoring
7. Documentation: Write clear guide

**Suggested Tasks**
- Chatbot fine-tuned on your domain
- Code generation from docstrings
- Summarization of long documents
- Translation to low-resource language
- Question answering on custom knowledge base

## Module 04 Comprehensive Summary

**Core Concepts Mastered**

1. RNNs & sequence processing
2. Vanishing/exploding gradients
3. LSTMs & GRUs (practical)
4. Attention mechanisms
5. Transformers (deep understanding)
6. Fine-tuning & transfer learning
7. Decoding strategies
8. Optimization techniques
9. Interpretability methods
10. Production deployment

**Practical Skills**

✓ Implement RNN/LSTM from scratch
✓ Debug sequence model training
✓ Fine-tune transformers
✓ Implement beam search
✓ Analyze attention patterns
✓ Apply gradient clipping
✓ Use batch normalization
✓ Optimize for deployment
✓ Monitor in production
✓ Mitigate bias

**Key Insights**

1. Sequence models need careful gradient management
2. Attention is more powerful than we expected
3. Transformers: Fully parallelizable = game changer
4. Transfer learning works remarkably well
5. Scaling laws predict future progress
6. Data quality matters more than size
7. Interpretability is hard but important
8. Production is 10x harder than research
9. Safety and bias are mandatory, not optional
10. This field moves fast: stay updated!

**What's Next**

✓ Module-04: Complete (you are here!)
→ Module-05: Advanced architectures
→ Module-06: Capstone projects
→ Research papers (weekly reading)
→ Contribute to open-source
→ Publish your own work

You've learned the fundamentals.
Now build something amazing!

