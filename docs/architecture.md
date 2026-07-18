# MarkGPT Architecture Diagram

## Full Model Architecture

```
Input Tokens [batch_size, seq_len]
        ↓
    Token Embedding Layer
        ↓ [batch_size, seq_len, d_embd]
    Position Embedding Layer
        ↓ [batch_size, seq_len, d_embd]
    Embedding Dropout
        ↓
    ┌─────────────────────────────┐
    │  Transformer Block 1         │
    │  ├─ LayerNorm                │ 
    │  ├─ CausalSelfAttention      │
    │  │  ├─ Q,K,V Projections     │
    │  │  ├─ Scaled Dot-Product    │
    │  │  ├─ Causal Mask           │
    │  │  └─ Output Projection     │
    │  ├─ Residual Connection      │
    │  ├─ LayerNorm                │
    │  ├─ Feed-Forward Network     │
    │  │  ├─ Linear(d → 4d)        │
    │  │  ├─ GELU Activation       │
    │  │  └─ Linear(4d → d)        │
    │  └─ Residual Connection      │
    └─────────────────────────────┘
        ↓
    ... (repeat n_layer times) ...
        ↓
    Final LayerNorm
        ↓ [batch_size, seq_len, d_embd]
    Language Model Head (Linear)
        ↓ [batch_size, seq_len, vocab_size]
    Logits (unnormalized)
        ↓
    Output Probabilities (softmax)
```

## Attention Head Computation

```
Input: [batch_size, seq_len, d_embd]
        ↓
    Split into n_head heads
    Each has shape [batch_size, n_head, seq_len, head_size]
        ↓
    Q, K, V Projections
        ↓
    Scaled Dot-Product: scores = (Q @ K^T) / √head_size
    Shape: [batch_size, n_head, seq_len, seq_len]
        ↓
    Apply Causal Mask (zero out future positions)
        ↓
    Softmax over key dimension
        ↓
    Attention weights × Values
        ↓
    Concatenate n_head heads
        ↓
    Output Projection
        ↓
Output: [batch_size, seq_len, d_embd]
```

## Feed-Forward Network

```
Input: x [batch_size, seq_len, d_embd]
    ↓
Linear(d_embd → 4*d_embd)
    ↓
GELU Activation
    ↓
Linear(4*d_embd → d_embd)
    ↓
Output: [batch_size, seq_len, d_embd]
```

## Model Sizes

### MarkGPT-Nano (~2M parameters)
- Embedding Dim: 128
- Attention Heads: 4 (32-dim each)
- Layers: 4
- FFN Inner Dim: 512

### MarkGPT-Small (~10M parameters)
- Embedding Dim: 256
- Attention Heads: 8 (32-dim each)
- Layers: 6
- FFN Inner Dim: 1024

### MarkGPT-Base (~85M parameters)
- Embedding Dim: 512
- Attention Heads: 8 (64-dim each)
- Layers: 12
- FFN Inner Dim: 2048

## Key References

- Vaswani et al. (2017): "Attention Is All You Need" (original Transformer)
- Radford et al. (2019): "Language Models are Unsupervised Multitask Learners" (GPT-2)
- Dao et al. (2022): "FlashAttention: Fast and Memory-Efficient Exact Attention"
