# Attention Mechanisms & Mathematical Foundation

## Transformer Attention Explained

### Core Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ = Query matrix (batch_size, seq_len, d_k)
- $K$ = Key matrix
- $V$ = Value matrix
- $d_k$ = Dimension of keys
- $\sqrt{d_k}$ = Scaling factor (prevents vanishing gradients)

### Multi-Head Attention

```
input (batch, seq_len, d_model=768)
    ↓
Split into 12 heads (batch, seq_len, d_model=64)
    ├─ Head 1: Self-attention
    ├─ Head 2: Self-attention
    ├─ ...
    └─ Head 12: Self-attention
    ↓
Concatenate (batch, seq_len, d_model=768)
    ↓
Linear projection
```

**Formula**:
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$$

Where each head computes:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Why Multiple Heads?

- **Head 1**: Might focus on subject-verb relations
- **Head 2**: Might focus on adjectives modifying nouns
- **Head 3**: Might focus on pronouns and references
- **Head 4-12**: Different syntactic/semantic patterns

---

## RoPE (Rotary Position Embeddings)

### Motivation

Traditional absolute positional encodings:
- Fixed positions (don't extrapolate beyond training length)
- Added to embeddings (hurts semantic meaning)

RoPE solution:
- Rotate vectors in complex plane
- Naturally encodes relative positions
- Extrapolates to longer sequences

### Formula

For position $m$ and dimension pair $(2i, 2i+1)$:

$$\begin{pmatrix} x_i^m \\ x_{i+1}^m \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_i \\ x_{i+1} \end{pmatrix}$$

Where $\theta_i = 10000^{-2i/d}$

### Implementation

```python
def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to tensor x."""
    # x: (batch, seq_len, num_heads, head_dim)
    # cos, sin: (seq_len, head_dim // 2)
    
    # Split into real and imaginary parts
    x_rot = torch.cat(
        [-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]],
        dim=-1
    )
    
    # Rotate: z' = z * e^(i*m*theta) = z*cos + z_rot*sin
    return x * cos + x_rot * sin
```

---

## Causal Masking (Why)

In language modeling, we can't let the model "cheat" by looking at future tokens:

```
Sequence: [The, cat, sat, on, mat]
                           ↙
Position 4: Model should predict "mat"
           But can only see: [The, cat, sat, on]
           NOT: [The, cat, sat, on, mat]
```

Causal mask pattern (upper triangle is masked):
```
Q/K positions:  0  1  2  3  4
Position 0:     ✓  ✗  ✗  ✗  ✗
Position 1:     ✓  ✓  ✗  ✗  ✗
Position 2:     ✓  ✓  ✓  ✗  ✗
Position 3:     ✓  ✓  ✓  ✓  ✗
Position 4:     ✓  ✓  ✓  ✓  ✓
```

**Implementation**:
```python
def attention_with_causal_mask(Q, K, V, seq_len):
    # Compute attention scores
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    
    # Create causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len)) == 1
    scores.masked_fill_(~mask, float('-inf'))
    
    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Combine with values
    output = attn_weights @ V
    return output
```

---

## Flash Attention (Why It's Fast)

### Traditional Attention Problem

```
Compute scores: O(Nˆ2) memory and time
    ↓
Store N×N matrix in GPU memory
    ↓
Softmax (requires full matrix)
    ↓
Multiply with values
```

For N=2048 (seq_len) and d_model=768:
- Attention matrix size: 2048 × 2048 × 4 bytes = 16MB
- Per batch: 16MB × 32 = 512MB (significant!)

### Flash Attention Solution

```
Divide into blocks:
    ↓
Compute attention for block 1
    ↓
Accumulate results
    ↓
Move to next block
    (Only one block in memory at a time)
```

**Formula**:
$$O_i = S_i D_i^{-1} V_j$$

Where:
- $D_i$ = diagonal matrix of softmax denominators
- Blocks processed sequentially

**Speedup**: 2-3x faster, 20% less memory!

---

## Grouped Query Attention (GQA)

Standard multi-head: h query heads, h key/value heads
→ **Issue**: Large memory for long sequences (KV cache grows)

Solution: Multiple queries share one key/value head

```
Standard Attention:  Q: h heads, K: h heads, V: h heads
GQA (h=8, g=2):     Q: 8 heads, K: 2 heads, V: 2 heads
                    (4 query heads per KV head)
```

**Memory Savings**:
- KV cache: h/g × smaller
- 2-3x reduction in some cases

---

## Cross-Attention (Encoder-Decoder Only)

MarkGPT uses decoder-only, but for reference:

```
Encoder Output
    ↓ (serves as K, V)
    
Decoder Query
    ↓ (self-attention with encoder K, V)
    
Attention(Q_decoder, K_encoder, V_encoder)
```

Used in models like T5, mBART for translation.

---

**Mathematical Reference v1.0**
**Last Updated**: 2024
