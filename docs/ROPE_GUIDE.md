# RoPE Implementation Deep Dive

## Background: Why RoPE Over Absolute Positions?

### Absolute Positional Encoding Problem

```python
# Traditional approach in original Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048):
        self.pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            -(math.log(10000.0) / d_model))
        
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:x.size(1)]
        return x

# Issues:
# 1. Fixed maximum sequence length (2048)
# 2. Poor extrapolation to longer sequences
# 3. Embedding + position = semantic pollution
# 4. Not rotation-invariant
```

### Relative Position Encoding Problem

```python
# ALiBi (Attention with Linear Biases) approach
# Adds distance-based bias to attention scores

def apply_alibi(attn_scores, max_distance=512):
    # attn_scores: (batch, num_heads, seq_len, seq_len)
    
    positions = torch.arange(attn_scores.size(-1))
    distance = positions.unsqueeze(1) - positions.unsqueeze(0)
    
    # Linear bias in log space
    bias = -abs(distance) * (1.0 / max_distance)
    return attn_scores + bias.unsqueeze(0).unsqueeze(0)

# Issues:
# 1. Requires explicit bias computation
# 2. Not as expressive as rotation-based
# 3. Separate from attention mechanism
```

---

## RoPE Solution

### Core Concept: Complex Plane Rotation

Treat 2D vector pairs as complex numbers and rotate:

```
Vector in 2D:     (x, y)
Complex number:   x + iy
Rotation by θ:    (x + iy) * e^(iθ) = result
                = (x*cos(θ) - y*sin(θ)) + i(x*sin(θ) + y*cos(θ))
                = (x*cos(θ) - y*sin(θ), x*sin(θ) + y*cos(θ))
```

For each position $m$ and dimension pair $(2i, 2i+1)$:

$$R_m^{(d)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

Where $\theta_i = 10000^{-2i/d}$ (borrowed from original Transformer)

### Why This Works

1. **Relative position encoding**: Attention score at $(q_m, k_n)$ depends on $m - n$
   - Proof: $(R_m^{(d)} q) \cdot (R_n^{(d)} k) = q \cdot R_{m-n}^{(d)} k$
   - i.e., rotation preserves inner product after relative rotation

2. **Extrapolation**: Works for sequences longer than training
   - Rotations are periodic and smooth
   - Model learns the pattern

3. **No semantic interference**: Position is in input space, not added

4. **Equivariance**: Respects rotational structure of attention

---

## Implementation in PyTorch

### Method 1: Efficient Matrix Implementation

```python
import torch
import math

def precompute_freqs_cis(dim, seq_len, base=10000):
    """
    Precompute cos/sin values for all positions.
    
    Args:
        dim: Model dimension (d_model)
        seq_len: Maximum sequence length
        base: Base for frequency calculation (10000)
    
    Returns:
        freqs: (seq_len, dim//2) - complex exponentials
        cos_vals: (seq_len, dim//2)
        sin_vals: (seq_len, dim//2)
    """
    # Compute θ_i = base^(-2i/d) for each i
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    # m * θ_i for all positions m
    t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)  # (seq_len, dim//2)
    
    # e^(i * m * θ_i) = cos(m*θ_i) + i*sin(m*θ_i)
    cos_vals = torch.cos(freqs)  # (seq_len, dim//2)
    sin_vals = torch.sin(freqs)  # (seq_len, dim//2)
    
    return freqs, cos_vals, sin_vals

def apply_rotary_pos_emb(x, cos_vals, sin_vals):
    """
    Apply RoPE to embeddings.
    
    Args:
        x: (batch, seq_len, num_heads, head_dim)
        cos_vals: (seq_len, head_dim//2) precomputed cosines
        sin_vals: (seq_len, head_dim//2) precomputed sines
    
    Returns:
        x_rotated: Same shape as x, with rotary embeddings applied
    """
    # Extract pairs: (d0, d1), (d2, d3), ...
    x1 = x[..., : x.shape[-1] // 2]      # First half
    x2 = x[..., x.shape[-1] // 2 :]      # Second half
    
    # Rotation formula: (x' = x*cos - y*sin, y' = x*sin + y*cos)
    cos = cos_vals.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
    sin = sin_vals.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
    
    x_rotated_1 = x1 * cos - x2 * sin
    x_rotated_2 = x1 * sin + x2 * cos
    
    # Interleave back
    x_rotated = torch.cat([x_rotated_1, x_rotated_2], dim=-1)
    return x_rotated
```

### Method 2: Efficient In-Place Using Complex Numbers

```python
def apply_rope_complex(q, k, cos_vals, sin_vals):
    """
    Optimized RoPE using PyTorch complex tensor support.
    
    Args:
        q: Query (batch, num_heads, seq_len, head_dim)
        k: Key (batch, num_heads, seq_len, head_dim)
        cos_vals, sin_vals: Precomputed position encodings
    
    Returns:
        q_rotated, k_rotated: Rotated embeddings
    """
    # Reshape to complex numbers for vectorized rotation
    # Interpret (a, b, c, d) as (a+bi, c+di)
    
    seq_len = q.shape[2]
    cos = cos_vals[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
    sin = sin_vals[:seq_len].unsqueeze(0).unsqueeze(0)
    
    # Replicate for interleaved pairs
    cos_full = torch.cat([cos, cos], dim=-1)  # (1, 1, seq_len, dim)
    sin_full = torch.cat([sin, sin], dim=-1)
    
    # Apply rotation
    q_rotated = q * cos_full - q.roll(q.shape[-1]//2, dims=-1) * sin_full
    k_rotated = k * cos_full - k.roll(k.shape[-1]//2, dims=-1) * sin_full
    
    return q_rotated, k_rotated
```

---

## Integration into Transformer

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Precompute RoPE
        _, self.cos_cached, self.sin_cached = precompute_freqs_cis(
            self.head_dim, max_seq_len
        )
        
        # Attention projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        batch, seq_len, dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE
        q = apply_rotary_pos_emb(q, self.cos_cached, self.sin_cached)
        k = apply_rotary_pos_emb(k, self.cos_cached, self.sin_cached)
        
        # Standard attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ v
        
        # Merge heads and project
        output = output.view(batch, seq_len, dim)
        output = self.out_proj(output)
        
        return output
```

---

## RoPE Variants

### Linear Interpolation (Extending Past Training Length)

```python
def apply_rope_with_extrapolation(q, k, cos_vals, sin_vals, 
                                  seq_len, train_seq_len=2048):
    """
    Extrapolate RoPE by linear scaling of frequencies.
    """
    if seq_len <= train_seq_len:
        return apply_rotary_pos_emb(q, cos_vals, sin_vals)
    
    # Scale frequencies for longer sequences
    scale = seq_len / train_seq_len
    cos_scaled = cos_vals.clone()
    sin_scaled = sin_vals.clone()
    
    # Recompute for extended range
    _, cos_ext, sin_ext = precompute_freqs_cis(
        cos_vals.shape[-1], seq_len, 
        base=10000 * scale
    )
    
    return apply_rotary_pos_emb(q, cos_ext, sin_ext)
```

### YaRN (Yet another RoPE eXtension)

Uses dynamic scaling per frequency:
- Higher frequencies: standard scaling
- Lower frequencies: aggressive NTK-based scaling

```python
def apply_rope_yarn(q, k, cos_vals, sin_vals, seq_len, 
                     ntk_alpha=1.0, train_seq_len=2048):
    """Yarn variant with NTK scaling."""
    
    # Compute frequency-dependent scale
    scale = (seq_len / train_seq_len - 1) * ntk_alpha + 1
    
    # Apply frequency-dependent scaling
    dim_half = cos_vals.shape[-1]
    freq_idx = torch.arange(dim_half) / (dim_half - 1)
    
    # Low frequency: aggressive scaling
    # High frequency: minimal scaling
    freq_scale = 1 + (scale - 1) * (1 - freq_idx)
    
    # Recompute with scaled base
    base_scaled = 10000 ** (1.0 / (freq_scale * dim_half))
    _, cos_yarn, sin_yarn = precompute_freqs_cis(
        2 * dim_half, seq_len, base=base_scaled
    )
    
    return apply_rotary_pos_emb(q, cos_yarn, sin_yarn)
```

---

## Benchmarking

```python
def benchmark_rope():
    seq_len = 4096
    batch_size = 32
    num_heads = 8
    head_dim = 64
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn_like(q)
    
    # Precompute
    _, cos_vals, sin_vals = precompute_freqs_cis(head_dim, seq_len)
    
    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        q_rotated = apply_rotary_pos_emb(q, cos_vals, sin_vals)
    end.record()
    torch.cuda.synchronize()
    
    print(f"Time per forward: {start.elapsed_time(end) / 100:.3f}ms")
    # Typical: ~0.5ms for full batch
```

---

**RoPE Implementation v1.0**
**Last Updated**: 2024
