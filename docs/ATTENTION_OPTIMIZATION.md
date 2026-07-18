# Attention Optimization Techniques

## Standard Attention: O(N²) Problem

### Memory/Compute Bottleneck

For sequence length N and d_model M:

```
Attention matrix size: N × N
Memory: O(N²)
Compute: O(N²)

Example:
- N = 4096 (seq_len)
- Attention matrix: 4096 × 4096 = 16.7M scalars
- Memory: 16.7M × 4 bytes = 67 MB (per head!)
- With 8 heads: 536 MB
- With batch_size=32: 17 GB (entire GPU!)
```

### Standard Attention Algorithm

```python
def standard_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, num_heads, seq_len, head_dim)
    Returns: (batch, num_heads, seq_len, head_dim)
    """
    
    # Compute scores: (batch, num_heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    
    # Apply mask (causal for LLMs)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)  # Requires full N² matrix
    
    # Weighted sum: (batch, num_heads, seq_len, head_dim)
    output = torch.matmul(attn_weights, V)
    
    return output

# Problem: Softmax requires full attention matrix in memory
```

---

## Flash Attention: Tiling & Recomputation

### Core Insight

Don't store full attention matrix. Compute blocks sequentially:

```
Block 1: Attention for tokens 0-511
  - Keep only block in memory
  - Accumulate results
  - Discard block

Block 2: Attention for tokens 512-1023
  - Repeat
  
Block N: ...
```

### Flash Attention Algorithm

```python
def flash_attention_simplified(Q, K, V, block_size=64):
    """
    Simplified Flash Attention concept.
    (Real impl much more optimized)
    """
    
    seq_len = Q.shape[2]
    output = torch.zeros_like(Q)
    
    # Block-wise computation
    for block_idx in range(0, seq_len, block_size):
        end_idx = min(block_idx + block_size, seq_len)
        
        # Get block of Q
        Q_block = Q[:, :, block_idx:end_idx, :]
        
        # Compute attention for this block against all K, V
        scores = torch.matmul(Q_block, K.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        
        # Accumulate output
        output[:, :, block_idx:end_idx, :] = torch.matmul(attn, V)
    
    return output

# Memory savings: Only keep one block + accumulator
# Memory: O(N) instead of O(N²)!
```

### Real Flash Attention (Conceptual)

Three key optimizations:

1. **Tiling**: Process attention in blocks (reduces VRAM)
2. **Recomputation**: Don't store full softmax, recompute in backward
3. **Fusion**: Fuse kernels (GPU compute optimization)

```python
# PyTorch 2.0+: Built-in Flash Attention
torch.nn.functional.scaled_dot_product_attention(
    q, k, v,
    attn_mask=mask,
    dropout_p=0.0 if not training else 0.1,
    is_causal=True  # Enable causal masking optimization
)

# Usage in transformer
class FlashAttentation(nn.Module):
    def forward(self, Q, K, V, mask=None):
        return F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            is_causal=True
        )
```

### Benchmarks

```
Standard Attention:
  - Forward: 100ms
  - Backward: 150ms
  - Memory: 17 GB (batch_size=32, seq_len=4096)

Flash Attention v1:
  - Forward: 70ms (30% faster)
  - Backward: 80ms (45% faster)
  - Memory: 3-4 GB (5x reduction!)

Flash Attention v2 (with pipelining):
  - Forward: 50ms (50% faster)
  - Backward: 60ms (60% faster)
  - Memory: 2-3 GB

Note: v2 heavily optimized for A100/GPU kernels
```

---

## Multi-Query Attention (MQA)

### Problem

Key-Value cache for long sequences is huge:

```
Standard Multi-Head:
- K cache: (batch, num_heads=8, seq_len, head_dim=64)
- V cache: (batch, num_heads=8, seq_len, head_dim=64)
- Total: 8 × seq_len × 64 × 4 bytes

At seq_len=1000:
- Per sample: 2 MB

At seq_len=4096 (longer context):
- Per sample: 2 MB × 4 = 8 MB (problematic for many samples)
```

### Solution: Share K/V

Instead of K/V per head → one K/V pair per model:

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Standard
        self.q_proj = nn.Linear(d_model, d_model)
        
        # Multi-query: Single K/V for all heads
        self.k_proj = nn.Linear(d_model, self.head_dim)  # Not × num_heads!
        self.v_proj = nn.Linear(d_model, self.head_dim)
        
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        
        # Q: (batch, seq_len, d_model)
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        
        # K, V: Single per model (not per head)
        k = self.k_proj(x)  # (batch, seq_len, head_dim)
        v = self.v_proj(x)  # (batch, seq_len, head_dim)
        
        # Broadcast K/V to all heads (implicitly in attention)
        scores = torch.matmul(q, k.unsqueeze(1).transpose(-2, -1))
        # q: (batch, num_heads, seq_len, head_dim) @ 
        # k.T: (batch, 1, head_dim, seq_len) 
        # → broadcast → (batch, num_heads, seq_len, seq_len)
        
        attn = torch.softmax(scores, dim=-1)
        
        # Same V for all heads
        output = torch.matmul(attn, v.unsqueeze(1))
        # output: (batch, num_heads, seq_len, head_dim)
        
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, d_model)
        
        return self.out_proj(output)
```

### Memory Impact

```
Standard: 8 × seq_len × 64 = 8 seq_len × 64
MultiQuery: 1 × seq_len × 64 = seq_len × 64
Savings: 8x (!!)

For long-context inference:
- 4096 tokens, 8 heads: 2 MB → 250 KB per sample
```

---

## Grouped Query Attention (GQA)

### Balance Between Standard & MQA

Standard Multi-Head: 1 K/V per head
Multi-Query: 1 K/V total
**Grouped**: n K/V groups < num_heads

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups=2):
        super().__init__()
        assert num_heads % num_kv_groups == 0
        
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_model // num_heads
        
        # Q: Full heads
        self.q_proj = nn.Linear(d_model, d_model)
        
        # K/V: Grouped (fewer than num_heads)
        kv_dim = (d_model // num_heads) * num_kv_groups
        self.k_proj = nn.Linear(d_model, kv_dim)
        self.v_proj = nn.Linear(d_model, kv_dim)
    
    def forward(self, x):
        q = self.q_proj(x)  # (batch, seq, d_model)
        k = self.k_proj(x)  # (batch, seq, kv_dim)
        v = self.v_proj(x)  # (batch, seq, kv_dim)
        
        # Reshape for attention
        batch = x.shape[0]
        seq_len = x.shape[1]
        
        # Q: (batch, num_heads, seq, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K: (batch, num_kv_groups, seq, head_dim)
        k = k.view(batch, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        
        # V: (batch, num_kv_groups, seq, head_dim)
        v = v.view(batch, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        
        # Expand for broadcasting (repeat k/v: each group → multiple heads)
        k = k.repeat_interleave(self.num_heads // self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_groups, dim=1)
        
        # Standard attention
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # ... rest of forward
```

### GQA Trade-offs

| Type | K/V Count | Quality | Memory | Latency |
|------|-----------|---------|--------|---------|
| Standard | 8 | 100% | 1x | 1x |
| GQA (8→4) | 4 | 98% | 0.65x | 0.98x |
| GQA (8→2) | 2 | 95% | 0.35x | 0.95x |
| Multi-Query | 1 | 90% | 0.25x | 0.90x |

GQA best practice: Use 4-8 groups for good balance.

---

## KV-Cache Management

During inference, cache grows:

```python
class KVCache:
    def __init__(self, max_length, batch_size, num_heads, head_dim):
        self.max_length = max_length
        self.current_length = 0
        
        # Preallocate
        self.k_cache = torch.zeros(
            batch_size, num_heads, max_length, head_dim
        )
        self.v_cache = torch.zeros(
            batch_size, num_heads, max_length, head_dim
        )
    
    def append(self, k, v):
        """Add new K/V to cache."""
        # k, v: (batch, num_heads, 1, head_dim) [single token]
        
        start = self.current_length
        end = start + k.shape[2]
        
        self.k_cache[:, :, start:end] = k
        self.v_cache[:, :, start:end] = v
        
        self.current_length = end
    
    def get(self):
        """Return cache up to current position."""
        return self.k_cache[:, :, :self.current_length],
               self.v_cache[:, :, :self.current_length]

# Usage during generation
cache = KVCache(max_length=4096, batch_size=1, num_heads=8, head_dim=64)

for token_idx in range(max_length):
    # Generate one token
    with torch.no_grad():
        k, v = compute_kv(input_tokens, ...)  # Current token K/V
    
    cache.append(k, v)
    
    # Attention uses cache
    attn_out = attention(q, cache.k_cache, cache.v_cache)
```

---

## MarkGPT Attention Configuration

```python
class MarkGPTAttentionConfig:
    # Flash Attention v2 (if available)
    use_flash_attention = True
    
    # Multi-head or grouped query
    attention_type = "multi_head"  # "multi_head" | "grouped_query"
    num_kv_groups = 8  # If GQA
    
    # Dropout
    attention_dropout = 0.1
    
    # KV cache
    cache_type = "dynamic"  # "dynamic" | "static"
    max_cache_length = 4096
    
    # Optimization
    gradient_checkpointing = True  # Trade compute for memory
    use_rope = True  # Rotary position embeddings
    
    # Quantization (for inference)
    quantize_kv = False

def create_optimized_attention(config):
    if config.use_flash_attention:
        # PyTorch 2.0+
        return FlashAttentionModule(config)
    elif config.attention_type == "grouped_query":
        return GroupedQueryAttention(config)
    else:
        return StandardMultiHeadAttention(config)
```

---

**Attention Optimization v1.0**
**Last Updated**: 2024
