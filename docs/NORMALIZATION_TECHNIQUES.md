# Normalization Techniques in Transformers

## Layer Normalization (Pre-Post Strategies)

### Standard Layer Norm

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- $\mu$, $\sigma^2$ = mean and variance across feature dimension
- $\gamma$, $\beta$ = learnable affine parameters
- $\epsilon$ = small constant for numerical stability (1e-5)

```python
class LayerNormalizationExplained(nn.Module):
    """Manual implementation for understanding."""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)     # (batch, seq_len, 1)
        
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta

# In practice, use PyTorch's optimized version
layer_norm = nn.LayerNorm(d_model=768)
```

### Pre-Normalization vs Post-Normalization

**Original Transformer (Post-Norm)**:
```
Attention
    ↓
Add (skip connection)
    ↓
LayerNorm ← Applied AFTER
    ↓
FFN
    ↓
Add
    ↓
LayerNorm
```

**Modern Practice (Pre-Norm)**:
```
LayerNorm ← Applied BEFORE
    ↓
Attention
    ↓
Add (skip connection)
    ↓
LayerNorm
    ↓
FFN
    ↓
Add
```

**Why Pre-Norm For MarkGPT?**
- **Depth**: Pre-norm trains deeper models more stably
- **Gradient flow**: Residuals bypass norm (cleaner gradients)
- **Learning rate**: Can use higher learning rates
- **Convergence**: Faster initial convergence

```python
class TransformerBlockPreNorm(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
    
    def forward(self, x):
        # Pre-norm: normalize, then apply module
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

---

## RMSNorm (Preferred in Modern LLMs)

Root Mean Square Normalization: simplification that often works better.

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$

Where:
$$\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}$$

**Key differences from LayerNorm**:
- No centering (no subtraction of mean)
- Only scale, no separate offset
- Faster to compute
- Works as well or better empirically

```python
class RMSNorm(nn.Module):
    """RMS normalization (used in Llama, Mistral)."""
    
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        rms = torch.sqrt(
            (x ** 2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (x / rms) * self.gamma

# Equivalently with numerically stable version:
def rmsnorm_stable(x, gamma, eps=1e-5):
    # norm = x / sqrt(mean(x^2) + eps)
    scale = 1.0 / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return x * scale * gamma
```

### Why RMSNorm Over LayerNorm?

| Property | LayerNorm | RMSNorm |
|----------|-----------|---------|
| Mean centering | Yes (subtract μ) | No |
| Scale factor | Yes (divide by σ) | Yes (divide by RMS) |
| Parameters | γ, β (2 per dim) | γ only (1 per dim) |
| Computing speed | Slower | ~2x faster |
| Empirical performance | Good | Slightly better |
| Memory usage | More (store mean) | Less |

---

## GroupNorm (For Low-Batch Scenarios)

When batch size is very small (< 2), LayerNorm/RMSNorm can be unstable.

GroupNorm divides channels into groups:

$$\text{GroupNorm}(x) = \frac{x - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}} \odot \gamma + \beta$$

Where group $g$ has subset of dimensions.

```python
# Standard PyTorch
group_norm = nn.GroupNorm(
    num_groups=32,      # Divide 768 → 32 groups of 24 channels
    num_channels=768,
    eps=1e-5
)

# Usage: similar to LayerNorm
# x: (batch, channels, height, width) for 4D
# or (batch, seq_len, channels) for 3D
```

**When to use GroupNorm?**
- Batch size < 2 (inference, very constrained)
- Batch size varies significantly
- Recommended: Stick with LayerNorm/RMSNorm for LLMs

---

## Batch Normalization (NOT Recommended for LLMs)

```python
# DON'T DO THIS FOR TRANSFORMERS
batch_norm = nn.BatchNorm1d(d_model)  # ← Problematic

# Issues:
# 1. Different batch statistics at train vs inference
# 2. Sequence length changes → statistics change
# 3. Breaks on inference with batch_size=1
# 4. Coupling between examples (unfair for individual processing)
```

**Exception**: Early ViT models used BatchNorm in patch embedding layer:
```python
# Some ViT architectures (less common now)
class ViTPatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.norm = nn.BatchNorm1d(768)  # Rare in LLMs
```

---

## Normalization Placement in MarkGPT

**Recommended Architecture**:

```python
class MarkGPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        # Pre-norm for stability
        self.norm_attn = nn.LayerNorm(d_model)  # or RMSNorm
        self.attn = MultiHeadAttention(d_model, num_heads)
        
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
    
    def forward(self, x, mask=None):
        # Attention block
        x = x + self.attn(self.norm_attn(x), mask=mask)
        
        # FFN block
        x = x + self.ffn(self.norm_ffn(x))
        
        return x

class MarkGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            MarkGPTBlock(config.d_model, config.num_heads, config.d_ff)
            for _ in range(config.num_layers)
        ])
        # Final norm before output
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
```

---

## Normalization Stability Tricks

### Issue: Vanishing Gradients in Deep Pre-Norm

```python
# Problem: Early layers get small gradients in pre-norm
# Solution: Increase residual connection strength

class StablePreNormBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        
        # Scale residuals to improve gradient flow
        self.residual_scale = 1.0 / math.sqrt(2)  # For 2 residuals
    
    def forward(self, x):
        x = x + self.residual_scale * self.attn(self.norm1(x))
        x = x + self.residual_scale * self.ffn(self.norm2(x))
        return x
```

### Issue: Numerical Precision in Very Deep Models

```python
# Use mixed precision: compute norm in float32, apply in float16

class FP32LayerNorm(nn.LayerNorm):
    """LayerNorm that maintains fp32 precision."""
    
    def forward(self, x):
        if x.dtype != torch.float32:
            x_fp32 = x.float()
            output = super().forward(x_fp32)
            return output.type_as(x)
        return super().forward(x)
```

---

## Normalization Validation Checklist

For debugging normalization issues:

```python
def diagnose_normalization(model, x):
    """Check normalization health."""
    
    # 1. Check norm statistics
    for name, param in model.named_parameters():
        if 'gamma' in name or 'weight' in name and 'norm' in name:
            print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}")
    
    # 2. Check activation range
    with torch.no_grad():
        for i, block in enumerate(model.blocks):
            x = block(x)
            mean = x.mean().item()
            std = x.std().item()
            max_val = x.max().item()
            if std > 100 or std < 0.01:
                print(f"⚠️ Block {i}: activation scale unusual (mean={mean:.2f}, std={std:.2f})")
    
    # 3. Check for NaN
    if torch.isnan(x).any():
        print("❌ NaN detected!")
    
    return x
```

---

**Normalization Techniques v1.0**
**Last Updated**: 2024
