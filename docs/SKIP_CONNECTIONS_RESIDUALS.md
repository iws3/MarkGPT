# Residual Connections & Skip Connections

## Residual Networks (ResNets) Motivation

### The Degradation Problem

Deep networks are hard to train because:

1. **Vanishing gradients**: $\frac{dL}{d\theta} = \frac{dL}{d\text{out}} \cdot \prod_{i} \frac{d\text{layer}_i}{d\text{layer}_{i-1}}$
   - Each multiplication is typically << 1
   - Exponential decay: $(0.9)^{100} \approx 0.0003$

2. **Saturation zones**: ReLU activation plateaus
   - Gradients become 0
   - No learning signal

### ResNet Solution: Skip Connection

$$\mathbf{y} = \mathcal{H}(\mathbf{x}) + \mathbf{x}$$

Learn the *residual* (difference), not the full transformation.

```
Standard:  x → [Conv] → [ReLU] → [Conv] → y
                         (learn y)

ResNet:    x → [Conv] → [ReLU] → [Conv] → (+) → y
           └────────────────────────────────┘
                  (learn y - x)
```

---

## Gradient Flow in Skip Connections

### Without Skip (Standard):
$$\frac{dL}{d\mathbf{x}} = \frac{dL}{d\mathbf{y}} \cdot \frac{d\mathcal{H}}{d\mathbf{x}}$$

If $|\frac{d\mathcal{H}}{d\mathbf{x}}| < 1$, gradient vanishes.

### With Skip:
$$\frac{dL}{d\mathbf{x}} = \frac{dL}{d\mathbf{y}} \cdot \left(\frac{d\mathcal{H}}{d\mathbf{x}} + \mathbf{I}\right)$$

The identity term ($\mathbf{I}$) ensures gradient magnitude ≥ 1:
- Gradient can always flow through the skip
- Residual learning can be small or large

---

## Transformers: Multi-Residual Structure

MarkGPT has skip connections everywhere:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
    
    def forward(self, x):
        # Skip connection 1: Attention
        x = x + self.attn(self.norm1(x))
        
        # Skip connection 2: FFN
        x = x + self.ffn(self.norm2(x))
        
        return x
```

### Gradient Path Analysis

Input signal can take 2^L paths through L layers:

```
Layer 1: Skip? [Y/N]  (2 paths)
    ↓
Layer 2: Skip? [Y/N]  (4 paths total)
    ↓
...
Layer L: Skip? [Y/N]  (2^L paths total)
```

This ensemble effect (multiple gradient paths) prevents vanishing gradients.

---

## Scaling Residuals

### Issue: Exploding Activations

With many skip connections, activations can grow:

```
x = x + residual_1
x = x + residual_2
x = x + residual_3  ← Growing too fast?
```

### Solution 1: Scale Residuals

```python
class ScaledResidualBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.scale = 1.0 / math.sqrt(2)  # For 2 residuals
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
    
    def forward(self, x):
        x = x + self.scale * self.attn(self.norm1(x))
        x = x + self.scale * self.ffn(self.norm2(x))
        return x
```

### Solution 2: Increase Initialization Variance

```python
def scaled_init(module, scale=1.0):
    """Initialize with larger variance for scaled residuals."""
    if isinstance(module, nn.Linear):
        fan_in = module.weight.shape[1]
        std = scale * math.sqrt(2.0 / fan_in)
        nn.init.normal_(module.weight, std=std)
```

---

## Types of Skip Connections

### 1. Identity Skip (MarkGPT-Style)

```python
# Identity: Dimensions match exactly
x = x + f(x)  # x and f(x) must have same shape
```

When used:
- Input and output have same dimension
- Typical in transformer blocks
- Simplest form

### 2. Projection Skip (Dimension Mismatch)

```python
class ProjectionSkip(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x, fx):
        return self.proj(x) + fx

# Example: Changing channels
class ResConvBlock(nn.Module):
    def forward(self, x):
        if x.shape[1] != self.out_channels:
            x = self.proj(x)
        x = x + self.conv1(x)
        return x
```

### 3. Dense Connections (DenseNet-Style)

```python
# Skip to ALL previous layers (research interest, not standard)
class DenseBlock(nn.Module):
    def forward(self, inputs):
        # inputs is list of all previous layer outputs
        outputs = [inputs[-1]]  # Current layer
        
        for layer in self.layers:
            x = layer(torch.cat(outputs, dim=1))
            outputs.append(x)
        
        return torch.cat(outputs, dim=1)
```

---

## Analysis of Skip Connections in MarkGPT

### Forward Pass

```python
def markgpt_forward(x, model):
    """Trace signal through layer."""
    
    # Embedding layer (no skip)
    x = model.embedding(x)  # (batch, seq_len, d_model)
    
    # Transformer blocks (2 skips each)
    for block in model.blocks:
        # Skip 1
        attn_out = block.attn(block.norm1(x))
        x = x + attn_out
        
        # Skip 2
        ffn_out = block.ffn(block.norm2(x))
        x = x + ffn_out
    
    # Output projection
    logits = model.lm_head(model.final_norm(x))
    
    return logits
```

### Backward Pass (Gradient Flow)

```python
def trace_gradients(loss, model):
    """Analyze gradient flow through skips."""
    
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: ∇L = {grad_norm:.6f}")
    
    # Check skip gradients specifically
    for i, block in enumerate(model.blocks):
        # Without skip, gradients would diminish
        # With skip, each component gets full gradient signal
```

---

## Skip Connection Variants

### Highway Networks (Gating Skip)

Learned gate for skip:

$$\mathbf{y} = \mathbf{T} \odot \mathcal{H}(\mathbf{x}) + (1 - \mathbf{T}) \odot \mathbf{x}$$

Where $\mathbf{T} = \sigma(\mathbf{W}_T \mathbf{x} + \mathbf{b}_T)$ is a learned gate.

```python
class HighwayLayer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.normal = nn.Linear(in_dim, in_dim)
        self.gate = nn.Linear(in_dim, in_dim)
    
    def forward(self, x):
        h = F.relu(self.normal(x))
        t = torch.sigmoid(self.gate(x))
        return t * h + (1 - t) * x
```

**When to use**: More complex conditional processing (not standard in MarkGPT).

### Gated Residual Units (GRU-style)

```python
class GatedResidual(nn.Module):
    def forward(self, x):
        h = self.layer(x)
        # Dynamically blend
        return h + self.gate(x) * (self.residual(x) - h)
```

---

## Common Issues & Debugging

### Issue 1: Growing Activations

**Symptom**: max(|x|) increases through layers → NaN

**Solution**:
```python
# Add scaling
x = x + 0.5 * self.attn(self.norm1(x))
x = x + 0.5 * self.ffn(self.norm2(x))

# Or monitor
with torch.no_grad():
    for i, block in enumerate(model.blocks):
        x = block(x)
        act_scale = x.abs().max().item()
        print(f"Block {i}: max|x| = {act_scale:.2f}")
```

### Issue 2: Dead Skip Connections

**Symptom**: Skip always dominates (residual branches learn nothing)

**Diagnosis**:
```python
def diagnose_skip_dominance(model, x):
    for i, block in enumerate(model.blocks):
        residual = block.ffn(block.norm2(x)) - block.attn(block.norm1(x))
        skip = x
        
        ratio = residual.abs().mean() / skip.abs().mean()
        if ratio < 0.01:
            print(f"⚠️ Block {i}: Skip dominates (ratio={ratio:.4f})")

```

**Fix**: Adjust scale or initialization.

---

## Skip Connection Best Practices for MarkGPT

```python
class MarkGPTBlockImproved(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        # Pre-norm (proven better)
        self.norm1 = RMSNorm(d_model)  # Instead of LayerNorm
        self.attn = MultiHeadAttention(d_model, num_heads)
        
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)  # Modern activation
        
        # Residual scaling (optional, helps deep models)
        self.alpha = 1.0  # Can be tuned
    
    def forward(self, x, mask=None):
        # Skip 1: Attention
        x = x + self.alpha * self.attn(self.norm1(x), mask=mask)
        
        # Skip 2: FFN
        x = x + self.alpha * self.ffn(self.norm2(x))
        
        return x

    # Initialization for deep models
    def init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.orthogonal_(param)
```

---

**Skip Connections & Residuals v1.0**
**Last Updated**: 2024
