# Activation Functions in Modern LLMs

## Choosing Activation Functions

### Original Transformer: ReLU

```
f(x) = max(0, x)

Issues:
- "Dead neurons": Negative values map to 0, gradients stuck
- Not smooth at x=0
- Poor initialization sensitivity
```

Modern LLMs use alternatives:

---

## GELU (Gaussian Error Linear Unit)

### Intuition

Approximates the CDF of a Gaussian:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

Where $\Phi(x)$ is the cumulative distribution function of standard normal.

**Approximation** (used in practice):
$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$

Or more commonly (PyTorch default):
$$\text{GELU}(x) \approx x \cdot \Phi(x) = \frac{x}{2}\left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

### Properties

- Smooth everywhere (both ReLU and tanh are not)
- Probabilistic interpretation
- No dead neurons
- Gradient flows well through transitions

### Implementation

```python
import torch
import torch.nn.functional as F

# Three variants (increasing accuracy)
def gelu_approx(x):
    """Simple approximation."""
    return 0.5 * x * (1.0 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / 3.14159)) * 
        (x + 0.044715 * x**3)
    ))

def gelu_error_function(x):
    """Using error function."""
    return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

def gelu_fast(x):
    """PyTorch built-in (fastest)."""
    return F.gelu(x, approximate='tanh')

def gelu_accurate(x):
    """Most accurate."""
    return F.gelu(x, approximate='none')

# In a model
class TransformerFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))
```

### Why GELU for Transformers?

- **Smoothness**: Gradients flow smoothly through network
- **Density**: Denser than ReLU (activates more neurons)
- **Empirically better**: Outperforms ReLU on BERT, GPT
- **Probabilistic**: Aligns with Gaussian assumptions in initialization

---

## SwiGLU (Developed for Llama)

### Formula

$$\text{SwiGLU}(x, W, V, W_2) = (\sigma(xW) \otimes xV)W_2$$

Where $\otimes$ is element-wise multiplication and $\sigma$ is sigmoid.

Breaking it down:
1. $xW$ → pass through sigmoid (gating)
2. $xV$ → linear projection
3. Multiply the two (gated linear unit)
4. $W_2$ → project back to d_model

### Why It Works

- **Gating mechanism**: Learn which features to pass
- **Skip-free**: Unlike residuals, information must be gated
- **Efficient**: Doesn't require explicit skip connections
- **Llama secret**: Used in all Llama models

### Implementation

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Expand to d_ff, then gate with another d_ff
        self.w = nn.Linear(d_model, d_ff)
        self.v = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        gate = torch.sigmoid(self.w(x))
        value = self.v(x)
        output = gate * value
        return self.w2(output)

# Usage in FFN
class LlamaFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_ff)
    
    def forward(self, x):
        return self.swiglu(x)
```

### Memory vs Accuracy Trade-off

Without the gating branch:
```
Standard FFN: d_model -> d_ff -> d_model
              4 * d_model params (if d_ff = 4*d_model)

With gating: d_model -> d_ff + d_ff -> d_model
             8 * d_model params!
```

**Solution**: Reduce d_ff when using SwiGLU:
- Standard FFN: d_model × 4 × d_model = 4d²
- SwiGLU: d_model × (2d_ff/3) × d_model = ~2.67d²
- **Same params, better performance!**

---

## GeGLU (Learnable Gating)

Similar to SwiGLU but uses GELU instead of sigmoid:

$$\text{GeGLU}(x, W, V, W_2) = (\text{GELU}(xW) \otimes xV)W_2$$

```python
class GeGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w = nn.Linear(d_model, d_ff)
        self.v = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        gate = self.gelu(self.w(x))
        value = self.v(x)
        return self.w2(gate * value)
```

**Comparison**:
- SwiGLU: Sharper gates (0 or 1 range from sigmoid)
- GeGLU: Smoother gates (full range from GELU)
- Performance: Similar, domain-dependent

---

## Comparison Table

| Activation | Props | Best For | Computational Cost | Dead Neurons |
|-----------|-------|----------|-------------------|--------------|
| ReLU | Simple, fast | Old models | Very low | Yes (problematic) |
| GELU | Smooth, probabilistic | BERT, GPT-2 | Low | No |
| SwiGLU | Gated, efficient | Llama-series | Medium (mixed) | No |
| GeGLU | Gated, smooth | T5, newer | Medium | No |

---

## Activation Function Selection Guide

### For Pretraining (MarkGPT-style)

```python
# Recommended: GELU
model = Transformer(
    activation="gelu",  # Default in HF transformers
    # ...
)

# Alternative: SwiGLU (if you have the compute)
model = Transformer(
    activation="swiglu",
    ffn_dim_multiplier=2/3,  # Reduce to keep params similar
)
```

### For Fine-tuning

```python
# Match pretraining activation
# Don't mix ReLU pretrained with GELU fine-tuned
# (gradient flow can be problematic)
```

### For Distillation

```python
# Use same activation as teacher model
# Different activation = harder alignment
```

---

## Advanced: Learnable Activation Functions

```python
class LearnableActivation(nn.Module):
    """Dynamically choose activation per layer."""
    def __init__(self):
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        # Interpolate between GELU and SwiGLU
        gelu_out = F.gelu(x)
        # (Simplified; real version would apply both)
        return self.alpha * gelu_out
```

Not commonly used in production (adds overhead), but interesting for research.

---

## ReLU Variants (Historical Reference)

For completeness (not recommended for new models):

```python
# Leaky ReLU: Allows small negative gradient
F.leaky_relu(x, negative_slope=0.01)

# ELU: Smooth negative region
F.elu(x, alpha=1.0)

# SELU: Self-normalizing for specific architectures
F.selu(x)
```

---

## Practical Tips

### Activation Function Initialization

Different activations have different initialization preferences:

```python
# For GELU (variance preservation)
nn.init.normal_(layer.weight, std=math.sqrt(2 / (in_features + out_features)))

# For SwiGLU with larger fan-out
nn.init.normal_(layer.weight, std=math.sqrt(1 / in_features))
```

### Mixed Precision Considerations

```python
# Some activations are less stable in fp16
# Solution: Use fp32 for critical activations

class MixedPrecisionFFN(nn.Module):
    def forward(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.w1(x)
            x = F.gelu(x)  # Keep GELU in fp32 if training with bf16
        x = self.w2(x)
        return x
```

### Profiling

```python
def profile_activation(activation_fn, input_shape=(32, 128, 768)):
    x = torch.randn(input_shape, device='cuda')
    
    import time
    start = time.perf_counter()
    for _ in range(100):
        _ = activation_fn(x)
    end = time.perf_counter()
    
    print(f"{activation_fn.__class__.__name__}: {(end-start)*10:.2f}ms")

# Results (typical):
# - GELU:    2-3ms
# - SwiGLU:  3-4ms (2 projections)
# - ReLU:    0.5ms (reference)
```

---

**Activation Function Reference v1.0**
**Last Updated**: 2024
