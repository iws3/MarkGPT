# Optimizers for Training LLMs

## SGD with Momentum (Baseline)

### Standard SGD

$$\theta_t = \theta_{t-1} - \eta \nabla L(\theta_{t-1})$$

Updates parameters directly by gradients.

**Issues for LLM training**:
- Oscillates around minima (slow convergence)
- Doesn't adapt learning rate per parameter
- Stuck at saddle points

### SGD with Momentum

$$v_t = \beta v_{t-1} + (1 - \beta) \nabla L(\theta_t)$$
$$\theta_t = \theta_{t-1} - \eta v_t$$

Accumulates past gradients (momentum), smooths updates.

```python
# PyTorch implementation
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,  # γ in some literature
    dampening=0,   # Typical 0 for full momentum
    weight_decay=1e-4
)

# Training loop
for epoch in range(epochs):
    for batch in loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()  # Applies momentum internally
```

**When to use**:
- Baseline for small models
- Not recommended for LLMs (too slow)

---

## Adam (Most Common)

### Adaptive Moment Estimation

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t)$$  (First moment, mean)
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(\theta_t))^2$$  (Second moment, variance)

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$  (Bias correction)
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Parameters**:
- $\beta_1$ = 0.9 (momentum decay)
- $\beta_2$ = 0.999 (variance decay)
- $\epsilon$ = 1e-8 (numerical stability)
- $\eta$ = learning rate (typically 1e-3 to 1e-5 for LLMs)

### Adam Manual Implementation

```python
class AdamExplained:
    """Manual Adam for understanding."""
    
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Initialize moment estimates
        self.m = [{} for _ in self.params]
        self.v = [{} for _ in self.params]
    
    def step(self):
        """Single optimization step."""
        self.t += 1
        
        for group, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            g = param.grad.data
            
            # Exponential moving averages
            if param not in self.m[group]:
                self.m[group][param] = torch.zeros_like(g)
                self.v[group][param] = torch.zeros_like(g)
            
            m = self.m[group][param]
            v = self.v[group][param]
            
            # Update biased moments
            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data.add_(m_hat / (v_hat.sqrt() + self.eps), alpha=-self.lr)

# Standard PyTorch
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # L2 regularization
)
```

### Adam Variants for LLMs

**AdamW** (Decoupled Weight Decay):
```python
# AdamW separates weight decay from gradient updates
# Better for LLM training than Adam with weight_decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # Truly decoupled
)
```

**Why AdamW?**
- L2 regularization in standard Adam couples with learning rate
- AdamW applies weight decay directly
- More stable training for large models
- Recommended default for MarkGPT

---

## Lion (Recent, Promising)

Evolved Sign Momentum (Li et al., 2023)

$$\theta_t = \theta_{t-1} - \eta \text{sign}(m_t)$$

Where:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t)$$

**Key insight**: Use gradient *direction*, not magnitude.

```python
# Manual Lion
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                
                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)
                
                m = state['momentum']
                beta1, beta2 = group['betas']
                
                # Update momentum
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update parameter using sign of momentum
                p.data.add_(torch.sign(m), alpha=-group['lr'])
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
```

### Lion vs AdamW

| Property | AdamW | Lion |
|----------|-------|------|
| Convergence speed | Fast initial | Similar |
| Memory per param | 2× (m, v) | 1× (m) |
| Stability | High | High |
| Learning rate magnitude | Sensitive (1e-4) | Robust (1e-3) |
| Compute cost | Low | Very low |
| Empirical performance | State-of-art | Slightly better |

**Verdict**: Lion is promising but add expensive. Use AdamW for safety.

---

## Scheduler Strategies

### Learning Rate Schedules
   
#### Constant (Baseline)
```python
# No scheduler, constant lr
optimizer = AdamW(params, lr=1e-4)
```

#### Linear Warmup + Cosine Decay (Recommended)

$$\text{lr}_t = \begin{cases}
\frac{t \cdot \text{lr}_\max}{t_\text{warmup}} & \text{if } t < t_\text{warmup} \\
\frac{\text{lr}_\max}{2} \left(1 + \cos\left(\pi \frac{t - t_\text{warmup}}{t_\text{total} - t_\text{warmup}}\right)\right) & \text{otherwise}
\end{cases}$$

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# Combining warmup + cosine
def get_scheduler(optimizer, num_steps, warmup_steps=1000):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(num_steps - step) / float(max(1, num_steps - warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)

# Usage
scheduler = get_scheduler(optimizer, total_steps=100000, warmup_steps=2000)

for step, batch in enumerate(loader):
    loss = model(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update lr
```

#### Polynomial Decay

$$\text{lr}_t = \text{lr}_0 \cdot (1 - t/t_{\max})^p$$

Usually $p = 1$ (linear) or $p = 2$ (quadratic).

---

## Gradient Clipping & Normalization

### Global Gradient Clipping

```python
# Clip gradient norm
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0  # Common value
)

# Training loop
for step, batch in enumerate(loader):
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    
    # Clip before step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    scheduler.step()
```

### Gradient Value Clipping (Per-element)

```python
for param in model.parameters():
    if param.grad is not None:
        param.grad.data.clamp_(-1.0, 1.0)
```

### When to Use Clipping?

- **Always** for LLMs (prevents loss spikes)
- **Especially** with high learning rates
- **Monitor**: If clipping frequently (loss > threshold), lower lr

---

## Optimizer Configuration for MarkGPT

### Recommended Settings

```python
# MarkGPT training configuration
config_optimizer = {
    "optimizer_type": "AdamW",
    "learning_rate": 5e-5,  # For 200M+ models
    "weight_decay": 0.01,
    "betas": (0.9, 0.95),   # More conservative than default (0.999)
    "eps": 1e-8,
    "gradient_clip_norm": 1.0,
    "warmup_steps": 2000,   # Roughly 1% of total steps
    "total_steps": 200000,
    "lr_scheduler": "cosine",
}

# Implementation
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config_optimizer["learning_rate"],
    betas=config_optimizer["betas"],
    weight_decay=config_optimizer["weight_decay"]
)

scheduler = LinearWarmupCosineScheduler(
    optimizer,
    warmup_steps=config_optimizer["warmup_steps"],
    total_steps=config_optimizer["total_steps"]
)
```

### Size-Specific Adjustments

| Model Size | LR | Batch Size | Warmup Steps |
|-----------|----|-----------| -------------|
| Nano (70M) | 1e-4 | 128-256 | 500 |
| Small (200M) | 5e-5 | 64-128 | 1000 |
| Base (500M) | 2e-5 | 32-64 | 2000 |

---

## Debugging Optimizer Issues

```python
def debug_training():
    # 1. Check gradient magnitudes
    grad_norms = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    print(f"Gradient norm: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}")
    
    # 2. Check learning rate
    for param_group in optimizer.param_groups:
        print(f"Current LR: {param_group['lr']:.6f}")
    
    # 3. Monitor loss
    if loss.isnan():
        print("❌ NaN loss detected! Reduce learning rate.")
    elif loss > 2 * prev_loss:
        print("⚠️ Loss spike! Check gradients or reduce LR.")
    
    # 4. Check optimizer state
    for param_group in optimizer.param_groups:
        print(f"Iterations: {optimizer.state[param_group['params'][0]]['step']}")
```

---

**Optimizers & Schedulers v1.0**
**Last Updated**: 2024
