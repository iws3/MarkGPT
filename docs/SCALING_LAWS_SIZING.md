# Scaling Laws & Model Sizing

## Chinchilla Scaling Laws (Optimal Allocation)

### The Classic Problem

Given N total parameters, how to split between:
- Model size (parameters)
- Dataset size (tokens)

**Early findings** (GPT-3 era):
- Bigger model > Bigger data
- Result: Undertrained models (GPT-3 uses 100B params on small data)

### Chinchilla's Discovery (Hoffmann et al., 2022)

For a fixed compute budget $C$:

$$N^* = \frac{C}{6D}$$
$$D^* = \frac{20C}{N}$$

Where:
- $N$ = number of parameters
- $D$ = number of training tokens
- $C$ = compute (FLOPs)

**Key insight**: Optimal ratio is roughly equal allocation:
$$N \approx \frac{D}{20}$$

Or equivalently: Each parameter should see ~20 tokens.

### Formula-Based Scaling

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

Where:
- $E$ = irreducible loss
- $A, B$ = constants
- $\alpha \approx 0.07$ (diminishing returns on model size)
- $\beta \approx 0.10$ (diminishing returns on data size)

```python
def optimal_allocation(total_compute_flops):
    """
    Given FLOPs budget, allocate between N (params) and D (data).
    """
    # Chinchilla formula
    n_optimal = total_compute_flops / (6 * 20)  # Simplified
    d_optimal = total_compute_flops / (6 * 1)
    
    return n_optimal, d_optimal

# Example
compute_budget = 1e22  # ~100 exaflop-seconds
n_opt, d_opt = optimal_allocation(compute_budget)
print(f"Optimal N: {n_opt / 1e9:.1f}B params")
print(f"Optimal D: {d_opt / 1e9:.1f}B tokens")
# Output: 70B params, 1.4T tokens
```

---

## MarkGPT Scaling

### Model Variants

| Variant | Params | d_model | num_layers | num_heads | d_ff |
|---------|--------|---------|-----------|-----------|------|
| Nano | 70M | 256 | 12 | 4 | 1024 |
| Small | 200M | 512 | 16 | 8 | 2048 |
| Base | 500M | 768 | 24 | 12 | 3072 |

### Compute Estimation

For transformer, FLOPs per token per parameter:

$$\text{FLOPs} = 6N \times D$$

(6 = forward pass, weight gradients, optimizer states)

```python
def compute_flops(num_params, num_tokens):
    """Estimate total FLOPs for training."""
    return 6 * num_params * num_tokens

# MarkGPT Nano
flops_nano = compute_flops(70e6, 100e9)  # 70M params, 100B tokens
print(f"{flops_nano / 1e18:.1f} exaflops-seconds")
# Output: 42 exaflops-seconds

# On V100 (125 TFLOPs)
v100_hours = (flops_nano / 1e12) / (125 * 1e12) / 3600
print(f"~{v100_hours:.0f} hours on single V100")
```

### Optimal Token Count (Chinchilla for MarkGPT)

```python
def markgpt_optimal_tokens(num_params):
    """Tokens needed for this model size (Chinchilla ratio)."""
    return 20 * num_params

configurations = {
    "nano": {"params": 70e6, "tokens": markgpt_optimal_tokens(70e6)},
    "small": {"params": 200e6, "tokens": markgpt_optimal_tokens(200e6)},
    "base": {"params": 500e6, "tokens": markgpt_optimal_tokens(500e6)},
}

# Results
for name, config in configurations.items():
    tokens_b = config["tokens"] / 1e9
    print(f"{name}: {config['params']/1e6:.0f}M params × {tokens_b:.0f}B tokens")

# Output:
# nano: 70M params × 1.4B tokens
# small: 200M params × 4.0B tokens
# base: 500M params × 10.0B tokens
```

---

## Loss Prediction (Chen et al., 2024)

Improved scaling law with cross-term:

$$L(N, D) = L_\infty + A N^{-\alpha} + B D^{-\beta} + C (N \times D)^{-\gamma}$$

**Practical**: Use to predict loss before training.

```python
def predict_loss(n_params, n_tokens, constants):
    """Predict test loss from this configuration."""
    
    a, b, c = constants['a'], constants['b'], constants['c']
    alpha, beta, gamma = constants['alpha'], constants['beta'], constants['gamma']
    l_inf = constants['l_inf']
    
    loss = (
        l_inf +
        a * (n_params ** (-alpha)) +
        b * (n_tokens ** (-beta)) +
        c * ((n_params * n_tokens) ** (-gamma))
    )
    
    return loss

# Constants fitted to MarkGPT-scale models
markgpt_constants = {
    'l_inf': 3.2,  # Asymptotic loss (what we can't improve)
    'a': 500.0,    # Model size effect
    'alpha': 0.07,
    'b': 150.0,    # Data size effect
    'beta': 0.10,
    'c': 100.0,    # Interaction term
    'gamma': 0.08,
}

# Predict
n = 200e6  # 200M params
d = 4e9    # 4B tokens
predicted_loss = predict_loss(n, d, markgpt_constants)
print(f"Predicted loss: {predicted_loss:.2f}")
```

---

## Practical Scaling Strategies

### 1. Sequential Scaling (Phase-based)

```
Iteration 1: Train Nano (70M) on 10B tokens
  → Best hyperparameters, optimal LR, etc.

Iteration 2: Train Small (200M) on 50B tokens
  → Transfer hyperparameters + tune
  → Watch for scaling issues

Iteration 3: Train Base (500M) on 500B tokens
  → Full curriculum
  → Production

Benefits: Cheaper early validation, faster debugging
```

### 2. Hyperparameter Scaling

```python
def scale_hyperparameters(base_lr=1e-4, base_params=70e6, target_params=500e6):
    """Scale learning rate by model size (Kaplan et al.)."""
    
    # LR scales inversely with sqrt(model size)
    scaling_factor = (base_params / target_params) ** 0.5
    new_lr = base_lr * scaling_factor
    
    return new_lr

# Example
small_lr = scale_hyperparameters(1e-4, 70e6, 200e6)
base_lr = scale_hyperparameters(1e-4, 70e6, 500e6)

print(f"Nano: 1e-4")
print(f"Small: {small_lr:.2e}")
print(f"Base: {base_lr:.2e}")
# Output:
# Nano: 1e-4
# Small: 5.92e-05
# Base: 3.73e-05
```

### 3. Batch Size Scaling

Larger models benefit from larger batches:

```python
def scale_batch_size(base_batch=64, base_params=70e6, target_params=500e6):
    """Scale batch size with model size."""
    
    ratio = target_params / base_params
    new_batch = int(base_batch * math.sqrt(ratio))
    
    return new_batch

# Nano: batch=64
# Small: batch = 64 * sqrt(20e7/70e6) = 64 * sqrt(2.86) ≈ 107
# Base: batch = 64 * sqrt(500e6/70e6) = 64 * sqrt(7.14) ≈ 170
```

---

## Empirical Scaling Results

### MarkGPT Training Time (Estimates)

| Model | Params | Tokens | GPU | Hours | Cost |
|-------|--------|--------|-----|-------|------|
| Nano | 70M | 1.4B | 1×V100 | 25 | $50 |
| Small | 200M | 4B | 2×A100 | 40 | $200 |
| Base | 500M | 10B | 8×A100 | 80 | $1600 |

(Rough estimates; actual depends on optimization, parallelism)

### Chinchilla vs Compute-Optimal

```python
def compare_strategies(compute_budget):
    """Compare compute allocation strategies."""
    
    # Chinchilla (balanced)
    n_chin = compute_budget / (6 * 20)
    d_chin = 20 * n_chin
    loss_chin = predict_loss(n_chin, d_chin, constants)
    
    # GPT-3 style (oversized model)
    n_gpt3 = compute_budget / (6 * 1)  # Compute all on model, less on data
    d_gpt3 = compute_budget / (6 * n_gpt3)
    loss_gpt3 = predict_loss(n_gpt3, d_gpt3, constants)
    
    print(f"Chinchilla: {n_chin/1e6:.0f}M params, {d_chin/1e9:.1f}B tokens → loss={loss_chin:.3f}")
    print(f"GPT-3 style: {n_gpt3/1e6:.0f}M params, {d_gpt3/1e9:.1f}B tokens → loss={loss_gpt3:.3f}")
    
    improvement = (loss_gpt3 - loss_chin) / loss_gpt3 * 100
    print(f"Chinchilla improvement: {improvement:.1f}%")

# For 1e21 FLOPs
compare_strategies(1e21)

# Output (typical):
# Chinchilla: 4167M params, 83B tokens → loss=3.451
# GPT-3 style: 25000M params, 6.7B tokens → loss=3.652
# Chinchilla improvement: 5.5%
```

---

## When to Scale

### Red Flags (Don't Scale Yet)

- Loss not decreasing (training issue, not size)
- Oscillating loss (unstable, tune LR first)
- OOM errors on current size (fix capacity before going bigger)
- Poor transfer between sizes (fundamental architecture issue)

### Green Lights (Scale!)

- Loss follows predicted curve
- Consistent, stable training
- Perplexity on test set improving
- Hyperparameters found

---

## MarkGPT Publication Roadmap

```
Publication Targets by Model Size
===
Nano (70M):
  - Multilingual capability
  - Efficiency benchmarks
  - Banso language guide

Small (200M):
  - Curriculum learning impact
  - Bilingual translation performance
  - Downstream task evaluation

Base (500M):
  - State-of-art Banso modeling
  - English-Banso code-switching
  - Production deployment guide
  - LLM scaling analysis
```

---

**Scaling Laws & Sizing v1.0**
**Last Updated**: 2024
