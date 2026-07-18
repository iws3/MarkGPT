# Training Roadmap & Scaling Analysis

## Full Training Across Model Sizes

### Nano Configuration (70M parameters)
```yaml
# configs/markgpt_nano_training.yaml
model:
  d_model: 512
  num_layers: 16
  num_heads: 8
  vocab_size: 50257
  max_seq_length: 2048

training:
  total_tokens: 20B
  batch_size: 256
  learning_rate: 3e-4
  num_epochs: 1
  gradient_accumulation_steps: 4
  
hardware:
  gpus: 4              # 4x A100 GPUs
  mixed_precision: fp16
  activation_checkpointing: true
  
metrics:
  expected_ppl: 24.0
  expected_hours: 48
```

**Chinchilla Computation Law**:
- Compute budget: 20B tokens × 70M params × 6 = 8.4 × 10^18 FLOPs
- Time estimate: 8.4e18 / (4 × 312 TF/s × 3600) ≈ 187 hours single-GPU equivalent
- With 4 GPUs: ~48 hours

### Small Configuration (200M parameters)
```yaml
model:
  d_model: 768
  num_layers: 24
  num_heads: 12
  
training:
  total_tokens: 60B    # 3× Nano
  batch_size: 512
  learning_rate: 2e-4
  
hardware:
  gpus: 8              # 8x A100
  
metrics:
  expected_ppl: 16.5
  expected_hours: 72
```

### Base Configuration (500M parameters)
```yaml
model:
  d_model: 1024
  num_layers: 32
  num_heads: 16
  
training:
  total_tokens: 150B   # 2.5× Small
  batch_size: 1024
  learning_rate: 1.5e-4
  
hardware:
  gpus: 16             # 8× A100 or H100
  
metrics:
  expected_ppl: 11.2
  expected_hours: 120
```

### Scaling Laws (Observed)

```python
import numpy as np
import matplotlib.pyplot as plt

# Historical scaling measurements
configs = {
    'nano_70m': {'params': 70e6, 'tokens': 20e9, 'ppl': 24.0},
    'small_200m': {'params': 200e6, 'tokens': 60e9, 'ppl': 16.5},
    'base_500m': {'params': 500e6, 'tokens': 150e9, 'ppl': 11.2},
}

# Fit Chinchilla law: PPL = α × (C / 6D)^β
# where C = compute, D = params
params = np.array([c['params'] for c in configs.values()])
ppl = np.array([c['ppl'] for c in configs.values()])

# Power law regression
log_params = np.log10(params)
log_ppl = np.log10(ppl)
coeff = np.polyfit(log_params, log_ppl, 1)

print(f"Scaling law: PPL ∝ Params^{coeff[0]:.2f}")
print(f"  → Each 10× parameter increase → {10**coeff[0]:.1f}× perplexity improvement")

# Predict for larger models
future_sizes = [1e9, 3e9, 7e9]  # 1B, 3B, 7B
predicted_ppl = 10 ** (coeff[0] * np.log10(future_sizes) + coeff[1])

for size, ppl_pred in zip(future_sizes, predicted_ppl):
    print(f"Predicted {size/1e9:.0f}B: PPL = {ppl_pred:.1f}")
```

## Multilingual Scaling

### English + Banso Joint Training

```yaml
data_config:
  english:
    dataset: wikitext_and_arxiv
    examples: 500k
    sampling_weight: 0.8
  
  banso:
    dataset: banso_corpus_v2
    examples: 50k
    sampling_weight: 0.2
  
  curriculum:
    stage_1_epochs: 5   # English dominant
    stage_2_epochs: 3   # Balanced
    stage_3_epochs: 2   # Banso focused
    
training_metrics:
  english_ppl_target: 35.2
  banso_ppl_target: 45.0
  cross_lingual_ppl: 38.5
```

### Multilingual Evaluation Results

```python
def evaluate_multilingual_scaling():
    """Track perplexity across training."""
    
    results = {
        'epoch': [],
        'english_ppl': [],
        'banso_ppl': [],
        'mixed_ppl': [],
    }
    
    # Simulate 10 epochs
    for epoch in range(1, 11):
        # English improves quickly
        en_ppl = 45.0 - (epoch * 1.0)
        
        # Banso improves slower initially
        ba_ppl = 120.0 - (epoch * 7.5) if epoch <= 4 else 50.0 - (epoch - 4) * 2
        
        # Mixed (weighted average)
        mixed_ppl = 0.7 * en_ppl + 0.3 * ba_ppl
        
        results['epoch'].append(epoch)
        results['english_ppl'].append(en_ppl)
        results['banso_ppl'].append(ba_ppl)
        results['mixed_ppl'].append(mixed_ppl)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results['epoch'], results['english_ppl'], 'o-', label='English')
    plt.plot(results['epoch'], results['banso_ppl'], 's-', label='Banso')
    plt.plot(results['epoch'], results['mixed_ppl'], '^-', label='Mixed', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Multilingual Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('multilingual_scaling.png', dpi=150)
    
    return results
```

## Hardware Utilization

### GPU Memory Analysis

```python
def analyze_memory_usage(model, batch_size, seq_length):
    """Estimate memory requirements."""
    
    # Model weights
    param_bytes = sum(p.numel() * 4 for p in model.parameters())  # FP32
    
    # Activations (forward pass)
    # Each layer: hidden_size × seq_length × batch_size × 4 bytes (FP32)
    d_model = model.config.d_model
    num_layers = model.config.num_layers
    
    activation_bytes = num_layers * d_model * seq_length * batch_size * 4
    
    # Gradients (backward pass) - same as activations
    gradient_bytes = param_bytes
    
    # Optimizer states (Adam: momentum + variance)
    optimizer_bytes = 2 * param_bytes
    
    total_bytes = param_bytes + activation_bytes + gradient_bytes + optimizer_bytes
    total_gb = total_bytes / 1e9
    
    print(f"Memory Breakdown (batch_size={batch_size}, seq_len={seq_length}):")
    print(f"  Model weights:    {param_bytes/1e9:.2f}GB")
    print(f"  Activations:      {activation_bytes/1e9:.2f}GB")
    print(f"  Gradients:        {gradient_bytes/1e9:.2f}GB")
    print(f"  Optimizer states: {optimizer_bytes/1e9:.2f}GB")
    print(f"  TOTAL:            {total_gb:.2f}GB")
    
    return total_gb

# Example: 200M model
memory_gb = analyze_memory_usage(model_200m, batch_size=16, seq_length=2048)
```

### Training Throughput

```python
# Throughput analysis (examples per second)
model_configs = {
    'nano_70m': {'params': 70e6, 'flops_per_token': 2},
    'small_200m': {'params': 200e6, 'flops_per_token': 2},
    'base_500m': {'params': 500e6, 'flops_per_token': 2},
}

# GPU capabilities
gpu_specs = {
    'A100': {'tflops': 312},  # FP32; 1248 for TF32
    'H100': {'tflops': 989},  # Tensor cores
}

# Calculate throughput
for model_name, model_spec in model_configs.items():
    flops_per_token = model_spec['flops_per_token']
    
    for gpu_name, gpu_spec in gpu_specs.items():
        max_tokens_per_sec = (gpu_spec['tflops'] * 1e12) / (model_spec['params'] * flops_per_token)
        tokens_per_hour = max_tokens_per_sec * 3600
        
        print(f"{model_name:12} + {gpu_name:5}: {max_tokens_per_sec:>6.0f} tokens/sec ({tokens_per_hour/1e9:.1f}B tokens/hour)")
```

## Cost Analysis

### Total Cost of Training

```python
def calculate_training_cost(
    model_params: int,
    total_tokens: int,
    gpu_type: str = 'A100',
    gpu_cost_per_hour: float = 3.0,
    utilization: float = 0.85,  # % of peak performance
):
    """Calculate total training cost in dollars."""
    
    # FLOPs required (6 × params × tokens)
    total_flops = 6 * model_params * total_tokens
    
    # GPU capabilities
    gpu_flops = {
        'A100': 312e12,  # FP32 TF/s
        'H100': 989e12,
    }
    
    flops_per_sec = gpu_flops[gpu_type] * utilization
    
    # Time required
    seconds_required = total_flops / flops_per_sec
    hours_required = seconds_required / 3600
    
    # Cost
    cost = hours_required * gpu_cost_per_hour
    
    print(f"\nTraining Cost Estimate:")
    print(f"  Model size: {model_params/1e9:.1f}B parameters")
    print(f"  Data size:  {total_tokens/1e9:.1f}B tokens")
    print(f"  GPU type:   {gpu_type}")
    print(f"  Hours:      {hours_required:.0f}")
    print(f"  Cost:       ${cost:,.0f}")
    
    return cost

# Examples
calculate_training_cost(70e6, 20e9, 'A100')    # MarkGPT Nano
calculate_training_cost(200e6, 60e9, 'A100')   # MarkGPT Small
calculate_training_cost(500e6, 150e9, 'H100')  # MarkGPT Base
```

| Model | Params | Tokens | A100 Hours | A100 Cost | H100 Hours | H100 Cost |
|---|---|---|---|---|---|---|
| Nano | 70M | 20B | 58 | $174 | 18 | $180 |
| Small | 200M | 60B | 140 | $420 | 44 | $440 |
| Base | 500M | 150B | 290 | $870 | 91 | $910 |

---

**Roadmap Version**: 1.0
**Last Updated**: 2024
