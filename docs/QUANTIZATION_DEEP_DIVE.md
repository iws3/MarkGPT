# Quantization Deep Dive

## INT8 Quantization

### Post-Training Quantization (PTQ)

```python
import torch
from torch.quantization import quantize_dynamic

# Load model
model = torch.load('model.pt', map_location='cpu')
model.eval()

# Dynamic quantization (INT8)
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.LSTM, torch.nn.LSTMCell},
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model, 'model_quantized.pt')

# Size comparison
original_size = os.path.getsize('model.pt') / 1e6
quantized_size = os.path.getsize('model_quantized.pt') / 1e6
print(f"Original: {original_size:.1f}MB → Quantized: {quantized_size:.1f}MB ({100*quantized_size/original_size:.1f}%)")
```

### Quantization Aware Training (QAT)

```python
from torch.quantization import prepare_qat, convert
import torch.quantization as quantization

# Step 1: Insert fake quantization in training
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
prepared_model = prepare_qat(model, inplace=False)

# Step 2: Train with quantization simulation
for epoch in range(num_epochs):
    for batch in train_loader:
        output = prepared_model(batch)
        loss = criterion(output, batch['labels'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Step 3: Convert to fully quantized model
quantized_model = convert(prepared_model, inplace=False)

# Save for inference
torch.save(quantized_model.state_dict(), 'model_qat.pt')
```

## INT4 Quantization with BitsAndBytes

### Setup BitsAndBytes

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Define 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "markgpt-base",
    device_map="auto",
    quantization_config=bnb_config,
)

# Check memory usage
print(model.get_memory_footprint() / 1e9, "GB")
```

### Fine-tune Quantized Model with LoRA

```python
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Add LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Now trainable with LoRA on top of 4-bit quantized base
model.print_trainable_parameters()
# output: trainable params: 123,456 || all params: 6,234,567 || trainable%: 1.98%
```

## Calibration Strategies

### Entropy-Based Calibration

```python
def entropy_calibration(model, calibration_data, num_bits=8):
    """Find optimal quantization ranges using entropy."""
    
    # Collect statistics
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu())
        return hook
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook_fn(name)))
    
    # Collect activations on calibration data
    model.eval()
    with torch.no_grad():
        for batch in calibration_data:
            model(batch)
    
    # Compute quantization ranges
    ranges = {}
    for name, acts in activations.items():
        acts_cat = torch.cat(acts, dim=0)
        min_val = acts_cat.min()
        max_val = acts_cat.max()
        
        # Entropy-based: find threshold that minimizes KL divergence
        # between original and quantized distributions
        thresholds = np.linspace(0.1 * max_val, max_val, 100)
        best_threshold = max_val
        min_kl = float('inf')
        
        for threshold in thresholds:
            # Simulate quantization
            clipped = torch.clamp(acts_cat, min_val, threshold)
            quantized = (clipped * (2**num_bits - 1) / threshold).round()
            
            # Compute KL divergence
            p, _ = np.histogram(acts_cat.numpy(), bins=128)
            q, _ = np.histogram(quantized.numpy(), bins=128)
            p = p / p.sum() + 1e-8
            q = q / q.sum() + 1e-8
            kl = np.sum(p * np.log(p / q))
            
            if kl < min_kl:
                min_kl = kl
                best_threshold = threshold
        
        ranges[name] = (min_val.item(), best_threshold.item())
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    return ranges
```

## Quantization-Aware Training (QAT) vs Post-Training (PTQ)

| Method | Accuracy | Speed | Memory | Training Required |
|---|---|---|---|---|
| **PTQ (INT8)** | ~98% | 4x | 4x | No |
| **QAT (INT8)** | ~99% | 4x | Same | Yes (~1 epoch) |
| **PTQ (INT4)** | ~92% | 10x | 11x | No |
| **QAT (INT4)** | ~97% | 10x | Same | Yes (~3-5 epochs) |

**Recommendation**:
- Use PTQ for quick deployment (minimal accuracy loss for INT8)
- Use QAT for production (better accuracy, especially for INT4)
- Use INT4 only if memory ultra-critical (phones, edge devices)

## Evaluation

### Quantization Impact Assessment

```python
def assess_quantization(model, quantized_model, test_loader):
    """Compare original vs quantized model performance."""
    
    results = {
        'accuracy': {'original': 0, 'quantized': 0},
        'latency': {'original': [], 'quantized': []},
        'memory': {}
    }
    
    model.eval()
    quantized_model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            # Original model
            start = time.time()
            pred_orig = model(batch)
            results['latency']['original'].append(time.time() - start)
            
            # Quantized model
            start = time.time()
            pred_quant = quantized_model(batch)
            results['latency']['quantized'].append(time.time() - start)
            
            # Accuracy (example)
            acc_orig = (pred_orig.argmax(-1) == batch['labels']).float().mean()
            acc_quant = (pred_quant.argmax(-1) == batch['labels']).float().mean()
            
            results['accuracy']['original'] += acc_orig.item()
            results['accuracy']['quantized'] += acc_quant.item()
    
    # Average metrics
    n = len(test_loader)
    results['accuracy']['original'] /= n
    results['accuracy']['quantized'] /= n
    
    results['latency']['original'] = np.mean(results['latency']['original'])
    results['latency']['quantized'] = np.mean(results['latency']['quantized'])
    
    # Memory
    results['memory']['original'] = model.get_memory_footprint() / 1e9
    results['memory']['quantized'] = quantized_model.get_memory_footprint() / 1e9
    
    # Print report
    print("\n" + "="*60)
    print("QUANTIZATION IMPACT ASSESSMENT")
    print("="*60)
    print(f"Accuracy:   {results['accuracy']['original']:.4f} → {results['accuracy']['quantized']:.4f}")
    print(f"  Δ: {(results['accuracy']['quantized'] - results['accuracy']['original'])*100:+.2f}%")
    
    print(f"\nLatency:    {results['latency']['original']*1000:.2f}ms → {results['latency']['quantized']*1000:.2f}ms")
    print(f"  Speedup: {results['latency']['original'] / results['latency']['quantized']:.2f}x")
    
    print(f"\nMemory:     {results['memory']['original']:.2f}GB → {results['memory']['quantized']:.2f}GB")
    print(f"  Reduction: {(1 - results['memory']['quantized'] / results['memory']['original'])*100:.1f}%")
    print("="*60 + "\n")
    
    return results

assess_quantization(model, quantized_model, test_loader)
```

---

**Guide Version**: 1.0
**Last Updated**: 2024
