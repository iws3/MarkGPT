# Model Compression & Efficiency

## Knowledge Distillation (Student-Teacher)

### Concept

Transfer knowledge from large (trained) model → small model.

```
Teacher Model (Ni Large)
  ├─ 500M params
  ├─ Perplexity: 16
  └─ Trained for months

        ↓ Distillation

Student Model (Smaller)
  ├─ 70M params
  ├─ Perplexity: 24 (worse, but much faster)
  └─ Trained in days

Result: 7x smaller, 10x faster, acceptable quality
```

### Training Loss

```
L_total = (1 - α) * L_CE + α * L_KL

Where:
- L_CE: Standard cross-entropy (hard targets)
- L_KL: KL divergence between teacher and student logits
- α: Weight (typically 0.5-0.9)
```

### Implementation

```python
class DistillationTrainer:
    def __init__(self, teacher_model, student_model, temperature=4.0):
        self.teacher = teacher_model.eval()  # Frozen
        self.student = student_model
        self.temperature = temperature
        self.alpha = 0.7
    
    def distillation_loss(self, student_logits, teacher_logits, targets):
        """Compute combined loss."""
        
        # Standard CE (hard targets)
        ce_loss = F.cross_entropy(student_logits, targets)
        
        # Soft targets: KL divergence of softened distributions
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        # Total loss
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss
        
        return total_loss
    
    def train_step(self, batch):
        input_ids = batch['input_ids'].to('cuda')
        targets = batch['targets'].to('cuda')
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids)
        
        # Student forward
        student_logits = self.student(input_ids)
        
        # Loss
        loss = self.distillation_loss(student_logits, teacher_logits, targets)
        
        return loss

# Training
trainer = DistillationTrainer(teacher, student)
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

for epoch in range(epochs):
    for batch in loader:
        loss = trainer.train_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### When to Use

- Deploying to resource-constrained devices
- Real-time inference requirements
- Edge devices (mobile, embedded)

---

## Quantization (Reducing Precision)

### INT8 Post-Training (Simplest)

```python
import torch.quantization as quant

# Calibrate on representative data
model.qconfig = quant.get_default_qconfig('fbgemm')
quant.prepare(model, inplace=True)

# Forward pass on calibration data (10-100 batches)
with torch.no_grad():
    for batch in calibration_loader:
        _ = model(batch)

# Convert to quantized
quant.convert(model, inplace=True)

# Result: 4x smaller, 2-3x faster
model.save('markgpt_int8.pt')

# Inference
quantized_output = model(input_ids)  # Automatic dequantization
```

### INT4 Quantization (with BitsAndBytes)

```python
from bitsandbytes.nn import Linear4bit

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model (automatically quantizes to INT4)
model = TransformerLM.from_pretrained(
    'markgpt-small',
    quantization_config=bnb_config,
    device_map='auto'
)

# Inference
logits = model(input_ids)  # Dequantized internally
```

### GPTQ (Better INT4)

```python
# Install: pip install auto-gptq

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Quantize-aware training
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

model = AutoGPTQForCausalLM.from_pretrained(
    'markgpt-small',
    quantize_config=quantize_config
)

# Apply GPTQ
model.quantize(calibration_dataset)
model.save_quantized('markgpt_gptq.bin')

# Inference: Near fp16 quality at INT4 speeds
```

### Comparison

| Method | Bits | Speed | Quality | Size | Ease |
|--------|------|-------|---------|------|------|
| FP32 | 32 | 1x | 100% | 800MB | Easy |
| FP16 | 16 | 2x | 98% | 400MB | Easy |
| INT8 | 8 | 4x | 85% | 200MB | Easy |
| INT4 NF4 | 4 | 8x | 75% | 100MB | Medium |
| INT4 GPTQ | 4 | 8x | 90% | 100MB | Hard |

---

## Pruning (Remove Weights)

### Magnitude Pruning (Simple)

```python
def magnitude_prune(model, prune_ratio=0.1):
    """Remove 10% of weights with smallest magnitude."""
    
    total_params = 0
    pruned_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get weight mask
            mask = torch.abs(module.weight) > torch.quantile(
                torch.abs(module.weight),
                prune_ratio
            )
            
            # Set small weights to 0
            module.weight.data *= mask.float()
            
            pruned_params += (~mask).sum().item()
            total_params += mask.numel()
    
    print(f"Pruned {pruned_params / total_params * 100:.1f}% of weights")
    return model

# Use
pruned_model = magnitude_prune(model, prune_ratio=0.3)
torch.save(pruned_model, 'markgpt_pruned.pt')
```

### Structured Pruning (Remove Entire Heads/Layers)

```python
def prune_attention_heads(model, num_heads_to_prune=2):
    """Remove entire attention heads (structured)."""
    
    for block in model.transformer_blocks:
        # Identify important heads (using attention weights)
        importance = block.attn.head_importance()  # Custom method
        
        # Remove least important heads
        heads_to_remove = torch.argsort(importance)[:num_heads_to_prune]
        
        # Reshape and remove
        # (Implementation: modify attention shape)
    
    return model
```

### Results

- Magnitude pruning: 30% pruning → 10-15% speed, minimal accuracy loss
- Structured pruning: Remove layers → 40% speed, some accuracy loss

---

## Low-Rank Adaptation (LoRA) for Efficiency

Instead of fine-tuning all parameters, update only low-rank adaptations:

```python
from peft import LoraConfig, get_peft_model

# Original model
model = TransformerLM.load_pretrained('markgpt-small')

# LoRA configuration
lora_config = LoraConfig(
    r=8,                           # Rank
    lora_alpha=16,
    target_modules=['q_proj', 'v_proj'],  # Apply to attention
    lora_dropout=0.1,
)

# Wrap model
lora_model = get_peft_model(model, lora_config)

# Trainable params: Only ~0.8% of original!
print(lora_model.print_trainable_parameters())
# trainable params: 819,200 || all params: 113,222,656 || trainable%: 0.72

# Fine-tune
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-4)

for batch in loader:
    logits = lora_model(batch['input_ids'])
    loss = criterion(logits, batch['targets'])
    loss.backward()
    optimizer.step()

# Save only LoRA weights (~8MB instead of 400MB)
lora_model.save_pretrained('markgpt_lora_banso')
```

---

## Model Ensembling (Cheap Improvement)

Run multiple models and average:

```python
def ensemble_inference(input_ids, models, weights=None):
    """Combine predictions from multiple models."""
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    ensemble_logits = None
    
    for model, weight in zip(models, weights):
        with torch.no_grad():
            logits = model(input_ids)
            if ensemble_logits is None:
                ensemble_logits = weight * logits
            else:
                ensemble_logits += weight * logits
    
    # Most confident prediction
    predicted_tokens = torch.argmax(ensemble_logits, dim=-1)
    
    return predicted_tokens

# Usage
models = [
    TransformerLM.load('markgpt_v1.pt'),
    TransformerLM.load('markgpt_v2.pt'),
]

output = ensemble_inference(input_ids, models)
```

Cost: 2x inference time, ~3-5% accuracy improvement

---

## MarkGPT Deployment Options

```python
class MarkGPTDeployment:
    def __init__(self, mode='balanced'):
        self.mode = mode
    
    def load_model(self):
        if self.mode == 'quality':
            # Full model
            return TransformerLM.load('markgpt_base.pt')
        
        elif self.mode == 'balanced':
            # INT8 quantized
            return TransformerLM.load('markgpt_int8.pt')
        
        elif self.mode == 'speed':
            # Distilled + INT4
            model = TransformerLM.load('markgpt_student_int4.pt')
            return model
        
        elif self.mode == 'light':
            # Pruned + quantized (for mobile)
            return TransformerLM.load('markgpt_mobile.onnx')

# Inference latency comparison
configurations = {
    'quality': {'size': '800MB', 'latency': '100ms', 'token_cost': '1x'},
    'balanced': {'size': '200MB', 'latency': '50ms', 'token_cost': '1.1x'},
    'speed': {'size': '100MB', 'latency': '20ms', 'token_cost': '1.3x'},
    'light': {'size': '30MB', 'latency': '5ms', 'token_cost': '1.8x'},
}
```

---

**Model Compression v1.0**
**Last Updated**: 2024
