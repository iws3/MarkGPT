# Frequently Asked Technical Questions (Extended)

## Model Architecture Questions

### Q: Why use RoPE instead of absolute positional encoding?
**A**: Rotary Position Embedding (RoPE) provides several advantages:
- **Better extrapolation**: Works for longer sequences than training length
- **Theoretically grounded**: Based on complex number rotation
- **Efficient**: No extra parameters added
- **Compatible**: Works with Flash Attention

```python
# RoPE implementation in MarkGPT
def apply_rope(q, k, freqs, t_offset=0):
    """Apply rotary position embedding."""
    # freqs shape: (seq_len, head_dim/2)
    # q, k shape: (batch, num_heads, seq_len, head_dim)
    
    seq_len = q.shape[-2]
    cos = torch.cos(freqs.unsqueeze(0).unsqueeze(0))
    sin = torch.sin(freqs.unsqueeze(0).unsqueeze(0))
    
    # Complex number rotation
    q_rot = (q[..., :head_dim//2] * cos - q[..., head_dim//2:] * sin, 
             q[..., :head_dim//2] * sin + q[..., head_dim//2:] * cos)
    k_rot = (k[..., :head_dim//2] * cos - k[..., head_dim//2:] * sin,
             k[..., :head_dim//2] * sin + k[..., head_dim//2:] * cos)
    
    return q_rot, k_rot
```

---

### Q: What's the difference between LoRA and full fine-tuning?

**A**: Comparison table:

| Aspect | Full Fine-tuning | LoRA |
|---|---|---|
| **Trainable Params** | 100% | 1-5% |
| **Memory** | High | 4-10x less |
| **Speed** | Slower | 2-3x faster |
| **Performance** | Slightly better | ~99% of full |
| **Stability** | Sensitive to LR | More robust |
| **Deployment** | Single model | Base + adapter |
| **Checkpoints** | 500MB-2GB | 1-10MB |

**When to use**:
- Use LoRA for: Custom domains, quick iteration, resource-constrained
- Use full fine-tuning for: Maximum performance, one-time training

---

## Training Questions

### Q: How do I know if my training is converging properly?

**A**: Look for these signals:

✓ **Good signs**:
- Smooth loss curve (no spikes)
- Consistent improvement in validation metrics
- Gradient norms in reasonable range (0.1 - 10)
- Perplexity approaches target

✗ **Bad signs**:
- Loss plateaus early
- Massive gradient spikes (>100)
- NaN or Inf values
- No validation improvement for many epochs

**Diagnostic script**:
```python
def diagnose_training(model, dataloader, num_batches=100):
    """Check training health."""
    
    grad_norms = []
    losses = []
    
    model.train()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        output = model(**batch)
        loss = output.loss
        loss.backward()
        
        # Check gradient norms
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norms.append(grad_norm.item())
        losses.append(loss.item())
        
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Loss range: [{min(losses):.3f}, {max(losses):.3f}]")
    print(f"Gradient norms: [{min(grad_norms):.3f}, {max(grad_norms):.3f}]")
    
    if max(losses) > 3 * min(losses):
        print("⚠️ Warning: High loss variance, may need lower learning rate")
```

---

### Q: Should I use warmup? For how long?

**A**: Yes, warmup is important. Guidelines:

- **Nano (70M)**: 500-1000 steps
- **Small (200M)**: 2000-5000 steps
- **Base (500M)**: 5000-10000 steps

**Formula**: `warmup_steps ≈ training_steps / 50` to `1 / 100`

```python
def linear_warmup_cosine_decay(epoch, num_epochs, warmup_epochs=0.1):
    """Standard schedule."""
    warmup_steps = int(warmup_epochs * 100)  # Epochs to steps
    total_steps = num_epochs * 100
    
    if epoch < warmup_steps:
        return epoch / warmup_steps
    else:
        return 0.5 * (1 + math.cos((epoch - warmup_steps) / (total_steps - warmup_steps) * math.pi))
```

---

## Inference Questions

### Q: How can I speed up inference?

**A**: Techniques in order of impact:

1. **Batch Inference** (3-10x): Process multiple requests together
2. **KV Caching** (3-5x): Reuse attention key-values
3. **Quantization** (2-4x): INT8/INT4 precision reduction
4. **Flash Attention** (1.5-2x): I/O efficient attention
5. **Speculative Decoding** (1.5-3x): Draft model verification

**Quick implementation**:
```python
# Enable all at once
model = AutoModel.from_pretrained(
    'markgpt-small',
    device_map='auto',
    load_in_8bit=True,  # Quantization
    attn_implementation='flash_attention_2'  # Flash Attention
)

# KV caching automatic in generation
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    use_cache=True,  # Enabled by default
)
```

---

### Q: How do I choose between temperature and top-p?

**A**: Different use cases:
- **Temperature**: Control randomness (0.1-2.0)
  - 0.1: Deterministic, same every time
  - 1.0: Balanced randomness
  - 2.0: Very creative, more errors
- **Top-p**: Nucleus sampling (0.8-0.99)
  - Cumulative probability threshold
  - Works across all temperature settings
  - Often more stable than temperature alone

**Recommendation**: Use both together:
```python
# Conservative: high-quality outputs
outputs = model.generate(input_ids, temperature=0.7, top_p=0.9)

# Creative: diverse outputs
outputs = model.generate(input_ids, temperature=1.2, top_p=0.95)

# Research: reproducible
outputs = model.generate(input_ids, temperature=0.001, top_p=1.0, seed=42)
```

---

## Multilingual Questions

### Q: Should I train on English and Banso jointly or separately?

**A**: Joint training (our approach) is generally better:

**Joint Advantages**:
- Shared representations enable transfer
- Mutual improvement (English helps Banso)
- Single model handles both
- Resources efficiently used

**Separate Advantages**:
- Maximum performance for each language
- Independent optimization

**Our evidence** (from TRAINING_ROADMAP.md):
```
Bilingual Model (Joint):
  English PPL:  36.1 (vs 35.2 monolingual, -0.3% drop)
  Banso PPL:    45.3 (vs 98.5 zero-shot, +55% improvement!)
  
Result: Slight English degradation << Massive Banso gain
        → Use joint training model
```

---

### Q: How do I handle domain-specific terminology?

**A**: Extend the tokenizer:

```python
# Add domain tokens
tokenizer.add_tokens([
    "<BANSO_DOMAIN>",
    "<MEDICAL_TERM>",
    "<TECHNICAL_TERM>"
])

# Resize model embeddings
model.resize_token_embeddings(len(tokenizer))

# Initialize new token embeddings
with torch.no_grad():
    # Use average of existing embeddings
    existing_weight = model.transformer.wte.weight[:50257]
    model.transformer.wte.weight[50257:] = existing_weight.mean(dim=0).unsqueeze(0).expand(3, -1)

# Fine-tune on domain data with new tokens
```

---

## Evaluation Questions

### Q: What's a "good" perplexity for MarkGPT?

**A**: Depends on the model and dataset:

| Model | English | Banso | Notes |
|---|---|---|---|
| Nano | 24.0 | 45.0 | After training on 20B tokens |
| Small | 16.5 | 32.0 | After 60B tokens |
| Base | 11.2 | 24.0 | After 150B tokens |

**Remember**: Lower is better, but context matters
- Domain-specific data → naturally lower PPL
- Out-of-domain → higher PPL
- Compare against similar models

---

### Q: Should I test on multilingual benchmarks?

**A**: Yes, but be aware of limitations:

**Good benchmarks**:
- FLORES (machine translation)
- GLUE (English understanding)
- XNLI (cross-lingual inference)
- X-SQuAD (cross-lingual QA)

**For Banso specifically**:
- Limited benchmarks available
- Consider creating custom evaluation sets
- Human evaluation crucial
- Document your rubric

---

## Deployment Questions

### Q: How do I deploy to production?

**A**: Basic checklist:

1. ✓ **Model preparation**
   - Compress (quantize if needed)
   - Export to ONNX (optional)
   - Test on target hardware

2. ✓ **API setup**
   - Choose framework (Flask, FastAPI, vLLM)
   - Implement rate limiting
   - Add authentication

3. ✓ **Monitoring**
   - Latency tracking
   - Error rates
   - GPU/CPU utilization
   - User feedback

4. ✓ **Scaling**
   - Load balancer setup
   - Multiple replicas
   - Auto-scaling rules

**Quick Flask example**:
```python
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
model = AutoModel.from_pretrained('markgpt-small')
tokenizer = AutoTokenizer.from_pretrained('markgpt-small')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json['prompt']
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=100)
    
    text = tokenizer.decode(output_ids[0])
    return jsonify({'output': text})
```

---

**FAQ Version**: 2.0
**Last Updated**: 2024
**Total Questions**: 20+
