# Model Comparison & Selection Guide

## Model Size Comparison

### Feature Comparison

| Feature | Nano | Small | Base |
|---|---|---|---|
| **Parameters** | 70M | 200M | 500M |
| **Model Size (FP32)** | 280MB | 800MB | 2.0GB |
| **Quantized (INT8)** | 70MB | 200MB | 500MB |
| **Min GPU VRAM** | 2GB | 4GB | 8GB |
| **Max Seq Len** | 2048 | 2048 | 2048 |
| **Training Hrs (8x A100)** | ~30 | ~45 | ~120 |

### Performance Comparison

| Metric | Nano | Small | Base | Target |
|---|---|---|---|---|
| **English PPL** | 24.0 | 16.5 | 11.2 | <10 |
| **Banso PPL** | 45.0 | 32.0 | 24.0 | <20 |
| **Inference Latency (50 tokens)** | 45ms (GPU) | 85ms | 180ms | <500ms |
| **Throughput (tokens/sec, A100)** | 1100 | 600 | 280 | >100 |

---

## Selection Decision Tree

```
What's your primary constraint?
│
├─ SPEED (latency critical)
│  └─ Use MarkGPT-Nano
│     (45ms latency, ideal for real-time)
│
├─ PERFORMANCE (accuracy critical)
│  └─ Use MarkGPT-Base
│     (11.2 PPL, highest quality)
│
├─ BALANCE (default choice)
│  └─ Use MarkGPT-Small
│     (16.5 PPL, 85ms latency, good tradeoff)
│
└─ COST (minimal resources)
   └─ Use MarkGPT-Nano + Quantization
      (70MB model, CPU-compatible)
```

---

## Use Case Recommendations

### Interactive Chatbot
**Best**: MarkGPT-Small or MarkGPT-Nano
- Low latency critical (human expects <1s response)
- Small throughput (1-10 concurrent chats)
- Can be deployed on edge devices

```python
model = AutoModel.from_pretrained('markgpt-small', device_map='auto')
# Single GPU sufficient
```

### Batch Processing / Research
**Best**: MarkGPT-Base or MarkGPT-Large
- Latency less critical
- High throughput needed (process 1000s documents)
- Accuracy important

```python
model = AutoModel.from_pretrained('markgpt-base')
batch_size = 256  # Large batches for utilization
```

### Mobile/Edge Deployment  
**Best**: MarkGPT-Nano (quantized) or MarkGPT-Nano (pruned)
- Extreme memory constraints (<1GB)
- Accept lower accuracy
- CPU-only typical

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModel.from_pretrained(
    'markgpt-nano',
    quantization_config=bnb_config,
    device_map='cpu'
)
```

### Research / Fine-tuning Target
**Best**: MarkGPT-Base
- Start large, then compress
- More stable convergence
- Better transfer learning

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=16, lora_alpha=32)
model = get_peft_model(model, lora_config)
# Fine-tune on specific domain
```

---

## Domain-Specific Recommendations

### Education (MarkGPT as Teaching Tool)
- **Model**: MarkGPT-Small
- **Setup**: Single GPU server
- **Deployment**: Web interface
- **Cost**: ~$100-200/month

### Business Intelligence
- **Model**: MarkGPT-Base
- **Setup**: Distributed (2-4 GPUs)
- **Deployment**: API service
- **Cost**: ~$500-1000/month

### Research / Science
- **Model**: MarkGPT-Base or larger
- **Setup**: Multi-GPU cluster
- **Deployment**: Batch jobs
- **Cost**: Variable (on-demand)

### Low-Resource Regions (Banso focus)
- **Model**: MarkGPT-Nano (quantized)
- **Setup**: CPU or mobile
- **Deployment**: Local/offline
- **Cost**: Free (open source)

---

## Fine-tuning & Adaptation Guide

### When to Fine-tune vs Use OOB

**Use Off-the-Shelf (OOB)**:
- General text generation
- When domain knowledge not critical
- Time/resource constrained
- Want immediate results

**Fine-tune**:
- Domain-specific terminology
- Different writing style
- Private/sensitive data
- Want optimized performance

### Fine-tuning Complexity vs Gain

```
Fine-tuning Effort (hours)
│     ┌─────────────────────────────────
│     │ Full Fine-tune (24-48h)
│ 100 │         ↑ Accuracy gain: ~3-5%
│     │
│ 50  │  LoRA Fine-tune (4-8h)
│     │      ↑ Accuracy gain: ~2-3%
│     │
│ 10  │  Prompt Engineering
│     │  ↑ Accuracy gain: ~1-2%
│ 0   └─────────────────────────────────
      None  Few-shot Prompt  LoRA  Full-FT
                    Fine-tuning Strategy
```

**Recommendation**: Start with prompt engineering, then LoRA if needed.

---

## Cost-Benefit Analysis

### Scenario: Production Chatbot (10K users)

**Option 1: MarkGPT-Nano (CPU)**
- Inference: $100/month
- Storage: $10/month
- Monitoring: $20/month
- **Total**: $130/month

**Option 2: MarkGPT-Small (Single GPU)**
- GPU hours: $500/month
- Storage: $10/month
- Monitoring: $20/month
- **Total**: $530/month

**Option 3: MarkGPT-Base (Multi-GPU)**
- GPU hours: $1500/month
- Load balancer: $100/month
- Storage: $20/month
- Monitoring: $50/month
- **Total**: $1670/month

**Recommendation**: Start with Option 1 (CPU), upgrade if needed.

---

## Quick Sizing Tool

```python
def estimate_resource_requirements(num_users, avg_tokens_per_request=100):
    """Quick sizing calculator."""
    
    # Typical metrics
    tokens_per_sec_nano = 100  # CPU
    tokens_per_sec_small = 300  # Single GPU
    tokens_per_sec_base = 700   # Multi-GPU
    
    # Calculate required throughput
    requests_per_sec = num_users * 0.1  # 10% of users at once
    tokens_per_sec_needed = requests_per_sec * avg_tokens_per_request
    
    print(f"\nRequirements for {num_users} users:")
    print(f"  Required throughput: {tokens_per_sec_needed:.0f} tokens/sec")
    
    if tokens_per_sec_needed < 100:
        print(f"  ✓ USE: MarkGPT-Nano (CPU, cheapest)")
    elif tokens_per_sec_needed < 300:
        print(f"  ✓ USE: MarkGPT-Small (1x GPU)")
    else:
        print(f"  ✓ USE: MarkGPT-Base (2-4x GPUs)")

# Examples
estimate_resource_requirements(100)    # Small app
estimate_resource_requirements(1000)   # Medium app
estimate_resource_requirements(10000)  # Large app
```

---

## Experimentation Strategy

### Phase 1: Validate Approach (Week 1)
1. Prototype with MarkGPT-Nano
2. Test on sample data
3. Measure baseline metrics
4. Iterate on prompts

### Phase 2: Optimize (Week 2-3)
1. Upgrade to MarkGPT-Small if needed
2. Fine-tune on 5% of data
3. A/B test with users
4. Measure improvements

### Phase 3: Scale (Week 4+)
1. Full fine-tuning if >2% improvement
2. Deploy to production
3. Monitor metrics
4. Plan capacity upgrades

---

**Model Selection Guide v1.0**
**Last Updated**: 2024
