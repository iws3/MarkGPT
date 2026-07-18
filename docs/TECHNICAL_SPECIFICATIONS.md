# Technical Specifications Document

## MarkGPT-Final Release Specifications

### Version Information
- **Version**: 1.0.0
- **Release Date**: 2024
- **Status**: Stable Production Release
- **Maintenance**: Long-term support (24 months minimum)

---

## Model Specifications

### Architecture Summary
```
MarkGPT: Decoder-only Transformer
├─ Input: Token IDs (BPE tokenized)
├─ Processing:
│  ├─ Token Embeddings (learned)
│  ├─ Positional Embeddings (RoPE)
│  └─ Stack of N Transformer Blocks
│      ├─ Multi-head Self-Attention (causal mask)
│      ├─ Position-wise Feed-Forward
│      ├─ Layer Normalization (pre-norm)
│      └─ Residual Connections
└─ Output: Next token logits (vocab_size)
```

### Model Sizes

| Name | Params | d_model | num_heads | num_layers | Max Seq | Training Tokens |
|---|---|---|---|---|---|---|
| Nano | 70M | 512 | 8 | 16 | 2048 | 20B |
| Small | 200M | 768 | 12 | 24 | 2048 | 60B |
| Base | 500M | 1024 | 16 | 32 | 2048 | 150B |
| Medium | 1.3B | 2048 | 16 | 40 | 4096 | 300B |

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Attention Heads | 8/12/16 | Divisible by d_model |
| Head Dimension | 64 | Standard across models |
| FFN Hidden Dim | 4 × d_model | Transformer standard |
| Activation | GELU | Better than ReLU for LLMs |
| Dropout Rate | 0.1 | Prevent overfitting |
| Initialization | N(0, 0.02) | Stable training |
| Layer Norm | Pre-norm | Avoids collapse |
| Positional Encoding | RoPE | Better extrapolation |

---

## Training Specifications

### Dataset Composition
```
Training Data (Multi-source):
├─ English (80%)
│  ├─ Wikipedia: 20M articles
│  ├─ BookCode: 5M pages
│  ├─ ArXiv: 2M papers
│  └─ Web (cleaned): 30M pages
└─ Banso (20%)
   ├─ BANSO corpus: 50K sentences
   ├─ Community contributions: 30K
   └─ Synthetic augmentation: 70K
```

### Training Procedure
```python
# Configuration
batch_size = 256
learning_rate = 1e-4
warmup_steps = 5000
total_epochs = 3
mixed_precision = "bf16" (or fp16)
gradient_accumulation = 2
max_grad_norm = 1.0

# Distributed setup
num_gpus = 8
distributed = True
sync_batch_norm = True

# Checkpointing
save_steps = 500
eval_steps = 500
save_total_limit = 3
```

### Optimization Details
```
Optimizer: AdamW
- β₁ = 0.9
- β₂ = 0.95
- ε = 1e-8

Learning Rate Schedule: Cosine Annealing
- Warmup: linear from 0 → 1e-4 over 5000 steps
- Decay: cosine decay to 1% of max over remaining steps
- Restart: 1 restart per epoch

Regularization:
- Weight decay: 0.01
- Dropout: 0.1 per layer
- Gradient clipping: 1.0
```

---

## Inference Specifications

### Hardware Requirements

#### Minimum (CPU Only)
- RAM: 4GB (for Nano), 8GB (for Small)
- Storage: 200MB (Nano), 500MB (Small)
- CPU: Modern multi-core (Intel i5 or equivalent)

#### Recommended (GPU)
- GPU Memory: 6GB (A40 equivalent)
- RAM: 8GB
- Storage: 1GB

#### Optimal (Production)
- GPU: 2+ GPUs for batching/failover
- GPU Memory: 8-16GB (A100 or H100)
- RAM: 16GB+
- Network: 1Gbps+ for API service

### Inference Performance

| Hardware | Model | Batch Size | Latency | Throughput |
|---|---|---|---|---|
| CPU (i9) | Nano | 1 | 850ms | 1.2 tok/s |
| A40 | Nano | 32 | 180ms (batch) | 180 tok/s |
| A100 | Small | 64 | 250ms (batch) | 250 tok/s |
| H100 | Base | 256 | 300ms (batch) | 850 tok/s |

### Memory Requirements (FP16)

| Model | Model Size | KV Cache (2k) | Total (bs=1) | Total (bs=32) |
|---|---|---|---|---|
| Nano | 130MB | 8MB | 138MB | 390MB |
| Small | 380MB | 24MB | 404MB | 1.1GB |
| Base | 950MB | 60MB | 1.0GB | 2.9GB |

---

## API Specification

### REST Endpoint: `/v1/completions`

**Request**:
```json
{
  "prompt": "Once upon a time, there was",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 50,
  "repetition_penalty": 1.0,
  "stop": ["\n\n"]
}
```

**Response**:
```json
{
  "id": "cmpl-71234567",
  "object": "text_completion",
  "created": 1673845000,
  "model": "markgpt-small",
  "choices": [{
    "text": " was a great adventure about to begin...",
    "index": 0,
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 15,
    "total_tokens": 23
  }
}
```

### Response Codes
```
200 OK          Successful generation
400 Bad Request Invalid parameters
429 Rate Limited Too many requests
503 Unavailable Service temporarily down
```

---

## Software Dependencies

### Python Environment
```
python >= 3.10
torch >= 2.0.0
transformers >= 4.30.0
accelerate >= 0.20.0
peft >= 0.4.0
bitsandbytes >= 0.40.0
datasets >= 2.10.0
wandb >= 0.14.0
```

### Optional
```
flash-attn >= 2.0.0  (for Flash Attention)
onnxruntime >= 1.14.0 (for ONNX inference)
tensorrt >= 8.5.0 (for TensorRT optimization)
```

---

## Compliance & Licensing

### Model Weights
- **License**: Apache 2.0 (for MarkGPT code)
- **Usage**: Commercial and research allowed
- **Attribution**: Required (cite paper/GitHub)
- **Modifications**: Allowed with license included

### Training Data
```
English Data:
├─ Wikipedia: CC BY-SA 3.0
├─ ArXiv: CC BY 4.0
├─ Web (cleaned): Various, commercial use filtered
└─ Licensed for training, distribution of original required

Banso Data:
└─ Community-contributed under partnership agreement
   (See PARTNERSHIP_FRAMEWORK.md)
```

### Model Card
Documentation available at:
- `capstone/markgpt/model_card.md`
- [Hugging Face Model Hub](https://huggingface.co/markgpt)

### Ethical Use Policy
```
Prohibited Uses:
- [ ] Deliberate misinformation/disinformation
- [ ] Harassment or intimidation
- [ ] Illegal activity
- [ ] Unauthorized access to systems
- [ ] Deception without consent

Recommended Responsible AI Practices:
- [ ] Use with human-in-the-loop for critical decisions
- [ ] Monitor for biased outputs
- [ ] Transparent disclosure when using model outputs
- [ ] Regular audits and bias testing
```

---

## Maintenance & Support

### Bug Reporting
- GitHub Issues: https://github.com/markgpt-community/MarkGPT-LLM-Curriculum/issues
- Email: bugs@markgpt.org
- Response time: 48 hours (critical), 1 week (normal)

### Security Issues
- **Do NOT** open public GitHub issues
- Email: security@markgpt.org
- Response time: 24 hours
- Coordinated disclosure process

### Support Channels
1. **Documentation**: Check FAQs and guides first
2. **Discussions**: GitHub Discussions for Q&A
3. **Community**: Discord server for peer support
4. **Issues**: Bug reports and feature requests

### Version Management
```
Versioning Scheme: MAJOR.MINOR.PATCH

1.0.0  → Initial release (current)
1.1.0  → Minor features/improvements (planned)
2.0.0  → Breaking changes/major refactor (future)

Long-term Support (LTS):
- Version 1.x: Bugfixes and security patches for 24 months
- Version 2.x: Extended support available for enterprise customers
```

---

## References

- **Paper**: [MarkGPT Preprint](https://arxiv.org/abs/XXXX.XXXXX) (coming 2024)
- **Source Code**: [GitHub Repository](https://github.com/markgpt-community/MarkGPT-LLM-Curriculum)
- **Model Weights**: [Hugging Face Hub](https://huggingface.co/markgpt)
- **Community**: [Discord Server](https://discord.gg/markgpt)

---

**Specification Version**: 1.0
**Last Updated**: 2024
**Approved By**: Technical Steering Committee
