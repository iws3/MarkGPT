# System Architecture & Infrastructure

## MarkGPT System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────┐
│              User Applications                   │
│  (Web UI, API Clients, Research Tools)          │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│         API Layer (REST/gRPC)                    │
│  ├─ Authentication & Rate Limiting              │
│  ├─ Request Validation                          │
│  └─ Response Formatting                         │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│     Inference Engine (PyTorch)                   │
│  ├─ Model Loading & Caching                     │
│  ├─ Batch Processing                            │
│  ├─ KV Cache Management                         │
│  └─ GPU/CPU Fallback                            │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│      Hardware Layer                              │
│  ├─ GPU (A100/H100/A40)                         │
│  ├─ CPU (Scaling)                               │
│  └─ Memory (VRAM/RAM)                           │
└─────────────────────────────────────────────────┘
```

### Component Details

#### 1. API Layer
- **Framework**: FastAPI (async, performant)
- **Authentication**: JWT tokens or API keys
- **Rate Limiting**: Token bucket algorithm
- **Monitoring**: Prometheus metrics export

```python
# API endpoint structure
@app.post("/v1/completions")
async def create_completion(request: CompletionRequest) -> CompletionResponse:
    """Generate text completion."""
    # Validation → Inference → Response
```

#### 2. Inference Engine
- **Batch Manager**: Groups requests for efficiency
- **Cache Manager**: KV cache lifecycle
- **Device Manager**: Auto GPU/CPU selection
- **Quantization**: INT8/INT4 support

```python
class InferenceEngine:
    def __init__(self, model_name, device_map='auto'):
        self.model = load_model(model_name, device_map)
        self.cache_manager = KVCacheManager()
        self.batch_manager = BatchManager()
    
    async def generate(self, prompt, config):
        # 1. Encode prompt
        # 2. Batch with pending requests
        # 3. Forward pass
        # 4. Update cache
        # 5. Return output
```

---

## Training Infrastructure

### Distributed Training Setup

```
Master Node (Rank 0)
├─ Orchestration
├─ Logging & Monitoring
└─ Checkpointing

Worker Nodes (Rank 1-7)
├─ Data Loading
├─ Forward Pass
├─ Backward Pass
└─ Gradient Sync (via NCCL)

All nodes:
├─ GPU: 8x A100 (312 TFLOPS each)
├─ NVLink: 600 GB/s interconnect
└─ Network: 400Gbps RoCE
```

### Training Workflow

```
1. Data Preparation
   ├─ Download from sources
   ├─ Tokenize using BPE
   ├─ Create shards (1 per worker)
   └─ Store in fast cache (SSD)

2. Model Initialization
   ├─ Load from scratch or checkpoint
   ├─ Replicate to all workers
   └─ Initialize optimizer states

3. Training Loop (for each epoch)
   for batch in dataloader:
   ├─ Forward pass (compute loss)
   ├─ Backward pass (compute gradients)
   ├─ All-reduce gradients across workers
   ├─ Weight update via optimizer
   └─ Log metrics & checkpoint if needed

4. Validation & Evaluation
   ├─ Run evaluation every N steps
   ├─ Compute perplexity on validation set
   ├─ Save best checkpoint
   └─ Early stopping if no improvement
```

---

## Monitoring & Observability

### Key Metrics to Track

**Training Metrics**:
- Loss (smoothed and raw)
- Learning rate (scheduler)
- Gradient norm (health check)
- Throughput (examples/sec, tokens/sec)

**System Metrics**:
- GPU utilization (%)
- GPU memory used
- CPU usage
- Network bandwidth
- Disk I/O

**Application Metrics** (Inference):
- Requests per second
- Latency (p50, p95, p99)
- Error rate
- Cache hit rate
- Queue depth

### Monitoring Stack

```yaml
Prometheus:
  ├─ Metrics collection
  ├─ Time-series storage
  └─ 15-day retention

Grafana:
  ├─ Dashboard visualization
  ├─ Custom panels
  └─ Alert rules

Weights & Biases:
  ├─ Training visualization
  ├─ Hyperparameter tracking
  └─ Model versioning
```

---

## Database & Storage

### Data Organization

```
/data/
├─ raw/                           (Original sources)
│  ├─ wikipedia/
│  ├─ arxiv/
│  ├─ web_crawl/
│  └─ banso_community/
│
├─ processed/                      (Tokenized, cleaned)
│  ├─ train_shards/
│  │  ├─ shard_0.bin
│  │  ├─ shard_1.bin
│  │  └─ ...
│  ├─ val/
│  └─ test/
│
└─ indices/                        (Fast lookup)
   ├─ train_index.json
   └─ vocab.json
```

### Storage Backend

```
Optimal setup:
├─ Hot   (SSD, NVMe): Current shard + index (100GB)
├─ Warm  (Network NAS): All shards (5TB)
└─ Cold  (Cloud bucket): Archives (S3/GCS)

Performance:
- SSD random read: 100k IOPS
- Network NAS: 1GB/s
- Allows prefetching between shards
```

---

## Scaling Strategies

### Horizontal Scaling (Multiple GPUs/Machines)

```python
# Distributed Data Parallel (DDP)
from torch.nn.parallel import DistributedDataParallel

model = AutoModel.from_pretrained("markgpt-base")
model = DistributedDataParallel(model, device_ids=[rank])

# Data sharding
train_data_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=DistributedSampler(train_dataset),  # Auto-shard
    num_workers=4,
    pin_memory=True,
)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_data_loader):
        # Forward, backward, optimize
        ...
```

### Vertical Scaling (Single GPU Optimization)

```python
# Memory optimizations
model.gradient_checkpointing_enable()  # Trade compute for memory
torch.set_float32_matmul_precision('medium')  # Speed vs precision
model = torch.compile(model)  # Graph optimization

# Batch size search
from torch_batch_finder import find_optimal_batch_size
optimal_bs = find_optimal_batch_size(model, max_memory_gb=16)
```

---

## Disaster Recovery

### High Availability

```
Production Setup:
├─ Primary Inference Cluster     (Primary traffic)
├─ Standby Inference Cluster     (Fast failover)
├─ Cross-region replication      (Geo-redundancy)
└─ Database backup → Cloud Storage
```

### Backup & Recovery

```bash
# Daily backup of:
- Model checkpoints (incremental)
- Training logs
- Configuration

# Recovery procedure:
1. Identify last good checkpoint
2. Create temporary training instance
3. Load and validate checkpoint
4. Resume training or serve model
5. Monitor metrics return to normal
```

---

## Security Architecture

### API Security

```python
from fastapi.security import HTTPBearer, HTTPAuthenticationCredentials

security = HTTPBearer()

@app.post("/v1/completions")
async def create_completion(
    request: CompletionRequest,
    credentials: HTTPAuthenticationCredentials = Depends(security)
):
    # Verify token
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=403)
    
    # Rate limit check
    if not check_rate_limit(credentials.credentials):
        raise HTTPException(status_code=429)
    
    # Process request
    return await inference_engine.generate(request)
```

### Model Security

- **Input validation**: Detect and block adversarial inputs
- **Output filtering**: Remove sensitive information if requested
- **Audit logging**: Log all inferences for compliance
- **Access control**: Fine-grained permissions per user/organization

---

## Cost Optimization

### Infrastructure Costs

| Component | Daily Cost | Annual | Optimization |
|---|---|---|---|
| GPU (8x A100) | $240 | $87.6K | Use spot instances, reserved |
| Storage (5TB) | $5 | $1.8K | Compress, archive old data |
| Bandwidth | $20 | $7.3K | CDN, local caching |
| Monitoring | $10 | $3.6K | Sampling, retention limits |
| **TOTAL** | **$275** | **$100K** | Service model dependent |

### Cost Reduction

1. **Batch operations** (10-20% savings)
2. **Quantization** (2-4x throughput)
3. **Smart caching** (50%+ cache hits)
4. **Spot instances** (70% cheaper clouds)
5. **Reserved capacity** (30-40% discounts)

---

**Architecture Version**: 1.0
**Last Updated**: 2024
