# API Reference

## Overview

Complete API reference for MarkGPT Python modules.

## Core Model API

### `class MarkGPT`

**Main transformer model class**

```python
from src.model.markgpt import MarkGPT, MarkGPTConfig

# Initialize with config
config = MarkGPTConfig(
    vocab_size=10000,
    d_model=512,
    num_layers=12,
    num_heads=8,
    d_ff=2048,
    flash_attention=True
)
model = MarkGPT(config)

# Or load pre-trained
model = MarkGPT.load_pretrained('markgpt-small')
```

#### Methods

**`forward(input_ids, attention_mask=None) -> Tensor`**
- Forward pass through model
- Returns: logits of shape (batch, seq_len, vocab_size)

**`generate(prompt_ids, max_length=100, temperature=1.0, top_k=50) -> Tensor`**
- Auto-regressive text generation
- Returns: token IDs of shape (batch, max_length)

**`save_pretrained(path: str)`**
- Save model weights and config

**`load_pretrained(cls, path: str) -> MarkGPT`**
- Class method to load pre-trained model

### `class MarkGPTConfig`

**Configuration dataclass**

```python
@dataclass
class MarkGPTConfig:
    vocab_size: int = 10000
    d_model: int = 512
    num_layers: int = 12
    num_heads: int = 8
    d_ff: int = 2048
    max_seq_length: int = 1024
    dropout: float = 0.1
    flash_attention: bool = True
    use_causal_mask: bool = True
```

## Tokenizer API

### `class Tokenizer`

```python
from src.tokenizer.tokenizer import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.load('markgpt_vocab.pkl')

# Or train from scratch
tokenizer = Tokenizer(vocab_size=10000)
tokenizer.train('corpus.txt')
tokenizer.save('markgpt_vocab.pkl')
```

#### Methods

**`encode(text: str) -> List[int]`**
- Convert text to token IDs

**`decode(token_ids: List[int]) -> str`**
- Convert token IDs back to text

**`get_vocab_size() -> int`**
- Return vocabulary size

**`save(path: str)`**
- Persist tokenizer to disk

### `class BPETokenizer`

**Byte-pair encoding tokenizer subclass**

```python
from src.tokenizer.bpe import BPETokenizer

bpe = BPETokenizer(vocab_size=10000)
bpe.train_from_corpus('corpus.txt')

# Analyze token distribution
fertility = bpe.analyze_fertility(corpus)  # tokens per word
print(f"Average fertility: {fertility}")
```

## Training API

### `class Trainer`

```python
from src.training.trainer import Trainer
from src.model.markgpt import MarkGPT

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=5e-4,
    num_epochs=10,
    mixed_precision=True
)

trainer.fit()
checkpoint = trainer.get_checkpoint()
```

#### Methods

**`fit() -> Dict`**
- Train model; returns metrics dict

**`evaluate(loader) -> Dict`**
- Run evaluation; returns loss, perplexity

**`save_checkpoint(path: str)`**
- Save training state

**`load_checkpoint(path: str)`**
- Resume from checkpoint

### `class EarlyStopping`

```python
from src.training.training_utils import EarlyStopping

early_stop = EarlyStopping(
    patience=5,
    min_delta=0.001,
    checkpoint_dir='checkpoints/'
)

for epoch in range(100):
    val_loss = train_epoch()
    if early_stop(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

## Inference API

### RESTful API (Flask)

```bash
# Start server
python -m src.inference.api --port 8000

# Example request
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "John 3:16",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

#### Endpoints

**`POST /v1/completions`**
- Text completion
- **Parameters**:
  - `prompt` (str): Input text
  - `max_tokens` (int): Max output length
  - `temperature` (float): Sampling temperature (0-2)
  - `top_k` (int): Top-k sampling
- **Returns**: `{"text": "...", "tokens": 25}`

**`POST /v1/embeddings`** (if enabled)
- Get sentence embeddings
- **Parameters**:
  - `text` (str): Input text
- **Returns**: `{"embedding": [...]}`

### Python Inference Client

```python
from src.inference.client import MarkGPTClient

client = MarkGPTClient(url='http://localhost:8000')

# Completion
response = client.complete(
    prompt="And God said,",
    max_tokens=50,
    temperature=0.7
)
print(response['text'])

# Batch
responses = client.complete_batch(
    prompts=["John 3:16", "Genesis 1"],
    max_tokens=50
)
```

## Dataset API

### `class BibleDataset`

```python
from src.utils.datasets import BibleDataset

dataset = BibleDataset(
    path='data/bible.txt',
    tokenizer=tokenizer,
    seq_length=1024,
    language='en'
)

# Access like normal torch Dataset
sample = dataset[0]
print(sample['input_ids'], sample['target_ids'])
```

### `class MultilingualDataset`

```python
from src.utils.datasets import MultilingualDataset

dataset = MultilingualDataset(
    en_path='data/bible_en.txt',
    banso_path='data/bible_banso.txt',
    tokenizer=tokenizer,
    seq_length=1024,
    mix_ratio=0.7  # 70% English, 30% Banso
)
```

## Utilities API

### `evaluate_model(model, loader, device) -> Dict`

```python
from src.utils.eval import evaluate_model

metrics = evaluate_model(model, test_loader, device='cuda')
print(f"Perplexity: {metrics['perplexity']:.2f}")
print(f"Loss: {metrics['loss']:.4f}")
```

### `print_model_summary(model)`

```python
from src.utils.model_utils import print_model_summary

print_model_summary(model)
# Output:
# MarkGPT(
#   vocab: 10000
#   d_model: 512
#   layers: 12
#   params: 50M
#   memory: 200MB
# )
```

### `get_gpu_memory_usage(model)`

```python
from src.utils.gpu_utils import get_gpu_memory_usage

used_gb, total_gb = get_gpu_memory_usage(model)
print(f"GPU: {used_gb:.2f}GB / {total_gb:.2f}GB")
```

## Configuration Files

### YAML Config

```yaml
# configs/markgpt_small.yaml
model:
  name: markgpt-small
  vocab_size: 10000
  d_model: 512
  num_layers: 12
  num_heads: 8
  d_ff: 2048
  flash_attention: true

training:
  batch_size: 64
  learning_rate: 5.0e-4
  num_epochs: 10
  warmup_steps: 1000
```

### Python Config

```python
from dataclasses import dataclass
from src.model.markgpt import MarkGPTConfig

config = MarkGPTConfig(
    vocab_size=10000,
    d_model=512,
    num_layers=12,
    num_heads=8
)

# Access fields
print(config.d_model)  # 512
```

## Type Hints

All MarkGPT functions use type hints:

```python
from typing import Dict, List, Optional, Tuple
import torch

def train_step(
    batch: Dict[str, torch.Tensor],
    model: MarkGPT,
    optimizer: torch.optim.Optimizer
) -> Tuple[float, torch.Tensor]:
    """Train one batch."""
    pass
```

## Error Handling

```python
try:
    model = MarkGPT.load_pretrained('nonexistent')
except FileNotFoundError as e:
    print(f"Model not found: {e}")

try:
    tokens = tokenizer.encode(text)
except ValueError as e:
    print(f"Tokenization failed: {e}")
```

## Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("Training started")
logger.warning("GPU memory low")
logger.error("Training failed")
```

---

**API Version**: 1.0.0
**Last Updated**: 2024
**Documentation**: Full docstrings in source code
