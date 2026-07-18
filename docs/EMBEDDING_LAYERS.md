# Embedding Layers & Tokenization

## Word Embeddings (Input Layer)

### Purpose

Convert discrete tokens (integers 0-9999) to continuous vectors (embeddings).

$$\text{embed}(i) = \mathbf{e}_i \in \mathbb{R}^{d_\text{model}}$$

Where $\mathbf{e}_i$ is learned during training.

### Implementation

```python
import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )
    
    def forward(self, input_ids):
        # input_ids: (batch, seq_len) with values in [0, vocab_size)
        # output: (batch, seq_len, d_model)
        return self.embedding(input_ids)

# MarkGPT example
embedding = EmbeddingLayer(vocab_size=10000, d_model=768)
token_ids = torch.tensor([[1, 42, 999, 501], [2, 45, 102, 999]])  # (batch=2, seq_len=4)
embedded = embedding(token_ids)  # (2, 4, 768)
```

### Memory Footprint

$$\text{Memory} = \text{vocab\_size} \times d_\text{model} \times 4 \text{ bytes}$$

For MarkGPT:
- vocab_size = 10,000
- d_model = 768
- Memory = 10,000 × 768 × 4 = **30.7 MB** per model

Not large, but matters for billions-parameter models.

---

## Tied Embeddings (Input = Output)

### Motivation

Sharing input and output embeddings usually works well:

```python
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = TransformerStack(d_model, num_layers)
        
        # Output projection: Tied to embedding
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.embedding.weight  # TIED
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits
```

### Tied vs Untied

| Property | Tied | Untied |
|----------|------|--------|
| Parameters | N | 2N |
| Memory | Lower | 1.53x |
| Performance | Often better* | Slightly worse |
| Regularization | Implicit | None |

*Better: Reduces overfitting, acts as implicit regularization.

### When NOT to Tie?

- Input and output have different vocabularies
- Different embedding dimensions needed
- Very large vocabularies (BigBird, etc.)

---

## Positional Encodings (Via Embedding Augmentation)

### Absolute Positional Encoding (Original)

Add position information to embeddings:

$$\text{emb}_i = \text{token\_emb}_i + \text{pos\_emb}_i$$

```python
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
    
    def forward(self, x, positions=None):
        # x: (batch, seq_len, d_model)
        if positions is None:
            positions = torch.arange(x.shape[1], device=x.device)
        
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb.unsqueeze(0)  # Broadcast to batch
```

**Issues**:
- Fixed max_seq_len
- Poor to longer sequences

### Rotary Position Embeddings (Modern)

Instead of adding, rotate embeddings (see ROPE_GUIDE.md)

```python
# MarkGPT uses RoPE, not additive positions
# Applied in self-attention, not in embedding layer
```

---

## Multiple Embedding Types

### Embedding Table Organization

```python
class MultiEmbedding(nn.Module):
    """Combine token + position embeddings (modern approach)."""
    
    def __init__(self, vocab_size, d_model, max_seq_len=2048):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Optional: ALiBi or other positional strategy
        # (RoPE is in attention, not here)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids):
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.dropout(x)  # Regularization
        return x
```

---

## Special Tokens & Vocabulary Design

### Standard Special Tokens

For MarkGPT (and most LLMs):

| Token | ID | Usage |
|-------|----|-|
| `<unk>` | 0 | Unknown character |
| `<pad>` | 1 | Padding (for variable lengths) |
| `<bos>` | 2 | Beginning of sequence |
| `<eos>` | 3 | End of sequence |
| Custom | 4+ | User tokens |
| Text | 256+ | Actual text characters/BPE tokens |

```python
class SpecialTokens:
    UNK = 0
    PAD = 1
    BOS = 2
    EOS = 3

# Usage
def prepare_sequence(text, tokenizer):
    # Tokenize text
    token_ids = [SpecialTokens.BOS]
    token_ids.extend(tokenizer.encode(text))
    token_ids.append(SpecialTokens.EOS)
    return token_ids
```

### Vocabulary Cutoff

```python
vocab_size = 10000  # MarkGPT

# If text token > vocab_size, map to UNK
def tokenize_with_unk(text, tokenizer, vocab_size):
    token_ids = tokenizer.encode(text)
    token_ids = [min(tok, vocab_size - 1) for tok in token_ids]
    return token_ids
```

---

## Embedding Initialization

### Uniform Initialization (Standard)

```python
def init_embeddings_uniform(embedding_layer, fan_in=None):
    """Initialize embeddings uniformly."""
    if fan_in is None:
        fan_in = embedding_layer.embedding_dim
    
    bound = 3 / (2 * math.sqrt(fan_in))
    nn.init.uniform_(embedding_layer.weight, -bound, bound)

# Usage
embedding = nn.Embedding(10000, 768)
init_embeddings_uniform(embedding)
```

### Gaussian Initialization (Better for Transformers)

```python
def init_embeddings_gaussian(embedding_layer):
    """Initialize with Gaussian, variance-preserving."""
    d = embedding_layer.embedding_dim
    std = math.sqrt(2.0 / d)
    nn.init.normal_(embedding_layer.weight, mean=0, std=std)

embedding = nn.Embedding(10000, 768)
init_embeddings_gaussian(embedding)
```

### Pre-trained Initialization

```python
# Transfer learning: Load embeddings from pre-trained model
from transformers import AutoModel

pretrained = AutoModel.from_pretrained("gpt2")
my_model.embedding.weight.data = pretrained.wte.weight.data.clone()
```

---

## Embedding Analysis & Visualization

### Distribution Check

```python
def analyze_embeddings(model):
    """Diagnose embedding layer health."""
    emb = model.embedding.weight  # (vocab_size, d_model)
    
    # Statistics
    mean = emb.mean().item()
    std = emb.std().item()
    max_val = emb.max().item()
    min_val = emb.min().item()
    
    print(f"Embeddings: mean={mean:.4f}, std={std:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
    
    # Check norms
    norms = torch.norm(emb, dim=1)
    print(f"Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}")
    
    # Orthogonality (should be random)
    similarity = torch.matmul(emb, emb.T) / (norms.unsqueeze(1) * norms.unsqueeze(0) + 1e-8)
    off_diag = similarity[~torch.eye(len(similarity), dtype=bool)]
    print(f"Off-diagonal correlations: mean={off_diag.mean():.6f}")
```

### Nearest Neighbors

```python
def find_similar_embeddings(embedding_layer, token_id, k=10):
    """Find most similar tokens by embedding."""
    embeddings = embedding_layer.weight  # (vocab_size, d_model)
    token_emb = embeddings[token_id]  # (d_model,)
    
    # Cosine similarity
    similarities = torch.cosine_similarity(
        token_emb.unsqueeze(0),
        embeddings,
        dim=1
    )
    
    top_k = torch.topk(similarities, k).indices
    return top_k.tolist()

# Usage
similar_tokens = find_similar_embeddings(model.embedding, token_id=42, k=5)
print(f"Similar to token 42: {similar_tokens}")
```

---

## Embedding Compression & Quantization

### Dynamic Quantization

```python
def quantize_embeddings(embedding_layer):
    """Convert embeddings to int8 for inference."""
    weight = embedding_layer.weight.data
    
    # Find min/max
    min_val = weight.min()
    max_val = weight.max()
    
    # Map to int8 [-128, 127]
    quantized = ((weight - min_val) / (max_val - min_val) * 255 - 128).to(torch.int8)
    
    # Store scale/shift for reconstruction
    return quantized, min_val, max_val

# Dequantization during inference
def dequantize_embeddings(quantized, min_val, max_val):
    return (quantized.float() + 128) / 255 * (max_val - min_val) + min_val
```

### Product Quantization (For Very Large Vocabulary)

```python
class ProductQuantizationEmbedding(nn.Module):
    """PQ for massive vocabularies."""
    
    def __init__(self, vocab_size, d_model, num_subspaces=8):
        super().__init__()
        self.num_subspaces = num_subspaces
        self.subspace_dim = d_model // num_subspaces
        
        # Each subspace has its own codebook
        self.codebooks = nn.ModuleList([
            nn.Embedding(256, self.subspace_dim)  # 256 codewords per subspace
            for _ in range(num_subspaces)
        ])
        
        # Indices: (vocab_size, num_subspaces) each element in [0, 255]
        self.register_buffer(
            'indices',
            torch.randint(0, 256, (vocab_size, num_subspaces))
        )
    
    def forward(self, input_ids):
        # Lookup indices
        token_indices = self.indices[input_ids]  # (..., num_subspaces)
        
        # Gather from codebooks
        embeddings = []
        for i, codebook in enumerate(self.codebooks):
            embeddings.append(codebook(token_indices[..., i]))
        
        # Concatenate
        return torch.cat(embeddings, dim=-1)
```

Memory savings: `vocab_size * d_model` → `vocab_size * log(256) * num_subspaces`

---

## MarkGPT Embedding Configuration

```python
class MarkGPTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size  # 10,000
        self.d_model = config.d_model        # 768
        
        # Token embeddings
        self.token_emb = nn.Embedding(
            self.vocab_size,
            self.d_model,
            padding_idx=config.pad_token_id
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Initialize
        nn.init.normal_(self.token_emb.weight, mean=0, std=config.d_model ** -0.5)
        if config.pad_token_id is not None:
            with torch.no_grad():
                self.token_emb.weight[config.pad_token_id].zero_()
    
    def forward(self, input_ids):
        x = self.token_emb(input_ids)  # (batch, seq_len, d_model)
        x = self.dropout(x)
        return x
```

---

**Embedding Layers v1.0**
**Last Updated**: 2024
