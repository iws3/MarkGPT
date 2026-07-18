# Loss Functions for Language Models

## Cross-Entropy Loss (Standard)

### Definition

For language modeling (next-token prediction):

$$L_{\text{CE}} = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)$$

Where:
- $y_i$ = true one-hot vector (1 at true token, 0 elsewhere)
- $\hat{y}_i = \text{softmax}(\text{logits}_i)$ = predicted probability distribution

### Sum Over Sequences

In practice, sum over batch and sequence:

$$L = -\frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{t=1}^{T} \log P(y_t^{(b)} | \text{context})$$

Where $B$ = batch size, $T$ = sequence length.

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

# Method 1: Direct
logits = model(input_ids)  # (batch, seq_len, vocab_size)
targets = input_ids[:, 1:]  # Shift right (we predict next token)

loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),  # Flatten: (batch*seq_len, vocab_size)
    targets.reshape(-1),                      # Flatten: (batch*seq_len,)
    reduction='mean'
)

# Method 2: Manual
log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
true_log_probs = torch.gather(log_probs, -1, targets.unsqueeze(-1))  # Pick true token
loss = -true_log_probs.mean()
```

### Perplexity Metric

$$\text{Perplexity} = e^{\text{Cross-Entropy Loss}}$$

Lower is better. Interpretable as "effective vocabulary size the model is confused about".

```python
def compute_perplexity(loss):
    """Convert loss to perplexity."""
    return torch.exp(loss).item()

# Example: loss=3.5 → perplexity = e^3.5 ≈ 33.1
```

---

## Focal Loss (Hard Negative Mining)

### Motivation

Cross-entropy treats all examples equally:

```
Loss for P(y)=0.9 (easy, correct): -log(0.9) = 0.105
Loss for P(y)=0.1 (hard, wrong):   -log(0.1) = 2.303
Ratio: 22x difference, but CE scales equally
```

Focal loss down-weights easy examples:

$$L_{\text{focal}} = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)$$

Where:
- $p_t$ = model probability for true class
- $(1 - p_t)^{\gamma}$ = modulationfactor (down-weight easy examples)
- $\gamma$ = focusing parameter (typically 2)
- $\alpha_t$ = class weight

### Implementation for LLMs

```python
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        # logits: (batch, seq_len, vocab_size)
        # targets: (batch, seq_len)
        
        # Cross-entropy with probability
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction='none'
        )
        
        # Get probability of true class
        probs = torch.exp(-ce_loss)
        
        # Focal term: down-weight easy examples
        focal_weight = (1 - probs) ** self.gamma
        
        # Apply focal weight
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()

# Usage
criterion = FocalLoss(gamma=2.0)
loss = criterion(logits, targets)
```

### When to Use Focal Loss?

- Training data has long tail (some tokens much rarer)
- Model overfits to common patterns
- MarkGPT: Optional for Banso-heavy training (less common tokens)

---

## Label Smoothing

### Motivation

Hard targets (y=1) can cause overfitting:

```
Standard:      P(y=true) → 1,    P(y≠true) → 0
Label smoothed: P(y=true) → 0.9, P(y≠true) → 0.1/(vocab_size-1)
```

Regularizes by smoothing target distribution.

### Formula

$$L_{\text{LS}} = -\sum_i [(1 - \epsilon) y_i + \epsilon / K] \log \hat{y}_i$$

Where $\epsilon$ = smoothing factor (usually 0.1), $K$ = number of classes.

### Implementation

```python
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, vocab_size, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits, targets):
        # logits: (batch, seq_len, vocab_size)
        # targets: (batch, seq_len)
        
        log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
        
        # Smooth targets
        target_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
        target_dist.scatter_(-1, targets.unsqueeze(-1), self.confidence)
        
        # KL divergence: smooth_target || pred
        loss = -(target_dist * log_probs).sum(dim=-1)
        
        return loss.mean()

# Usage
criterion = LabelSmoothingLoss(vocab_size=10000, smoothing=0.1)
loss = criterion(logits, targets)
```

### PyTorch Built-in

```python
# PyTorch 1.10+
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
```

---

## Contrastive Loss (Contrastive Learning)

Beyond next-token classification, enforce representation similarity.

### NTXent Loss (InfoNCE)

$$L = -\log \frac{e^{\cos(z_i, z_j) / \tau}}{\sum_{k} e^{\cos(z_i, z_k) / \tau}}$$

Where:
- $z_i$, $z_j$ = embeddings of positive pair
- $\tau$ = temperature
- Denominator = all other pairs

Used for contrastive pre-training (not standard for LLMs, but emerging).

```python
def contrastive_loss(z_i, z_j, temperature=0.07):
    """NT-Xent loss for contrastive learning."""
    # z_i, z_j: (batch_size, d_model)
    
    # Normalize
    z_i_norm = F.normalize(z_i, dim=-1)
    z_j_norm = F.normalize(z_j, dim=-1)
    
    # Similarity matrix (batch_size, batch_size)
    similarities = torch.matmul(z_i_norm, z_j_norm.T) / temperature
    
    # Labels: diagonal is positive pair
    labels = torch.arange(z_i.shape[0])
    
    # Cross-entropy: predict which j matches each i
    loss_i_j = F.cross_entropy(similarities, labels)
    loss_j_i = F.cross_entropy(similarities.T, labels)
    
    return (loss_i_j + loss_j_i) / 2
```

---

## Vocabulary-Weighted Cross-Entropy

### Motivation

Some tokens are harder to predict (rare words, punctuation).

Weight by inverse frequency:

$$w_i = \frac{1}{\text{freq}_i}$$

Then loss becomes:

$$L = -\frac{1}{B \cdot T} \sum_b \sum_t w_{y_t^{(b)}} \log P(y_t)$$

### Implementation

```python
def compute_token_weights(dataset, vocab_size):
    """Compute weights based on token frequency."""
    counts = torch.zeros(vocab_size)
    
    for batch in dataset:
        input_ids = batch
        counts.add_(torch.bincount(input_ids.reshape(-1), minlength=vocab_size))
    
    # Inverse frequency weight
    weights = 1.0 / (counts + 1)  # +1 to avoid division by zero
    weights = weights / weights.sum() * vocab_size  # Normalize
    
    return weights

# Usage
weights = compute_token_weights(train_dataset, vocab_size)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
```

### For Multilingual (MarkGPT)

Weight English vs Banso differently:

```python
def multilingual_weights(vocab_size, english_freq, banso_freq, english_weight=1.0, banso_weight=1.5):
    """Weight Banso tokens higher (rarer)."""
    weights = torch.ones(vocab_size)
    weights[:english_freq] = english_weight
    weights[english_freq:english_freq+banso_freq] = banso_weight
    return weights / weights.sum() * vocab_size
```

---

## Loss Scaling (Mixed Precision)

When training with fp16, gradients can get very small (underflow).

### Gradient Scaling

```python
scaler = torch.cuda.amp.GradScaler()

for batch in loader:
    with torch.cuda.amp.autocast():
        logits = model(batch)
        loss = criterion(logits, targets)
    
    # Scale loss to prevent underflow
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

---

## MarkGPT Loss Configuration

```python
class MarkGPTLossConfig:
    # Standard
    loss_type = "cross_entropy"
    
    # Regularization
    label_smoothing = 0.1      # Slight regularization
    vocab_weight = False       # Don't weight by frequency
    gradient_clip_norm = 1.0   # Prevent exploding gradients
    
    # Multi-lingual consideration
    # Don't weight Banso heavily (training curriculum handles balance)
    
    # Loss tracking
    track_perplexity = True
    log_interval = 100  # Log every 100 steps

# Usage
def compute_loss(logits, targets, config):
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing,
        reduction='mean'
    )
    
    return loss_fn(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1)
    )
```

---

## Loss Debugging

```python
def diagnose_loss(loss, prev_loss=None):
    """Check for loss anomalies."""
    
    # 1. NaN check
    if torch.isnan(loss):
        print("❌ NaN loss!")
        return False
    
    # 2. Unexpected jump
    if prev_loss is not None and loss > 2 * prev_loss:
        print(f"⚠️ Loss spike: {prev_loss:.3f} → {loss:.3f}")
    
    # 3. Not decreasing
    if loss > 10.0:  # For vocab_size=10000
        print(f"⚠️ High loss ({loss:.2f}), check:")
        print("   - Data loading")
        print("   - Learning rate (too high?)")
        print("   - Model initialization")
    
    return True

# In training loop
for epoch in epochs:
    for step, batch in enumerate(loader):
        loss = train_step(batch)
        if not diagnose_loss(loss, prev_loss):
            break
        prev_loss = loss
```

---

**Loss Functions v1.0**
**Last Updated**: 2024
