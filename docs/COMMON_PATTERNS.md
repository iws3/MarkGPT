# Common Patterns & Best Practices

## Training Patterns

### Pattern 1: Warmup + Cosine Decay

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=1000,  # Restart period in steps
    T_mult=1,  # Don't grow restart period
    eta_min=1e-5
)

for step, batch in enumerate(train_loader):
    loss = train_step(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

### Pattern 2: Gradient Clipping

```python
optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.zero_grad()
```

**When to use**: 
- Transformer training (prone to gradient explosion)
- Multilingual training (different scales across languages)
- Long sequences (vanishing/exploding gradients)

### Pattern 3: Model Checkpointing

```python
def save_checkpoint(model, optimizer, epoch, metrics):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, f'ckpt_epoch_{epoch}.pt')

def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt['epoch'], ckpt['metrics']

# Usage
for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss = validate()
    
    if val_loss < best_loss:
        save_checkpoint(model, optimizer, epoch, {'train': train_loss, 'val': val_loss})
        best_loss = val_loss
```

## Evaluation Patterns

### Pattern: Per-Language Evaluation

```python
def evaluate_per_language(model, eval_data):
    """Evaluate separately by language for multilingual models."""
    
    results = {}
    
    for lang, dataset in eval_data.items():
        loader = DataLoader(dataset, batch_size=32)
        
        total_loss = 0
        for batch in loader:
            with torch.no_grad():
                logits = model(batch['input_ids'])
                loss = criterion(logits.view(-1, vocab_size), batch['labels'].view(-1))
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        ppl = math.exp(avg_loss)
        results[lang] = {'loss': avg_loss, 'perplexity': ppl}
    
    return results

# Usage
metrics = evaluate_per_language(model, {
    'en': english_dataset,
    'banso': banso_dataset
})
print(f"English PPL: {metrics['en']['perplexity']:.2f}")
print(f"Banso PPL: {metrics['banso']['perplexity']:.2f}")
```

### Pattern: Confidence Intervals

```python
import numpy as np
from scipy import stats

def bootstrap_ci(model, dataset, n_bootstraps=100, ci=0.95):
    """Estimate confidence intervals via bootstrap."""
    
    scores = []
    
    for _ in range(n_bootstraps):
        # Resample with replacement
        sample_indices = np.random.choice(len(dataset), len(dataset), replace=True)
        sample = Subset(dataset, sample_indices)
        loader = DataLoader(sample, batch_size=32)
        
        # Evaluate
        loss = evaluate(model, loader)
        scores.append(loss)
    
    # Compute CI
    alpha = 1 - ci
    lower = np.percentile(scores, alpha/2 * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)
    
    return lower, upper

# Usage
lower, upper = bootstrap_ci(model, test_dataset)
print(f"Perplexity: {np.mean([exp(l) for l in [lower, upper]]):.2f}")
```

## Generation Patterns

### Pattern: Temperature Sampling

```python
def temperature_sample(logits, temperature=1.0, top_k=50):
    """Sample from logits with temperature scaling."""
    
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Top-K filtering
    top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k)
    
    # Softmax and sample
    probs = torch.softmax(top_k_logits, dim=-1)
    next_idx = torch.multinomial(probs, num_samples=1)
    
    return top_k_indices[next_idx]
```

### Pattern: Beam Search

```python
def beam_search(model, prompt, beam_width=5, max_length=100):
    """Deterministic beam search for generation."""
    
    # Initialize
    input_ids = tokenizer.encode(prompt)
    sequences = [(input_ids, 0.0)]  # (tokens, cumulative_log_prob)
    
    for _ in range(max_length):
        candidates = []
        
        for seq, score in sequences:
            if seq[-1] == eos_token:
                candidates.append((seq, score))
                continue
            
            # Get logits for next token
            with torch.no_grad():
                logits = model(torch.tensor([seq]))
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            
            # Get top-K continuations
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)
            
            for log_prob, idx in zip(top_log_probs, top_indices):
                new_seq = seq + [idx.item()]
                new_score = score + log_prob.item()
                candidates.append((new_seq, new_score))
        
        # Keep top beam_width
        sequences = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    return sequences[0][0]
```

## Data Pipeline Patterns

### Pattern: Dynamic Bucketing

```python
def dynamic_bucket_sampler(dataset, batch_size=32, num_buckets=10):
    """Create batches with similar-length sequences."""
    
    # Sort by length
    lengths = [len(example['input_ids']) for example in dataset]
    sorted_indices = np.argsort(lengths)
    
    # Create buckets
    bucket_size = len(dataset) // num_buckets
    batches = []
    
    for bucket_idx in range(num_buckets):
        bucket_start = bucket_idx * bucket_size
        bucket_end = (bucket_idx + 1) * bucket_size
        bucket_indices = sorted_indices[bucket_start:bucket_end]
        
        # Create batches from bucket
        for batch_start in range(0, len(bucket_indices), batch_size):
            batch_end = batch_start + batch_size
            batch = bucket_indices[batch_start:batch_end]
            batches.append(batch)
    
    return batches
```

### Pattern: Online Augmentation

```python
def augment_batch(batch, tokenizer, p_noise=0.01, p_drop=0.05):
    """Apply on-the-fly augmentation."""
    
    for example in batch:
        tokens = example['input_ids']
        
        # Random noise: swap words
        if random.random() < p_noise:
            i, j = random.sample(range(len(tokens)), 2)
            tokens[i], tokens[j] = tokens[j], tokens[i]
        
        # Random dropout: remove tokens
        if random.random() < p_drop:
            idx = random.randint(0, len(tokens)-1)
            tokens.pop(idx)
        
        example['input_ids'] = tokens
    
    return batch
```

## Monitoring Patterns

### Pattern: Log Scalars to W&B

```python
import wandb

config = {
    'learning_rate': 5e-4,
    'batch_size': 64,
    'num_epochs': 10
}
wandb.init(project='markgpt', config=config)

for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss = validate()
    
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_ppl': math.exp(val_loss)
    })
```

### Pattern: Moving Average

```python
class EMA:
    """Exponential moving average for smoothing metrics."""
    
    def __init__(self, decay=0.99):
        self.decay = decay
        self.ema = None
    
    def update(self, value):
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.decay * self.ema + (1 - self.decay) * value
        return self.ema

# Usage
ema_loss = EMA(decay=0.99)
for batch in train_loader:
    loss = train_step(batch)
    smooth_loss = ema_loss.update(loss.item())
    print(f"Smooth loss: {smooth_loss:.4f}")
```

---

**Patterns Version**: 1.0
**Last Updated**: 2024
