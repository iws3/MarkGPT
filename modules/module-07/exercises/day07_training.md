# Module 7 Exercise: Training a Transformer from Scratch

## Objective
Train a small transformer language model end-to-end using synthetic data.

## Part 1: Build Model

```python
import torch
import torch.nn as nn

class TransformerLM(nn.Module):
    """Decoder-only transformer language model"""
    
    def __init__(self, vocab_size=100, d_model=64, num_layers=2, num_heads=2):
        super().__init__()
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)  # Max seq len 512
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model*4,
            batch_first=True,
            norm_first=True  # Pre-norm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, targets=None):
        """
        input_ids: (batch, seq_len)
        Returns: logits (batch, seq_len, vocab_size) and optional loss
        """
        
        seq_len = input_ids.shape[1]
        
        # Embed tokens + positions
        x = self.embed(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embed(pos_ids)
        
        # Causal mask (can't look at future tokens)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        
        # Forward through transformer
        x = self.transformer(x, src_mask=causal_mask)
        
        # Project to vocab
        logits = self.lm_head(x)
        
        if targets is not None:
            # Compute loss
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.shape[-1]), targets.view(-1))
            return logits, loss
        
        return logits

# Test model
model = TransformerLM(vocab_size=50, d_model=64, num_layers=2)
input_ids = torch.randint(0, 50, (2, 10))  # Batch of 2, seq len 10
logits = model(input_ids)
print(f"Logits shape: {logits.shape}")  # Should be (2, 10, 50)
```

## Part 2: Create Training Data

```python
def create_synthetic_data(dataset_size=1000, vocab_size=50, seq_len=10):
    """Generate synthetic sequences for training"""
    
    # TODO: Generate random token sequences
    input_ids = torch.randint(0, vocab_size, (dataset_size, seq_len))
    
    # TODO: Targets are inputs shifted by 1 (next-token prediction)
    targets = torch.zeros_like(input_ids)
    targets[:, :-1] = input_ids[:, 1:]
    targets[:, -1] = torch.randint(0, vocab_size, (dataset_size,))
    
    return input_ids, targets

# Example
train_inputs, train_targets = create_synthetic_data(dataset_size=100, vocab_size=50)
print(f"Input shape: {train_inputs.shape}")      # (100, 10)
print(f"Target shape: {train_targets.shape}")    # (100, 10)
```

## Part 3: Training Loop

```python
def train_transformer():
    """Train transformer on synthetic data"""
    
    model = TransformerLM(vocab_size=50, d_model=64, num_layers=2, num_heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    train_inputs, train_targets = create_synthetic_data(dataset_size=1000)
    
    model.train()
    losses = []
    
    for epoch in range(10):
        # TODO: Iterate through batches
        for i in range(0, len(train_inputs), 32):  # batch_size=32
            batch_input = train_inputs[i:i+32]
            batch_target = train_targets[i:i+32]
            
            # TODO: Forward pass
            logits, loss = model(batch_input, batch_target)
            
            # TODO: Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # TODO: Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # TODO: Optimizer step
            optimizer.step()
            
            losses.append(loss.item())
        
        # Log every epoch
        avg_loss = np.mean(losses[-len(train_inputs)//32:])
        print(f"Epoch {epoch}: loss = {avg_loss:.4f}")
    
    return model, losses

model, losses = train_transformer()
```

## Part 4: Monitoring Training

```python
import matplotlib.pyplot as plt

# Plot training loss
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Training step')
plt.ylabel('Loss')
plt.title('Training Transformer on Synthetic Data')
plt.grid(True)
plt.show()

# What to look for:
# ✅ Smoothly decreasing (good training)
# ❌ Increasing (learning rate too high)
# ❌ Plateauing (learning rate too low, or model capacity limited)
# ❌ Noisy (batch size too small)
```

## Part 5: Inference

```python
def generate_text(model, prompt_ids, max_len=20, temperature=1.0):
    """Generate text given a prompt"""
    
    model.eval()
    generated = prompt_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_len):
            # Get next token probabilities
            logits = model(generated[:, -512:])  # Use only last 512 to avoid memory
            next_logits = logits[0, -1, :]  # Last token logits
            
            # TODO: Apply temperature (lower = more deterministic)
            next_logits = next_logits / temperature
            
            # TODO: Sample from distribution
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    return generated

# Generate
prompt = torch.tensor([[1, 2, 3]])  # Start with tokens 1, 2, 3
generated = generate_text(model, prompt, max_len=10, temperature=0.1)
print(generated)  # Random token IDs (no real language here, just synthetic!)
```

## Challenge: Compare Different Architectures

```python
def compare_architectures():
    """Compare different model sizes"""
    
    configs = {
        "tiny": {"d_model": 32, "num_layers": 1, "num_heads": 2},
        "small": {"d_model": 64, "num_layers": 2, "num_heads": 2},
        "medium": {"d_model": 128, "num_layers": 4, "num_heads": 4},
    }
    
    results = {}
    
    for name, config in configs.items():
        # Train model
        model = TransformerLM(**config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        # (training code here...)
        final_loss = 0.0  # After training
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        results[name] = {"loss": final_loss, "params": num_params}
    
    # Plot results
    names = list(results.keys())
    params = [results[n]["params"] for n in names]
    losses = [results[n]["loss"] for n in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.bar(names, params)
    ax1.set_ylabel('Number of parameters')
    ax2.bar(names, losses)
    ax2.set_ylabel('Final loss')
    plt.tight_layout()
    plt.show()
```

## Key Takeaways

- ✅ Transformers are straightforward to implement
- ✅ Causal masking is crucial for autoregressive models
- ✅ Gradient clipping prevents training instability
- ✅ Temperature controls sampling behavior
- ✅ Monitoring loss is essential for debugging

## References

- Vaswani et al. (2017). "Attention is All You Need." *NeurIPS*.
- PyTorch Transformer Documentation: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
