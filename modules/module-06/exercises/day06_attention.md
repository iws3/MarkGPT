# Module 6 Exercise: Attention Mechanism Implementation

## Objective
Implement scaled dot-product attention to understand the core mechanism behind transformers.

## Part 1: Simple Attention

```python
import numpy as np
import torch
import torch.nn.functional as F

# Suppose we have:
# - A query: "What word should go next?" (embedding vector)
# - Keys: All previous words in sequence (embeddings)
# - Values: Information we retrieve for each key

# Example: predicting word after "the"
query = np.array([[0.1, 0.5, -0.2, 0.3]])        # (1, 4)  "What's next?"
keys = np.array([
    [0.2, 0.4, -0.1, 0.2],                        # (4, 4) "the"
    [0.3, 0.1, 0.5, 0.1],                         # (1, 4) "bible"
    [0.0, 0.6, 0.2, -0.3],                        # (1, 4) "speaks"
])
values = np.array([
    [1.0, 0.0, 0.0, 0.0],                         # Get "the" info
    [0.0, 1.0, 0.0, 0.0],                         # Get "bible" info
    [0.0, 0.0, 1.0, 0.0],                         # Get "speaks" info
])

# Step 1: Compute similarity (dot product) between query and each key
# query @ keys.T gives similarity scores
scores = query @ keys.T                            # (1, 3)
print(f"Scores (query·key): {scores}")

# Step 2: Scale by sqrt(d_k) to prevent saturation
d_k = keys.shape[-1]
scores = scores / np.sqrt(d_k)

# Step 3: Softmax to get attention weights (probability distribution)
attention_weights = F.softmax(torch.tensor(scores), dim=-1).numpy()
print(f"Attention weights: {attention_weights}")
# Should sum to 1.0

# Step 4: Weight values by attention
output = attention_weights @ values               # (1, 3) @ (3, 4) = (1, 4)
print(f"Output: {output}")
# This is a weighted average of the values!
```

## Part 2: Causal Masking (For Language Models)

Language models predict next token using **only previous tokens**, not future ones.

```python
def create_causal_mask(seq_len):
    """Create mask that prevents attending to future tokens"""
    
    # Lower triangular matrix (1 = can attend, 0 = cannot)
    mask = np.tril(np.ones((seq_len, seq_len)))
    # TODO: Convert to -inf values for softmax (masked positions become ~0)
    mask = np.where(mask == 1, 0, -np.inf)
    return mask

# Example with 4 tokens
seq_len = 4
mask = create_causal_mask(seq_len)
print(mask)
# Position 0 can only see position 0
# Position 1 can see positions 0-1
# Position 2 can see positions 0-2
# Position 3 can see positions 0-3

# Apply mask to scores before softmax
scores_masked = scores + mask
attention_weights = F.softmax(torch.tensor(scores_masked), dim=-1)
# Masked positions will have weight ~0 (can't attend)
```

## Part 3: Multi-Head Attention

Instead of one attention, use multiple "heads" to capture different types of relationships.

```python
def multi_head_attention(query, key, value, num_heads=2):
    """
    query, key, value: (seq_len, d_model)
    num_heads: Number of attention heads
    """
    
    d_model = query.shape[-1]
    d_k = d_model // num_heads
    
    # TODO: Reshape queries for multiple heads
    # From (seq, d_model) to (seq, num_heads, d_k)
    Q = query.reshape(-1, num_heads, d_k)
    K = key.reshape(-1, num_heads, d_k)
    V = value.reshape(-1, num_heads, d_k)
    
    # TODO: For each head, compute scaled dot-product attention
    outputs = []
    for h in range(num_heads):
        Q_h = Q[:, h, :]
        K_h = K[:, h, :]
        V_h = V[:, h, :]
        
        # Attention: softmax(Q @ K.T / sqrt(d_k)) @ V
        scores = Q_h @ K_h.T / np.sqrt(d_k)
        weights = F.softmax(torch.tensor(scores), dim=-1).numpy()
        out_h = weights @ V_h
        
        outputs.append(out_h)
    
    # TODO: Concatenate heads
    output = np.concatenate(outputs, axis=-1)
    return output

# Test
output = multi_head_attention(query, keys, values, num_heads=2)
print(f"Multi-head output shape: {output.shape}")
```

## Part 4: Visualizing Attention Patterns

```python
import matplotlib.pyplot as plt

# After training a model on Bible text:
model = load_trained_model()

sentence = "In the beginning was the Word"
tokens = tokenize(sentence)

# Get attention weights from first layer, first head
attentions = model.get_attention_weights(tokens)  # (num_layers, seq_len, seq_len)

# Plot for first layer
layer_idx = 0
attention_matrix = attentions[layer_idx]  # (seq_len, seq_len)

plt.figure(figsize=(8, 8))
plt.imshow(attention_matrix, cmap='hot')
plt.xlabel('Attending to position')
plt.ylabel('From position')
plt.title('Attention Pattern: "In the beginning was the Word"')
plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.yticks(range(len(tokens)), tokens)

# Add values in cells
for i in range(len(tokens)):
    for j in range(len(tokens)):
        plt.text(j, i, f'{attention_matrix[i, j]:.2f}',
                ha='center', va='center', color='white', fontsize=8)

plt.colorbar(label='Attention weight')
plt.tight_layout()
plt.show()

# What to look for:
# - Diagonal dominance: Model mainly looks at its own position
# - Column peaks: Certain positions (like nouns) collect attention
# - Row sparsity: Some positions attend to many vs. few places
```

## Challenge: Implement Attention Scratch to PyTorch

```python
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    """Attention mechanism as a PyTorch module"""
    
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch_size, seq_len, d_k)
        Returns: attention_output (same shape)
        """
        
        # TODO: Compute scores = Q @ K.T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # TODO: Apply mask if provided (prevent future attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # TODO: Softmax to get probabilities
        attention_weights = F.softmax(scores, dim=-1)
        
        # TODO: Weight values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
```

## Key Takeaways

- ✅ Attention = weighted average of values based on query-key similarity
- ✅ Softmax converts scores to probabilities
- ✅ Scaling prevents saturation in softmax
- ✅ Causal mask prevents looking at future tokens
- ✅ Multi-head allows different types of attention simultaneously

## References

- Vaswani et al. (2017). "Attention is All You Need." *NeurIPS*.
- Lin et al. (2022). "Transformers are Inefficient Language Models." *arXiv:2109.02143*.
