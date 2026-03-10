# Sampling & Generation Strategies

## Greedy Decoding (Baseline)

### Algorithm

At each step, pick highest probability token:

$$\hat{y}_t = \arg\max_i P(y_i | y_{<t})$$

```python
def greedy_decode(model, input_ids, max_length=100):
    """Generate text by always picking highest probability."""
    
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)  # (batch, seq_len, vocab_size)
        
        # Get last token's logits
        next_logits = logits[:, -1, :]  # (batch, vocab_size)
        
        # Greedy: argmax
        next_token = next_logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
        
        # Append and continue
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Check for EOS
        if (next_token == eos_token_id).all():
            break
    
    return input_ids
```

### Issues

- **Repetition**: Gets stuck in loops
- **Monotonous**: Always picks same high-probability token
- **Poor diversity**: No exploration

### When to Use

- Information retrieval (factual answers)
- Translation (faithful to source)
- Grammar-dependent tasks

---

## Temperature Sampling

### Concept

Adjust probability distribution "temperature":

$$P(y_i | \text{temp}) = \frac{e^{\text{logits}_i / T}}{\sum_j e^{\text{logits}_j / T}}$$

Where $T$ = temperature:
- $T = 1.0$: Standard softmax
- $T \to 0^+$: Sharper (more greedy)
- $T > 1$: Softer (more uniform)

### Intuition

```
Temperature=0.5 (sharp):
  Token A: 0.7, Token B: 0.2, Token C: 0.1
  → [0.89, 0.08, 0.03] (A even more likely)

Temperature=1.0 (standard):
  → [0.70, 0.20, 0.10] (unchanged)

Temperature=2.0 (soft):
  → [0.42, 0.33, 0.25] (more uniform)
```

### Implementation

```python
def temperature_sampling(logits, temperature=1.0, top_k=None, top_p=None):
    """Apply temperature and optional top-k/top-p filtering."""
    
    # Adjust by temperature
    scaled_logits = logits / temperature
    
    # Optional: Top-k filtering
    if top_k is not None:
        top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k)
        scaled_logits.fill_(-float('inf'))
        scaled_logits.scatter_(-1, top_k_indices, top_k_logits)
    
    # Optional: Top-p (nucleus) filtering
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_mask = cumsum_probs <= top_p
        sorted_mask[:, 0] = True  # Always include top-1
        
        scaled_logits.fill_(-float('inf'))
        scaled_logits.scatter_(-1, sorted_indices, sorted_logits)
        scaled_logits.masked_fill_(~sorted_mask, float('-inf'))
    
    # Sample
    probs = torch.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token

# Generation with temperature
def generate_with_temperature(model, input_ids, max_length=100, temperature=1.0):
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)
        
        next_logits = logits[:, -1, :]
        next_token = temperature_sampling(next_logits, temperature=temperature)
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    return input_ids
```

### Recommended Values

| Task | Temperature | Effect |
|------|-------------|--------|
| Code generation | 0.2-0.5 | Mostly deterministic |
| Q&A (factual) | 0.7 | Balanced |
| Creative writing | 1.0-1.2 | Diverse |
| Brainstorming | 1.5+ | Very exploratory |

---

## Top-K Sampling

### Algorithm

Only sample from top-k most probable tokens:

$$P(\hat{y}_t | y_{<t}) \propto \begin{cases}
P(y_i | y_{<t}) & \text{if } y_i \in \text{Top-K} \\
0 & \text{otherwise}
\end{cases}$$

```python
def top_k_sampling(logits, k=50):
    """Sample from top-K tokens only."""
    
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # Zero out other logits
    logits = torch.full_like(logits, float('-inf'))
    logits.scatter_(-1, top_k_indices, top_k_logits)
    
    # Sample
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token
```

### Issues

- Fixed k can be suboptimal
  - If distribution is concentrated: k too large includes garbage
  - If distribution is spread: k too small misses good tokens

---

## Top-P (Nucleus) Sampling

### Algorithm

Sample from smallest subset where cumulative probability ≥ p:

$$\text{Choose smallest subset } S \text{ such that } \sum_{i \in S} P(y_i) \geq p$$

Then renormalize and sample.

```python
def nucleus_sampling(logits, p=0.9):
    """Sample from top-p (nucleus) tokens."""
    
    # Sort by probability
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    
    # Cumulative sum
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Keep tokens up to cumulative probability p
    mask = cumsum_probs <= p
    mask[:, 0] = True  # Always include top-1
    
    # Zero out excluded tokens
    sorted_logits[~mask] = float('-inf')
    
    # Renormalize and sample
    probs = torch.softmax(sorted_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token

# Combine top-p with temperature
def nucleus_temperature_sampling(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    return nucleus_sampling(logits, p=p)
```

### Example

```
Probabilities: [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]

Top-p (p=0.8):
  0.5 + 0.3 = 0.8 ✓ (include)
  0.5 + 0.3 + 0.1 = 0.9 > 0.8 (exclude 0.1 and beyond)
  → Sample from [0.5, 0.3]
```

### Advantages

- **Dynamic**: Adjusts subset size based on distribution
- **Stable**: Works across different domains
- **Recommended**: Often better than top-k

---

## Beam Search

### Algorithm

Keep $b$ (beam width) best hypotheses at each step.

```
Step 0:        [THE]
                ↙  ↓  ↘
Step 1:   [A], [AN], [AND]  (top-3)
          ↙  ↓  ↘ ↙ ↓  ↘ ↙
Step 2:   Best 3 sequences of length 2
```

```python
def beam_search(model, input_ids, beam_width=5, max_length=50):
    """Decode using beam search."""
    
    batch_size = input_ids.shape[0]
    vocab_size = model.config.vocab_size
    
    # Initialize: (batch_size, seq_len, 1)
    sequences = input_ids.unsqueeze(-1)
    scores = torch.zeros(batch_size).unsqueeze(-1)  # Log-probabilities
    
    for _ in range(max_length):
        # Get logits for current sequences
        # (batch_size * beam_width, seq_len, vocab_size)
        logits = model(sequences.view(-1, sequences.shape[1]))
        
        # Get probabilities for next token
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        
        # Expand beam: (batch, beam, vocab_size)
        all_log_probs = scores.unsqueeze(-1) + log_probs.view(batch_size, -1, vocab_size)
        
        # Keep top-k
        # (batch, beam_width)
        top_scores, top_indices = torch.topk(
            all_log_probs.view(batch_size, -1),
            beam_width
        )
        
        # Reorder sequences
        # (implementation: track which sequences/tokens to keep)
        # Simplified (real beam search more complex)
    
    return sequences

# Simpler Alternative: Greedy + Reranking
def greedy_with_reranking(model, input_ids, num_candidates=3, max_length=50):
    """Generate multiple sequences, pick best."""
    
    best_sequence = None
    best_score = float('-inf')
    
    for _ in range(num_candidates):
        sequence = greedy_decode(model, input_ids, max_length)
        score = evaluate_sequence(sequence, model)  # LM loss as score
        
        if score > best_score:
            best_score = score
            best_sequence = sequence
    
    return best_sequence
```

### Trade-offs

| Method | Speed | Quality | Memory |
|--------|-------|---------|--------|
| Greedy | Fast | Low | Low |
| Sampling | Fast | Medium | Low |
| Beam (b=5) | Slower | Higher | 5x |
| Beam (b=10) | Much slower | Best | 10x |

---

## MarkGPT Generation Config

```python
class GenerationConfig:
    # Sampling
    temperature = 0.8
    top_p = 0.9
    top_k = None  # None = disabled
    
    # Stopping
    max_length = 512
    max_new_tokens = 256
    eos_token_id = 3
    pad_token_id = 1
    
    # Decoding
    do_sample = True  # Sampling vs greedy
    num_beams = 1    # Beam search width
    
    # Diversity
    diversity_penalty = 0.0  # Beam search only
    repetition_penalty = 1.2 # Penalize repeated tokens
    
    # Output
    output_scores = False
    return_dict_in_generate = True

# Usage
generated = model.generate(
    input_ids,
    max_length=512,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)
```

---

## Evaluation: BLEU, ROUGE, etc.

For machine translation (Banso-English):

```python
from nltk.translate.bleu_score import sentence_bleu

def evaluate_translation(hypothesis, reference):
    """BLEU score for translation quality."""
    
    hypothesis_tokens = hypothesis.split()
    reference_tokens = reference.split()
    
    bleu = sentence_bleu(
        [reference_tokens],
        hypothesis_tokens,
        weights=(0.25, 0.25, 0.25, 0.25)  # 1-4 gram weights
    )
    
    return bleu
```

---

**Generation & Sampling v1.0**
**Last Updated**: 2024
