# Inference Patterns & Code Examples

## Pattern 1: Streaming Generation

Useful for gradual output display in real-time applications.

```python
import torch
from typing import Iterator, List

class StreamingGenerator:
    """Generate tokens one at a time for streaming output."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Iterator[str]:
        """
        Generate tokens one at a time.
        
        Yields:
            Generated tokens one at a time
        """
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        input_ids = input_ids.to(self.device)
        
        # Track past for efficiency
        past_key_values = None
        
        self.model.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                # Get next token
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-p filtering
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumsum_probs <= top_p
                sorted_mask[..., 0] = True  # Always include best
                
                sorted_logits[~sorted_mask] = -float('inf')
                probs = torch.softmax(sorted_logits, dim=-1)
                
                # Sample
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token)
                
                # Update input for next iteration
                input_ids = next_token
                
                # Decode and yield
                token_str = self.tokenizer.decode(next_token[0])
                yield token_str
                
                # Stop if EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

# Usage
generator = StreamingGenerator(model, tokenizer)

prompt = "The future of artificial intelligence is"
print(f"Prompt: {prompt}")
print("Generation: ", end="", flush=True)

for token in generator.stream_generate(prompt, max_new_tokens=50):
    print(token, end="", flush=True)
print()
```

## Pattern 2: Batch Inference with Dynamic Padding

Maximize throughput on variable-length inputs.

```python
def batch_inference(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 512,
) -> List[torch.Tensor]:
    """Efficient batched inference supporting variable lengths."""
    
    all_outputs = []
    
    model.eval()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize with dynamic padding
        inputs = tokenizer(
            batch_texts,
            max_length=max_length,
            padding='longest',  # Pad to longest in batch, not 512
            truncation=True,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            all_outputs.append(outputs.logits)
    
    return torch.cat(all_outputs, dim=0)

# Usage: Process 1000 variable-length documents
docs = [load_doc(i) for i in range(1000)]
outputs = batch_inference(model, tokenizer, docs, batch_size=32)
```

## Pattern 3: Ensemble Decoding

Combine multiple model predictions for robustness.

```python
class EnsembleDecoder:
    """Ensemble multiple models for robust predictions."""
    
    def __init__(self, models: List[torch.nn.Module], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.weights = torch.tensor(self.weights) / sum(self.weights)
    
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Average logits from multiple models."""
        
        ensemble_logits = None
        
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
                
                if ensemble_logits is None:
                    ensemble_logits = weight * logits
                else:
                    ensemble_logits += weight * logits
        
        return ensemble_logits

# Usage: Ensemble 3 models for critical application
models = [load_model(f'checkpoint_{i}') for i in range(3)]
weights = [0.5, 0.3, 0.2]  # Weighted by performance
ensemble = EnsembleDecoder(models, weights)

# Generate with ensemble
input_ids = tokenizer.encode("Hello, how are you?", return_tensors='pt')
ensemble_logits = ensemble(input_ids)
predictions = ensemble_logits.argmax(-1)
```

## Pattern 4: Efficient KV Cache Management

Reduce memory in long-context generation.

```python
class KVCacheManager:
    """Manage KV cache for efficient long sequences."""
    
    def __init__(self, max_cache_size: int = 4096):
        self.max_cache_size = max_cache_size
        self.cache = {}
    
    def forward_with_cache(
        self,
        model,
        input_ids: torch.Tensor,
        past_key_values = None,
    ):
        """Forward pass with KV cache reuse."""
        
        seq_len = input_ids.shape[1]
        
        # Use cached KV if available
        if past_key_values is not None and seq_len == 1:
            # Only new token, use cache
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
        else:
            # Full sequence or first pass
            outputs = model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True
            )
        
        # Manage cache size (sliding window)
        cache = outputs.past_key_values
        if cache is not None and self.max_cache_size:
            # Keep only recent tokens
            cache = tuple(
                (k[:, :, -self.max_cache_size:, :], v[:, :, -self.max_cache_size:, :])
                for k, v in cache
            )
        
        return outputs, cache

# Usage: Generate 2000 tokens with bounded memory
cache_mgr = KVCacheManager(max_cache_size=1024)
past_kv = None

for i in range(2000):
    input_ids = next_token.unsqueeze(0)
    outputs, past_kv = cache_mgr.forward_with_cache(model, input_ids, past_kv)
    next_token = outputs.logits.argmax(-1).squeeze()
```

## Pattern 5: Speculative Decoding

Speed up inference with draft model.

```python
class SpeculativeDecoder:
    """Use small draft model to speed up large model."""
    
    def __init__(self, draft_model, target_model, tokenizer, num_speculate: int = 5):
        self.draft = draft_model
        self.target = target_model
        self.tokenizer = tokenizer
        self.num_speculate = num_speculate
    
    def decode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Speculative decoding for faster inference."""
        
        output_ids = input_ids.clone()
        
        with torch.no_grad():
            # Use draft to generate multiple tokens
            draft_out = self.draft(input_ids.unsqueeze(0))
            draft_tokens = draft_out.logits[:, -1, :].argmax(-1)
            
            # Build candidate continuation
            candidates = torch.cat([
                input_ids,
                draft_tokens.unsqueeze(0).expand(self.num_speculate, 1)
            ], dim=-1)
            
            # Verify with target model
            target_out = self.target(candidates)
            target_probs = torch.softmax(target_out.logits[:, -1, :], dim=-1)
            
            # Accept if probabilities align
            draft_probs = torch.softmax(draft_out.logits[:, -1, :], dim=-1)
            
            accepted = (target_probs.argmax(-1) == draft_tokens)
            
            if accepted.any():
                # Use some draft predictions
                num_accept = accepted.sum()
                output_ids = torch.cat([output_ids, draft_tokens[:num_accept]])
            else:
                # Use target prediction
                output_ids = torch.cat([output_ids, target_probs.argmax(-1).unsqueeze(0)])
        
        return output_ids

# Usage: Faster inference with small + large model
decoder = SpeculativeDecoder(small_model, large_model, tokenizer)
tokens = decoder.decode(input_ids)
```

---

**Guide Version**: 1.0
**Last Updated**: 2024
