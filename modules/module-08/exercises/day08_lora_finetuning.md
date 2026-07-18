# Module 8 Exercise: LoRA Fine-tuning

## Objective
Implement and apply LoRA (Low-Rank Adaptation) to efficiently fine-tune a pre-trained model.

## Background

LoRA is a parameter-efficient fine-tuning technique that only trains small "adapter" matrices instead of all model parameters. It's used in production by Meta, OpenAI, and Microsoft because:
- 100-1000x fewer trainable parameters
- Faster training
- Works with any model architecture
- Adapters are composable (multiple LoRAs for different tasks)

## Part 1: Implement LoRALinear

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        
        # Original pretrained weight (frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.weight.requires_grad_(False)
        
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bias.requires_grad_(False)
        
        # LoRA adapter: W_delta = B @ A
        self.r = rank
        self.alpha = alpha
        
        # Down-project: in_features → rank
        self.A = nn.Linear(in_features, rank, bias=False)
        
        # Up-project: rank → out_features
        self.B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)  # Start with zero delta
    
    def forward(self, x):
        """
        x: (batch, in_features)
        Returns: (batch, out_features)
        """
        
        # Original: W @ x + b
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        
        # Add LoRA: B @ A @ x * (alpha / rank)
        lora_delta = (self.alpha / self.r) * self.B(self.A(x))
        out = out + lora_delta
        
        return out
    
    def merge(self):
        """Merge LoRA into main weights (for inference)"""
        # W_final = W + (alpha/r) * B @ A
        W_delta = (self.alpha / self.r) * self.B.weight @ self.A.weight
        self.weight.data.add_(W_delta)
        
        # Disable LoRA (no longer needed)
        self.A.requires_grad_(False)
        self.B.requires_grad_(False)

# Test
lora_linear = LoRALinear(in_features=10, out_features=5, rank=2)
x = torch.randn(3, 10)
y = lora_linear(x)
print(f"Output shape: {y.shape}")  # (3, 5)
```

## Part 2: Convert Model to LoRA

```python
def inject_lora(model, modules_to_adapt=["c_attn", "c_proj"], rank=8):
    """Replace Linear layers in specific modules with LoRA"""
    
    lora_count = 0
    
    for name, module in model.named_modules():
        if not any(adapter in name for adapter in modules_to_adapt):
            continue
        
        # Replace Linear with LoRALinear
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                lora_linear = LoRALinear(
                    child.in_features,
                    child.out_features,
                    rank=rank
                )
                
                # Copy original weights
                lora_linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    lora_linear.bias.data.copy_(child.bias.data)
                
                setattr(module, child_name, lora_linear)
                lora_count += 1
    
    return lora_count

# Use with pre-trained model
pretrained_model = load_pretrained_model("gpt2")
lora_count = inject_lora(pretrained_model, modules_to_adapt=["c_attn"], rank=8)
print(f"Converted {lora_count} layers to LoRA")

# Now freeze all original weights
for param in pretrained_model.parameters():
    param.requires_grad_(False)

# Only LoRA parameters are trainable
for module in pretrained_model.modules():
    if isinstance(module, LoRALinear):
        module.A.weight.requires_grad_(True)
        module.B.weight.requires_grad_(True)
```

## Part 3: Fine-tuning on Bible Data

```python
def fine_tune_with_lora(model, train_dataloader, num_epochs=3, lr=1e-3):
    """Fine-tune model using LoRA"""
    
    # Only optimize LoRA parameters
    lora_params = [
        p for m in model.modules() 
        if isinstance(m, LoRALinear)
        for p in [m.A.weight, m.B.weight] if p.requires_grad
    ]
    
    optimizer = torch.optim.AdamW(lora_params, lr=lr)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_dataloader:
            input_ids = batch["input_ids"]
            targets = batch["labels"]
            
            # Forward
            logits = model(input_ids)
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.shape[-1]),
                targets.view(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")
    
    return model

# Fine-tune on Bible
fine_tune_with_lora(model, bible_dataloader, num_epochs=3)
```

## Part 4: Merging and Inference

```python
def merge_lora_and_save(model, save_path):
    """Merge LoRA into weights and save (for deployment)"""
    
    # Merge all LoRA modules
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    
    # Now model is same size and speed as original
    # (LoRA overhead eliminated)
    
    torch.save(model.state_dict(), save_path)

# Deploy
merge_lora_and_save(model, "bible_gpt2_finetuned.pt")

# Load and use
model_deploy = load_pretrained_model("gpt2")
model_deploy.load_state_dict(torch.load("bible_gpt2_finetuned.pt"))
model_deploy.eval()

# Generate Bible-style text
text = generate(model_deploy, prompt="In the beginning", max_len=100)
```

## Part 5: Composing Multiple LoRAs

One of LoRA's superpowers: you can have multiple adapters for different tasks!

```python
def apply_multiple_loras(model, lora_checkpoints):
    """Apply multiple LoRA adapters for different languages"""
    
    lora_modules = {
        "english": {},
        "banso": {},
    }
    
    # Load English LoRA
    inject_lora(model, rank=8)
    restore_checkpoint(model, "lora_english.pt")
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_modules["english"][id(module)] = (module.A.weight.clone(), module.B.weight.clone())
    
    # Swap to Banso LoRA
    restore_checkpoint(model, "lora_banso.pt")
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_modules["banso"][id(module)] = (module.A.weight.clone(), module.B.weight.clone())
    
    # Switch between languages
    def set_language(lang):
        for module in model.modules():
            if isinstance(module, LoRALinear):
                A_weight, B_weight = lora_modules[lang][id(module)]
                module.A.weight.data.copy_(A_weight)
                module.B.weight.data.copy_(B_weight)
    
    return set_language

# Use
switch_lang = apply_multiple_loras(model, {"english": "...", "banso": "..."})

# Generate English
switch_lang("english")
en_text = generate(model, "In the beginning")

# Generate Banso
switch_lang("banso")
banso_text = generate(model, "Fə́ gǔ")

# Same base model, different capabilities!
```

## Challenge: Compare LoRA vs. Full Fine-tuning

```python
def compare_finetuning_methods():
    """Memory and speed comparison"""
    
    import time
    
    # Method 1: LoRA
    model_lora = load_pretrained_model("gpt2")
    inject_lora(model_lora, rank=8)
    
    start = time.time()
    fine_tune_with_lora(model_lora, bible_dataloader, num_epochs=1)
    lora_time = time.time() - start
    lora_params = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
    
    # Method 2: Full fine-tuning
    model_full = load_pretrained_model("gpt2")
    for p in model_full.parameters():
        p.requires_grad_(True)  # Train everything
    
    start = time.time()
    train_full_network(model_full, bible_dataloader, num_epochs=1)
    full_time = time.time() - start
    full_params = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    
    # Results
    print(f"LoRA:")
    print(f"  Time: {lora_time:.1f}s")
    print(f"  Trainable params: {lora_params:,}")
    print(f"Full fine-tuning:")
    print(f"  Time: {full_time:.1f}s")
    print(f"  Trainable params: {full_params:,}")
    print(f"Speedup: {full_time/lora_time:.1f}x")
    print(f"Parameter reduction: {full_params/lora_params:.0f}x")
```

## Key Insights

- ✅ LoRA: Only ~1% of parameters trainable
- ✅ Speed: 2-10x faster than full fine-tuning
- ✅ Memory: ~50-70% less than full fine-tuning
- ✅ Quality: ~95% of full fine-tuning performance
- ✅ Composability: Multiple adapters for different tasks

## References

- Hu, J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
- QLoRA: https://github.com/artidoro/qlora (quantized LoRA, even more efficient)
