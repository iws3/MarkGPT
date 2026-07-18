# Fine-tuning Best Practices

## LoRA Fine-tuning Strategy

### Setup LoRA Adapter

```python
from peft import LoraConfig, get_peft_model

# Define LoRA config
lora_config = LoraConfig(
    r=8,  # Low rank
    lora_alpha=16,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which modules to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# output: trainable params: 123,456 || all params: 6,234,567 || trainable%: 1.98%
```

### Training with LoRA

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### Merging LoRA Weights

```python
# Merge LoRA adapter back into base model
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("./merged_model")

# For inference, just load merged model
merged_model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

## Domain Adaptation

### Multilingual Fine-tuning

```python
# Load base model
model = AutoModelForCausalLM.from_pretrained("markgpt-base")

# Create language-specific datasets
banso_train = load_dataset("banso_corpus", split="train")
english_train = load_dataset("english_corpus", split="train")

# Create mixed dataset with language tags
def prepare_mixed_dataset(datasets_dict):
    """Create balanced dataset from multiple languages."""
    
    all_examples = []
    
    for lang, dataset in datasets_dict.items():
        for example in dataset:
            # Add language token at start
            example['text'] = f"<LANG:{lang}> " + example['text']
            all_examples.append(example)
    
    return shuffle(all_examples)

mixed_train = prepare_mixed_dataset({
    'banso': banso_train,
    'english': english_train
})

# Train on mixed dataset
trainer = Trainer(model=model, args=args, train_dataset=mixed_train)
trainer.train()
```

### Domain-Specific Vocabulary

```python
# Extend tokenizer with domain terms
domain_tokens = ["<DOMAIN_TERM_1>", "<DOMAIN_TERM_2>", ...]

# Add special tokens
tokenizer.add_special_tokens({"additional_special_tokens": domain_tokens})

# Resize model embeddings
model.resize_token_embeddings(len(tokenizer))

# Initialize new tokens with average of existing embeddings
with torch.no_grad():
    num_existing = original_vocab_size
    for new_token_idx in range(num_existing, len(tokenizer)):
        # Initialize with average of existing tokens
        model.transformer.wte.weight[new_token_idx] = model.transformer.wte.weight[:num_existing].mean(dim=0)
```

## Evaluation During Fine-tuning

### Custom Metrics

```python
from datasets import load_metric
import numpy as np

def compute_metrics(eval_pred):
    """Compute metrics during evaluation."""
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Perplexity
    loss = np.mean(-np.log(np.max(softmax(logits, axis=-1), axis=-1)))
    perplexity = np.exp(loss)
    
    # Accuracy (on non-padding tokens)
    mask = labels != -100
    accuracy = (predictions[mask] == labels[mask]).mean()
    
    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
```

### Early Stopping

```python
from transformers import EarlyStoppingCallback

callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # Stop if no improvement for 3 evals
    early_stopping_threshold=0.0,  # threshold for improvement
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[callback],
)
```

## Knowledge Distillation

### Training Student Model

```python
class DistilledModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=4):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=4,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, x)
        return self.lm_head(x)

def distillation_loss(student_logits, teacher_logits, temperature=3.0):
    """KL divergence between student and teacher."""
    
    student_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    
    kl_loss = torch.nn.functional.kl_div(student_probs, teacher_probs, reduction='batchmean')
    
    return kl_loss

# Training loop
teacher_model.eval()  # Teacher stays frozen

for epoch in range(num_epochs):
    for batch in train_loader:
        # Get predictions
        with torch.no_grad():
            teacher_logits = teacher_model(batch['input_ids'])
        
        student_logits = student_model(batch['input_ids'])
        
        # Compute distillation loss
        loss = distillation_loss(student_logits, teacher_logits)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Hyperparameter Tuning

### Recommendation Table

| Hyperparameter | LoRA | Full | Notes |
|---|---|---|---|
| Learning Rate | 5e-4 | 5e-5 | LoRA can handle higher LR due to smaller param space |
| Batch Size | 8-16 | 2-4 | LoRA more memory efficient |
| Warmup Steps | 500-1000 | 2000-5000 | Depends on dataset size |
| Weight Decay | 0.01 | 0.01 | L2 regularization |
| Gradient Clipping | 1.0 | 1.0 | Prevent explosion |
| Epochs | 3-5 | 1-2 | LoRA can train longer without overfitting |

### Automatic Hyperparameter Search

```python
from ray import tune
from transformers import Trainer

def trial_objective(config):
    """Objective function for tuning."""
    
    training_args = TrainingArguments(
        output_dir=f"./trial_{config['lr']}",
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        learning_rate=config['lr'],
        warmup_steps=config['warmup_steps'],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    
    metrics = trainer.evaluate()
    return {"objective": metrics['eval_loss']}

# Define search space
search_space = {
    'lr': tune.loguniform(1e-5, 1e-3),
    'batch_size': tune.choice([8, 16, 32]),
    'warmup_steps': tune.choice([0, 500, 1000]),
    'epochs': tune.choice([1, 3, 5]),
}

# Run tuning
results = tune.run(
    trial_objective,
    config=search_space,
    num_samples=10,  # Number of trials
    verbose=1
)

best_config = results.get_best_config(metric="objective", mode="min")
print(f"Best config: {best_config}")
```

---

**Guide Version**: 1.0
**Last Updated**: 2024
