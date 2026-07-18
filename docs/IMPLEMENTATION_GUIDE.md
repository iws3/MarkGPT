# Implementation Guide: Building Custom Modules

## Creating a Custom Training Module

### Step 1: Define Your Module Class

```python
import torch
import torch.nn as nn
from torch.nn import TransformerBlock

class CustomMarkGPTModule(nn.Module):
    """Template for custom MarkGPT variants."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.d_model
        )
        
        # Positional embeddings (RoPE)
        self.rope = RotaryPositionalEmbedding(config.d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        # Get embeddings
        x = self.token_embeddings(input_ids)
        
        # Apply RoPE
        x = self.rope(x)
        
        # Pass through layers
        past_kv = []
        for layer_idx, layer in enumerate(self.layers):
            layer_past = past_key_values[layer_idx] if past_key_values else None
            x, kv = layer(x, attention_mask=attention_mask, past_key_values=layer_past)
            past_kv.append(kv)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return {
            'logits': logits,
            'past_key_values': past_kv
        }
```

### Step 2: Create Configuration

```python
from transformers import PretrainedConfig

class CustomMarkGPTConfig(PretrainedConfig):
    model_type = "custom_markgpt"
    
    def __init__(
        self,
        vocab_size=50257,
        d_model=768,
        num_layers=24,
        num_heads=12,
        ffn_dim=3072,
        dropout=0.1,
        max_seq_length=2048,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.max_seq_length = max_seq_length
```

### Step 3: Register with Hugging Face

```python
from transformers import AutoConfig, AutoModel, PreTrainedModel

# Register config
AutoConfig.register("custom_markgpt", CustomMarkGPTConfig)

# Create model class
class CustomMarkGPTPreTrainedModel(PreTrainedModel):
    config_class = CustomMarkGPTConfig
    base_model_prefix = "custom_markgpt"
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

class CustomMarkGPTForCausalLM(CustomMarkGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomMarkGPTModule(config)
        self.apply(self._init_weights)
    
    def forward(self, input_ids, labels=None, **kwargs):
        outputs = self.model(input_ids, **kwargs)
        
        if labels is not None:
            # Compute loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                outputs['logits'].view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            outputs['loss'] = loss
        
        return outputs

# Register
AutoModel.register(CustomMarkGPTConfig, CustomMarkGPTForCausalLM)
```

---

## Creating a Custom Fine-tuning Pipeline

### Step 1: Define Data Processing

```python
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling

def preprocess_function(examples, tokenizer, max_length=2048):
    """Preprocess text for causal LM."""
    
    # Tokenize
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length' if max_length else 'longest',
    )
    
    # For CLM, input and labels are the same
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized

# Apply to dataset
tokenizer = AutoTokenizer.from_pretrained('markgpt-small')
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

train_dataset = dataset['train'].map(
    preprocess_function,
    fn_kwargs={'tokenizer': tokenizer},
    batched=True,
    remove_columns=['text']
)

val_dataset = dataset['validation'].map(
    preprocess_function,
    fn_kwargs={'tokenizer': tokenizer},
    batched=True,
    remove_columns=['text']
)
```

### Step 2: Set Up Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./custom_checkpoint',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    learning_rate=2e-4,
    lr_scheduler_type='cosine',
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    fp16=True,
    dataloader_pin_memory=True,
    optim='adamw_torch',
    report_to=['wandb'],  # Weights & Biases
)
```

### Step 3: Create Trainer

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not masked LM, causal LM
    ),
)

# Train
results = trainer.train()

# Save
trainer.save_model('./final_model')
```

---

## Creating an Inference Engine

### Step 1: Define Engine Class

```python
import torch
from transformers import AutoModel, AutoTokenizer

class MarkGPTInferenceEngine:
    def __init__(self, model_name, device='cuda', quantize=False):
        if quantize:
            from transformers import BitsAndBytesConfig
            config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=200.0,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                quantization_config=config,
                device_map='auto'
            )
        else:
            self.model = AutoModel.from_pretrained(model_name).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=0.7):
        """Generate text from prompt."""
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        input_ids = input_ids.to(self.device)
        
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
        )
        
        return self.tokenizer.decode(output_ids[0])
```

### Step 2: Add Streaming Support

```python
def stream_generate(self, prompt, max_new_tokens=100):
    """Stream tokens one at a time."""
    
    input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(self.device)
    past_kv = None
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True
            )
        
        logits = output['logits'][:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
        
        token_str = self.tokenizer.decode(next_token[0])
        yield token_str
        
        input_ids = next_token
        past_kv = output['past_key_values']
```

---

## Creating Evaluation Metrics

### Custom Metric Implementation

```python
import torch
import numpy as np
from sklearn.metrics import f1_score

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    
    predictions, labels = eval_pred
    
    # Perplexity
    loss = np.mean(
        -np.log(np.max(torch.softmax(torch.tensor(predictions), dim=-1).numpy(), axis=-1))
    )
    perplexity = np.exp(loss)
    
    # Accuracy (non-padding tokens only)
    pred_ids = np.argmax(predictions, axis=-1)
    valid = labels != -100
    accuracy = (pred_ids[valid] == labels[valid]).mean()
    
    return {
        'perplexity': perplexity,
        'accuracy': accuracy,
    }
```

---

## Publishing to Hugging Face Hub

### Step 1: Create Model Card

```markdown
---
language:
- en
- banso
license: apache-2.0
---

# CustomMarkGPT

A custom variant of MarkGPT trained on...

## Model Details

- **Model Size**: 200M parameters
- **Training Data**: 60B tokens
- **Languages**: English, Banso

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('username/custom-markgpt')
tokenizer = AutoTokenizer.from_pretrained('username/custom-markgpt')

inputs = tokenizer("Hello, ", return_tensors='pt')
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

## Performance

- English Perplexity: 16.5
- Banso Perplexity: 32.0
```

### Step 2: Push to Hub

```python
from huggingface_hub import notebook_login

# Login
notebook_login()

# Push model
model.push_to_hub("username/custom-markgpt", private=False)
tokenizer.push_to_hub("username/custom-markgpt", private=False)
```

---

**Implementation Guide v1.0**
**Last Updated**: 2024
