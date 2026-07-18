# Fine-Tuning & Domain Adaptation

## Transfer Learning Strategy

### Continued Pre-training (CPT)

Further pre-train MarkGPT on domain-specific data (medical, legal, etc.):

```python
def continued_pretraining(model, domain_data_loader, device, epochs=3):
    """Fine-tune base MarkGPT on domain data."""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(domain_data_loader):
            input_ids = batch['input_ids'].to(device)
            
            # Forward pass
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {avg_loss:.4f}")
    
    return model
```

**Key differences from pre-training:**
- Lower learning rate (1e-4 vs 1e-3) - avoid catastrophic forgetting
- Fewer epochs - domain data often smaller
- Same tokenizer - distribution shift handled differently
- Monitor divergence from English capability

---

## Task-Specific Fine-Tuning

### Instruction Tuning (For Chat/Assistant Models)

Train on instruction-output pairs:

```python
class InstructionTuner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def prepare_data(self, instructions_json):
        """Convert instruction data → (input, output) pairs."""
        
        dataset = []
        
        with open(instructions_json) as f:
            data = json.load(f)
        
        for example in data:
            instruction = example['instruction']
            output = example['output']
            
            # Format: "Instruction: {instr}\nOutput: {output}"
            formatted = f"""Instruction: {instruction}

Output: {output}"""
            
            dataset.append(formatted)
        
        return dataset
    
    def finetune(self, instruction_texts, epochs=3, lr=5e-5):
        """Instruction tuning."""
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for text in instruction_texts:
                # Tokenize
                tokens = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to('cuda')
                
                # Forward
                outputs = self.model(
                    tokens['input_ids'],
                    labels=tokens['input_ids']
                )
                loss = outputs.loss
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                print(f"Loss: {loss.item():.4f}")

# Usage
tuner = InstructionTuner(model, tokenizer)
instruction_texts = tuner.prepare_data('instructions.json')
tuner.finetune(instruction_texts)
```

### Question Answering Fine-Tuning

```python
def qa_finetuning(model, tokenizer, qa_dataset):
    """Fine-tune for Q&A tasks."""
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for example in qa_dataset:
        question = example['question']
        context = example['context']
        answer = example['answer']
        
        # Format: "Q: {q}\nContext: {c}\nA: {a}"
        prompt = f"Q: {question}\nContext: {context}\nA: {answer}"
        
        # Only compute loss on answer part
        full_text = f"Q: {question}\nContext: {context}\nA:"
        
        tokens = tokenizer(
            prompt,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to('cuda')
        
        # Identify answer token positions
        answer_tokens = tokenizer(answer).input_ids
        answer_start = tokens['input_ids'].shape[1] - len(answer_tokens)
        
        # Create labels (ignore prefix)
        labels = tokens['input_ids'].clone()
        labels[:, :answer_start] = -100  # Ignore
        
        # Forward
        outputs = model(tokens['input_ids'], labels=labels)
        loss = outputs.loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

```

---

## Parameter-Efficient Fine-Tuning

### Low-Rank Adaptation (LoRA)

Fine-tune only small rank-r matrices instead of full weights:

```python
import loralib as lora
from transformers import get_linear_schedule_with_warmup

def apply_lora(model, r=8, lora_alpha=16):
    """Apply LoRA to model."""
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Replace with LoRA linear layer
            # Only for attention & FFN
            if 'self_attn' in name or 'mlp' in name:
                lora_module = lora.Linear(
                    module.in_features,
                    module.out_features,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.05
                )
                
                # Copy original weights
                lora_module.weight.data = module.weight.data.clone()
                
                # Replace in module
                parent = dict(model.named_modules())['.'.join(name.split('.')[:-1])]
                setattr(parent, name.split('.')[-1], lora_module)
    
    # Only train LoRA params
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    
    # Count trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model
```

**LoRA Benefits:**
- Original 500M MarkGPT → Only 3.6M trainable params (0.72%)
- Can fine-tune on single GPU (previously required multi-GPU)
- Easily combine multiple LoRA adapters for multi-task learning

### Adapter Layers

```python
class LoRAAdapter(nn.Module):
    def __init__(self, hidden_size, r=8):
        super().__init__()
        self.down = nn.Linear(hidden_size, r)
        self.up = nn.Linear(r, hidden_size)
        
    def forward(self, x):
        return self.up(F.relu(self.down(x)))

def integrate_adapters(model, r=8):
    """Add adapter layers to TransformerBlock."""
    
    for block in model.transformer.h:
        # Adapter in FFN
        original_ffn = block.mlp
        
        class WrappedFFN(nn.Module):
            def __init__(self, original, adapter):
                super().__init__()
                self.original = original
                self.adapter = adapter
            
            def forward(self, x):
                return self.original(x) + self.adapter(x)
        
        adapter = LoRAAdapter(model.config.hidden_size, r=r)
        block.mlp = WrappedFFN(original_ffn, adapter)
    
    return model
```

---

## Multi-Task Learning

Train on multiple tasks simultaneously:

```python
class MultiTaskTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tasks = {}
    
    def add_task(self, task_name, task_data, task_weight=1.0):
        """Register task."""
        self.tasks[task_name] = {
            'data': task_data,
            'weight': task_weight,
            'steps': 0,
            'loss_history': []
        }
    
    def train_step(self, task_name, batch):
        """Single training step."""
        
        task_info = self.tasks[task_name]
        
        if task_name == 'language_modeling':
            # Standard LM loss
            input_ids = batch['input_ids'].to('cuda')
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            
        elif task_name == 'masked_language_modeling':
            # MLM loss (mask 15% tokens)
            input_ids = batch['input_ids'].to('cuda')
            mask_indices = torch.rand(input_ids.shape) < 0.15
            input_ids[mask_indices] = self.tokenizer.mask_token_id
            
            outputs = self.model(input_ids, labels=batch['input_ids'].to('cuda'))
            loss = outputs.loss
            
        elif task_name == 'next_sentence_prediction':
            # NSP loss (predict if sentence B follows A)
            input_ids = batch['input_ids'].to('cuda')
            labels = batch['nsp_label'].to('cuda')
            
            # Get [CLS] token representation
            output = self.model(input_ids)
            cls_hidden = output.last_hidden_state[:, 0, :]
            
            # Classifier
            logits = self.classifier(cls_hidden)
            loss = F.cross_entropy(logits, labels)
        
        # Weight loss by task
        loss = loss * task_info['weight']
        
        # Track
        task_info['loss_history'].append(loss.item())
        task_info['steps'] += 1
        
        return loss
    
    def train(self, epochs=3):
        """Multi-task training."""
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            # Rotate tasks
            for task_name, task_info in self.tasks.items():
                
                # Sample batch
                batch = random.choice(task_info['data'])
                
                # Training step
                loss = self.train_step(task_name, batch)
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                avg_loss = np.mean(task_info['loss_history'][-100:])
                print(f"{task_name}: {avg_loss:.4f}")

# Usage
trainer = MultiTaskTrainer(model, tokenizer)
trainer.add_task('language_modeling', en_data, weight=1.0)
trainer.add_task('masked_language_modeling', en_data, weight=0.5)
trainer.add_task('next_sentence_prediction', en_data, weight=0.3)
trainer.train(epochs=3)
```

---

## Domain-Specific Fine-Tuning Examples

### Medical Text (MarkGPT-Med)

```python
def create_medical_markgpt():
    """Create MarkGPT variant for medical text."""
    
    model = MarkGPT.from_pretrained('markgpt_small')
    
    # Load medical corpus
    medical_data = load_dataset('medical_pubmed')
    
    # Continued pre-training on medical data
    model = continued_pretraining(
        model,
        medical_data,
        epochs=2,
        lr=5e-5
    )
    
    # Save
    model.save_pretrained('markgpt_med')
    
    return model
```

### Code Generation (MarkGPT-Code)

```python
def create_code_markgpt():
    """Create MarkGPT for code generation."""
    
    model = MarkGPT.from_pretrained('markgpt_base')
    
    # Load code corpus
    code_data = load_dataset('github_code')
    
    # CPT on code
    model = continued_pretraining(
        model,
        code_data,
        epochs=2,
        lr=1e-4
    )
    
    # Fine-tune on HumanEval tasks
    model = qa_finetuning(model, tokenizer, humaneval_dataset)
    
    return model
```

### Multilingual (MarkGPT-Multi)

```python
def create_multilingual_markgpt():
    """Balance English and Banso."""
    
    model = MarkGPT.from_pretrained('markgpt_base')
    
    # Mixed curriculum
    balanced_data = interleave_languages(
        english_corpus,
        banso_corpus,
        ratio=0.5  # 50-50
    )
    
    model = continued_pretraining(
        model,
        balanced_data,
        epochs=3,
        lr=1e-4
    )
    
    return model
```

---

**Fine-Tuning Guide v1.0**
**Last Updated**: 2024
