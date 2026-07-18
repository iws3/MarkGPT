# MarkGPT Quick Start Guide

## Installation

### Option 1: From Source

```bash
git clone https://github.com/iwstechnical/markgpt
cd markgpt
pip install -e .
```

### Option 2: From PyPI (Soon)

```bash
pip install markgpt
```

### Verify Installation

```python
from markgpt.model import MarkGPT
from markgpt.tokenizer import MarkGPTTokenizer

print("✓ MarkGPT installed successfully!")
```

---

## Load Pre-trained Model

### Nano (Lightweight, Fast)

```python
import torch
from markgpt.model import MarkGPT
from markgpt.tokenizer import MarkGPTTokenizer

# Load tokenizer
tokenizer = MarkGPTTokenizer.from_pretrained('markgpt-nano')

# Load model
model = MarkGPT.from_pretrained('markgpt-nano')
model.eval()

if torch.cuda.is_available():
    model = model.to('cuda')

print(f"✓ Loaded MarkGPT-Nano (70M parameters)")
```

### Small (Balanced)

```python
tokenizer = MarkGPTTokenizer.from_pretrained('markgpt-small')
model = MarkGPT.from_pretrained('markgpt-small')
model.eval()
model.to('cuda')

print(f"✓ Loaded MarkGPT-Small (200M parameters)")
```

### Base (Most Capable)

```python
tokenizer = MarkGPTTokenizer.from_pretrained('markgpt-base')
model = MarkGPT.from_pretrained('markgpt-base')
model.eval()
model.to('cuda')

print(f"✓ Loaded MarkGPT-Base (500M parameters)")
```

---

## Generate Text

### Basic Generation

```python
prompt = "The future of artificial intelligence is"

# Tokenize
input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')

# Generate
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )

# Decode
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### Batch Generation

```python
prompts = [
    "The future of AI is",
    "Machine learning is",
    "Deep learning enables",
]

# Tokenize batch
encoded = tokenizer(
    prompts,
    padding=True,
    truncation=True,
    return_tensors='pt'
).to('cuda')

# Generate batch
with torch.no_grad():
    outputs = model.generate(
        encoded['input_ids'],
        attention_mask=encoded['attention_mask'],
        max_new_tokens=30,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        num_return_sequences=1
    )

# Decode
for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Prompt {i+1}: {text}")
```

---

## Bilingual (English + Banso) Generation

### English Generation

```python
prompt_en = "Python programming is used for"

input_ids = tokenizer.encode(prompt_en, return_tensors='pt').to('cuda')

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=40)

result = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"English: {result}")
```

### Banso Generation

```python
# Banso vocabulary is reserved in tokenizer
prompt_banso = "Ulimi lwesibanso luyimpilo ye"  # "Banso language is the culture of"

input_ids = tokenizer.encode(prompt_banso, return_tensors='pt').to('cuda')

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=40,
        temperature=1.0  # Higher for more diversity in low-resource language
    )

result = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Banso: {result}")
```

### Code-Switching (Mixed Languages)

```python
# Start with English, may switch to Banso
prompt_mixed = "Imiscelanea: The history of ubuntu philosophy in Banso culture"

input_ids = tokenizer.encode(prompt_mixed, return_tensors='pt').to('cuda')

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.9
    )

result = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Mixed: {result}")
```

---

## Few-Shot Learning

### Sentiment Classification

```python
few_shot_prompt = """Classify sentiment as Positive or Negative.

Example 1: "I love this product!" -> Positive
Example 2: "This is terrible." -> Negative
Example 3: "Amazing experience!" -> Positive

Now classify: "The service was outstanding!"
Answer: """

input_ids = tokenizer.encode(few_shot_prompt, return_tensors='pt').to('cuda')

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=5,
        temperature=0.1
    )

result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result)
# Expected: "Positive"
```

### Q&A

```python
qa_prompt = """Answer the question based on context.

Context: Paris is the capital of France and is famous for the Eiffel Tower, museums, and art.

Question: What is the capital of France?
Answer: """

input_ids = tokenizer.encode(qa_prompt, return_tensors='pt').to('cuda')

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=10,
        temperature=0.1
    )

answer = tokenizer.decode(output[0], skip_special_tokens=True)
print(answer)
# Expected: "Paris"
```

### Code Generation

```python
code_prompt = """Complete the Python function:

def fibonacci(n):
    '''Return the nth Fibonacci number.'''
    """

input_ids = tokenizer.encode(code_prompt, return_tensors='pt').to('cuda')

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=30,
        temperature=0.1
    )

code = tokenizer.decode(output[0], skip_special_tokens=True)
print(code)
```

---

## Using with Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load from HF Hub (when available)
tokenizer = AutoTokenizer.from_pretrained("iwstechnical/markgpt-base")
model = AutoModelForCausalLM.from_pretrained("iwstechnical/markgpt-base")

# Generate
prompt = "Machine learning is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

---

## Streaming Generation (FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.8

@app.post("/generate")
async def generate(request: GenerationRequest):
    input_ids = tokenizer.encode(
        request.prompt,
        return_tensors='pt'
    ).to('cuda')
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature
        )
    
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {"generated_text": text}

# Run: uvicorn quickstart:app --reload
# Test: curl -X POST "http://localhost:8000/generate" \
#       -H "Content-Type: application/json" \
#       -d '{"prompt": "The future is", "max_tokens": 30}'
```

---

## Performance Tips

### Inference Speedup

```python
# Use half precision (fp16)
model = model.half()  # 2x faster on modern GPUs

# Disable gradient computation
with torch.no_grad():
    output = model.generate(...)

# Use smaller model for speed
model = MarkGPT.from_pretrained('markgpt-nano')

# Batch generation
outputs = model.generate(
    batch_input_ids,
    batch_size=32
)
```

### Memory Optimization

```python
# Load in 8-bit quantization (requires bitsandbytes)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "markgpt-base",
    load_in_8bit=True,
    device_map="auto"
)  # 4x memory savings

# On 8GB GPU
model = MarkGPT.from_pretrained('markgpt-base')
model = model.half().to('cuda')
```

### Cache Optimization

```python
# Use KV-cache to avoid recomputation
# (Already handled in model.generate())

# Reduce sequence length
input_ids = tokenizer.encode(
    prompt,
    max_length=256,
    truncation=True,
    return_tensors='pt'
)
```

---

## Common Issues & Solutions

### OutOfMemory Error

```python
# Solution 1: Use smaller model
model = MarkGPT.from_pretrained('markgpt-nano')

# Solution 2: Use fp16
model = model.half()

# Solution 3: Reduce batch size
batch_size = 2  # Instead of 32

# Solution 4: Use 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "markgpt-base",
    load_in_8bit=True
)
```

### Slow Generation

```python
# Solution 1: Reduce max_new_tokens
output = model.generate(input_ids, max_new_tokens=10)

# Solution 2: Use smaller model
model = MarkGPT.from_pretrained('markgpt-nano')

# Solution 3: Enable fp16
model = model.half()

# Solution 4: Use smaller temperature (less sampling)
output = model.generate(input_ids, temperature=0.1)
```

### CUDA Out of Memory

```python
# Solution: Move model to CPU for single inferences
model = model.to('cpu')
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids)
```

---

## Next Steps

1. **Fine-tuning**: See [FINETUNING_ADAPTATION.md](FINETUNING_ADAPTATION.md)
2. **Advanced**: See [ADVANCED_TECHNIQUES.md](ADVANCED_TECHNIQUES.md)
3. **Production**: See [MODEL_SERVING_INFERENCE.md](MODEL_SERVING_INFERENCE.md)
4. **Evaluation**: See [EVALUATION_METRICS.md](EVALUATION_METRICS.md)

---

**Quick Start v1.0**
**Last Updated**: 2024
