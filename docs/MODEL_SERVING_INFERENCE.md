# Model Serving & Inference APIs

## Local Inference (Transformers Library)

### Simplest Approach

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "markgpt-small"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate text
prompt = "The future of AI is"
inputs = tokenizer.encode(prompt, return_tensors='pt')

outputs = model.generate(
    inputs,
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

text = tokenizer.decode(outputs[0])
print(text)
```

### Optimized Inference (Faster)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model + optimizations
model = AutoModelForCausalLM.from_pretrained(
    'markgpt-small',
    torch_dtype=torch.float16,  # Lower precision
    device_map='auto'           # Use GPU if available
)

# Enable inference optimizations
model.eval()

# Static KV cache (reuse across calls)
with torch.inference_mode():
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        use_cache=True,  # KV caching
    )
```

---

## FastAPI Serving (REST Endpoint)

### Server Implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(title="MarkGPT API")

# Load model once
model = AutoModelForCausalLM.from_pretrained('markgpt-small').cuda()
tokenizer = AutoTokenizer.from_pretrained('markgpt-small')
model.eval()

# Request/response schemas
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.9

class GenerationResponse(BaseModel):
    generated_text: str
    tokens_used: int
    processing_time: float

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    import time
    start = time.time()
    
    try:
        # Tokenize
        inputs = tokenizer.encode(request.prompt, return_tensors='pt').cuda()
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_used = outputs.shape[1] - inputs.shape[1]
        
        return GenerationResponse(
            generated_text=generated_text,
            tokens_used=tokens_used,
            processing_time=time.time() - start
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "markgpt-small"}

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

### Client Usage

```python
import requests

api_url = "http://localhost:8000/generate"

response = requests.post(api_url, json={
    "prompt": "Explain quantum computing:",
    "max_tokens": 200,
    "temperature": 0.7
})

result = response.json()
print(result['generated_text'])
```

---

## Batch Processing (Higher Throughput)

```python
class BatchInferenceServer:
    def __init__(self, model, tokenizer, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.queue = []
        self.futures = []
    
    def add_request(self, prompt):
        """Queue a request."""
        import asyncio
        future = asyncio.Future()
        self.queue.append((prompt, future))
        
        # Process batch when queue is full
        if len(self.queue) >= self.batch_size:
            self.process_batch()
        
        return future
    
    def process_batch(self):
        """Process all queued requests together."""
        if not self.queue:
            return
        
        prompts = [p for p, _ in self.queue]
        futures = [f for _, f in self.queue]
        
        # Batch encode
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to('cuda')
        
        # Batch generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)
        
        # Batch decode
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        # Resolve futures
        for future, text in zip(futures, generated_texts):
            future.set_result(text)
        
        self.queue.clear()

# Usage
batch_server = BatchInferenceServer(model, tokenizer, batch_size=32)

# Queue requests
future1 = batch_server.add_request("Hello")
future2 = batch_server.add_request("World")

# Process when batch is full
result1 = asyncio.run(future1)
result2 = asyncio.run(future2)
```

---

## vLLM (High-Performance Serving)

### Why vLLM?

- Optimized attention kernel (Flash Attention)
- Continuous batch execution
- Efficient KV cache management
- 10-100x throughput improvement

```python
# Install: pip install vllm

from vllm import LLM, SamplingParams

# Load model
llm = LLM(
    model='markgpt-small',
    dtype='float16',
    gpu_memory_utilization=0.9  # Use 90% of GPU
)

# Create sampling params
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

# Batch inference
prompts = [
    "The future of AI",
    "Quantum computing is",
    "Machine learning enables",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)

# Throughput: ~500-1000 tokens/sec per GPU
```

### vLLM Server

```bash
# Start server
python -m vllm.entrypoints.openai.api_server \
    --model markgpt-small \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --port 8000

# OpenAI-compatible API
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "markgpt-small",
        "prompt": "The future of AI",
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

---

## Streaming Responses

### Server-Sent Events (SSE)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

@app.post("/generate-stream")
async def generate_stream(request: GenerationRequest):
    async def event_generator():
        prompt_tokens = tokenizer.encode(request.prompt, return_tensors='pt')
        
        for token_id in model.generate(
            prompt_tokens,
            max_new_tokens=request.max_tokens,
            **generation_kwargs
        ):
            token_text = tokenizer.decode(token_id)
            yield f"data: {json.dumps({'token': token_text})}\n\n"
            
            # Avoid buffering
            await asyncio.sleep(0)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# Client (JavaScript)
const response = await fetch('/generate-stream', {method: 'POST'});
const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    
    const text = decoder.decode(value);
    console.log(text);  // Real-time tokens
}
```

---

## Caching & Optimization

### Response Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_generate(prompt_hash: str, temperature: float):
    """Cache results for same prompts."""
    # Do expensive generation
    return result

# In endpoint
import hashlib
prompt_hash = hashlib.md5(request.prompt.encode()).hexdigest()
result = cached_generate(prompt_hash, request.temperature)
```

### Quantization for Serving

```python
# Serve quantized model (4x faster, 4x memory efficient)
quantized_model = AutoModelForCausalLM.from_pretrained(
    'markgpt-small-int4',
    load_in_4bit=True,
    device_map='auto'
)

# Use same API, but faster
outputs = quantized_model.generate(inputs)
```

---

## Production Checklist

```python
class MarkGPTProductionServer:
    def __init__(self):
        # Monitoring
        self.request_count = 0
        self.error_count = 0
        self.latencies = []
        
        # Load model (optimized)
        self.model = load_optimized_model()
        self.tokenizer = load_tokenizer()
    
    def handle_request(self, prompt):
        """Production-ready inference."""
        
        # 1. Input validation
        if not prompt or len(prompt) > 5000:
            raise ValueError("Invalid prompt")
        
        # 2. Rate limiting (external service, e.g., Redis)
        # check_rate_limit(user_id)
        
        # 3. Generation (with timeout)
        try:
            output = self.model.generate(
                tokenizer.encode(prompt, return_tensors='pt'),
                max_new_tokens=100,
                timeout=10  # seconds
            )
        except TimeoutError:
            self.error_count += 1
            raise
        
        # 4. Logging
        self.request_count += 1
        
        # 5. Metrics
        # record_metrics(latency=elapsed_time, tokens=output_tokens)
        
        return output
    
    def health_check(self):
        """Monitoring endpoint."""
        return {
            "status": "healthy",
            "requests": self.request_count,
            "errors": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count),
            "avg_latency": np.mean(self.latencies[-100:]),
        }

# Deploy
server = MarkGPTProductionServer()
app.add_event_handler("startup", server.warmup)
app.add_api_route("/health", server.health_check)
```

---

**Model Serving v1.0**
**Last Updated**: 2024
