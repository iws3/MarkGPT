# Hugging Face Hub Integration

## Publish MarkGPT to Model Hub

### Step 1: Create Repository

```bash
huggingface-cli repo create markgpt-base

# Login if needed
huggingface-cli login
```

### Step 2: Clone Repository

```bash
git clone https://huggingface.co/username/markgpt-base
cd markgpt-base
```

### Step 3: Upload Model

```python
from transformers import MarkGPTForCausalLM

model = MarkGPTForCausalLM.from_pretrained('local_path')
tokenizer = MarkGPTTokenizer.from_pretrained('local_path')

# Push to hub
model.push_to_hub("username/markgpt-base")
tokenizer.push_to_hub("username/markgpt-base")
```

### Step 4: Create Model Card

```markdown
# MarkGPT Base

## Model Description

MarkGPT is a 500M parameter bilingual language model trained on English and Banso text.

## Training Data

- English: Wikipedia, CC100, Books (~97B tokens)
- Banso: BANSO corpus, vernacular texts (~3B tokens)

## Training Setup

- Framework: PyTorch
- Optimizer: AdamW with warmup cosine schedule
- Hardware: 8× A100 GPUs
- Training time: 14 days
- Learning rate: 1e-4 with 10K warmup

## Capabilities

- English text generation
- Banso text generation
- Bilingual code-switching
- Few-shot learning (sentiment, Q&A)
- Instruction following (LoRA fine-tuned variants)

## Limitations

- Trained primarily on English
- Banso data is lower quality/smaller
- May reproduce training data
- Biases present in training corpus

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("username/markgpt-base")
model = AutoModelForCausalLM.from_pretrained("username/markgpt-base")

# Generate
input_ids = tokenizer.encode("The future of AI", return_tensors='pt')
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

## Citation

```bibtex
@model{markgpt2024,
  title={MarkGPT: A Bilingual Language Model for English and Banso},
  author={Your Name},
  institution={Your Organization},
  year={2024}
}
```

## License

MIT

## Contact

[Your contact info]
```

### Step 5: Push Card

```bash
cd markgpt-base

# Create model card
cat > README.md << 'EOF'
# MarkGPT Base
...content...
EOF

git add README.md
git commit -m "Add model card"
git push
```

---

## Share & Discovery

### On HuggingFace Hub

```python
# Link appears at: https://huggingface.co/username/markgpt-base

# Users can load directly:
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("username/markgpt-base")

# In Inference API (if enabled):
# POST https://api-inference.huggingface.co/models/username/markgpt-base
```

### Collections

```python
from huggingface_hub import create_collection

collection = create_collection(
    name="MarkGPT Models",
    description="Official MarkGPT model releases (Nano, Small, Base)"
)

# Add models
collection.add("username/markgpt-nano")
collection.add("username/markgpt-small")
collection.add("username/markgpt-base")
```

---

## ONNX Export

Export to ONNX for inference engine compatibility:

```python
from transformers import MarkGPTForCausalLM
from torch.onnx import export

model = MarkGPTForCausalLM.from_pretrained('markgpt-base')

# Prepare dummy input
dummy_input = torch.randn(1, 512, dtype=torch.long)

# Export
export(
    model,
    dummy_input,
    "markgpt_base.onnx",
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length', 2: 'vocab_size'}
    },
    do_constant_folding=True,
    opset_version=14
)

print("✓ Exported to markgpt_base.onnx")

# Verify
import onnx
model_onnx = onnx.load("markgpt_base.onnx")
onnx.checker.check_model(model_onnx)
print("✓ ONNX model valid")
```

### Load ONNX Model

```python
import onnxruntime as ort

# Load
session = ort.InferenceSession("markgpt_base.onnx")

# Prepare input
input_ids = np.array([[1, 2, 3]], dtype=np.int64)

# Run
outputs = session.run(None, {'input_ids': input_ids})

print(f"Output shape: {outputs[0].shape}")  # (1, 3, vocab_size)
```

---

## TensorRT Engine

Convert to TensorRT for NVIDIA GPU acceleration:

```bash
# Using trtexec
trtexec --onnx=markgpt_base.onnx \
        --saveEngine=markgpt_base.trt \
        --workspace=2048 \
        --fp16  # Enable half precision
```

### Load TensorRT

```python
import pycuda.driver as cuda
import tensorrt as trt

def load_trt_engine(engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    return engine

engine = load_trt_engine("markgpt_base.trt")
context = engine.create_execution_context()

# Run inference
# (See TensorRT documentation for buffer management)
```

---

## Triton Inference Server

Deploy using NVIDIA Triton:

### Model Config

```ini
# models/markgpt/config.pbtxt
name: "markgpt"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [512]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [1, 512, 50257]
  }
]

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
```

### Deploy

```bash
# Download Triton
wget https://github.com/triton-inference-server/server/releases/download/v2.32.0/tritonserver2.32.0-gpu.docker.tar.gz

# Load model
tritonserver --model-repository=/path/to/models

# Client
curl -X POST \
  -H "Content-Type: application/octet-stream" \
  -d @input.bin \
  http://localhost:8000/v2/models/markgpt/infer
```

---

## vLLM Serving

High-throughput serving with vLLM:

```python
import vllm

# Initialize engine
llm = vllm.LLM(
    model="username/markgpt-base",
    tensor_parallel_size=1,  # Single GPU
    gpu_memory_utilization=0.9
)

# Generate
prompts = [
    "The future of AI is",
    "Machine learning enables",
]

sampling_params = vllm.SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50,
    num_beams=1
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### vLLM FastAPI Server

```python
from fastapi import FastAPI
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

app = FastAPI()

llm_engine = None

@app.on_event("startup")
async def startup():
    global llm_engine
    engine_args = AsyncEngineArgs(
        model="username/markgpt-base",
        tensor_parallel_size=1
    )
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 50):
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=max_tokens
    )
    
    results = await llm_engine.generate(
        prompt,
        sampling_params,
        request_id=str(time.time())
    )
    
    return {"generated_text": results[0].outputs[0].text}

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## OLlama Support

Bundle for Ollama:

```dockerfile
# Dockerfile
FROM ollama/ollama

# Copy model
COPY markgpt_base.gguf /root/.ollama/models/markgpt_base

EXPOSE 11434
CMD ["ollama", "serve"]
```

### Convert to GGUF

```bash
pip install llama-cpp-python

# Quantize to GGUF
python convert.py \
  --model-dir ./markgpt_base \
  --outfile markgpt_base.gguf \
  --outtype q4_0  # 4-bit quantization
```

### Run with Ollama

```bash
# Add model
ollama create markgpt -f Modelfile

# Generate
ollama run markgpt "The future of AI"
```

---

## Community Deployment

### CivitAI (for image models, but similar pattern):

```json
{
  "name": "MarkGPT Base",
  "description": "500M bilingual language model",
  "creator": "Your Organization",
  "license": "MIT",
  "baseModel": "markgpt",
  "downloadUrl": "https://huggingface.co/username/markgpt-base/resolve/main/model.safetensors"
}
```

### Model Archive

Create downloadable pack:

```bash
mkdir -p markgpt_base_archive
cd markgpt_base_archive

# Copy files
cp -r ../markgpt_base/* .

# Create tarball
tar -czf markgpt_base.tar.gz .

# Upload to storage
gsutil cp markgpt_base.tar.gz gs://your-bucket/

# Share link (public bucket)
gsutil acl ch -u AllUsers:R gs://your-bucket/markgpt_base.tar.gz
```

---

**Integration & Deployment v1.0**
**Last Updated**: 2024
