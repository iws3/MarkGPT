# FAQ: Common Questions and Troubleshooting

## Installation & Setup

### Q: How do I install MarkGPT on my system?
**A:** See [INSTALL.md](INSTALL.md) for platform-specific instructions. Quick start:
```bash
git clone https://github.com/iwstechnical/MarkGPT-LLM-Curriculum
cd MarkGPT-LLM-Curriculum
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Q: Which GPU do I need for MarkGPT?
**A:** MarkGPT supports various GPUs:
- **Consumer GPUs**: NVIDIA RTX 3090, 4090 (24-40GB) — run Medium/Large
- **Data center GPUs**: A100 (40-80GB) — run any model
- **Apple Silicon**: M1/M2/M3 with MPS support — run Small/Base
- **AMD GPUs**: ROCm-compatible (MI200 series) — run any model
- **CPU Only**: Works but ~10x slower (Small only)

### Q: What Python version does MarkGPT require?
**A:** Python 3.10 or higher. Check with:
```bash
python --version
```

### Q: How much disk space do I need?
**A:** Approximately:
- Base install: 5GB
- Training data: 10GB
- Models (all variants): 5GB
- Checkpoints: 20-50GB (depending on training)
- **Total**: 40-70GB recommended

## Training

### Q: My training is very slow. What can I do?
**A:** Several optimization strategies:
1. **Use mixed precision**: `export MARKGPT_MIXED_PRECISION=true`
2. **Enable Flash Attention**: `export MARKGPT_USE_FLASH_ATTENTION=true` (A100, V100 only)
3. **Increase batch size**: More GPU memory = higher throughput
4. **Use gradient accumulation**: Effective larger batches without OOM
5. **Reduce sequence length**: Shorter sequences = faster training

### Q: I'm getting `CUDA out of memory` errors. How do I fix this?
**A:** Try these in order:
1. Reduce batch size: `export MARKGPT_BATCH_SIZE=32`
2. Reduce sequence length: `--seq-length 512`
3. Enable gradient accumulation: `--gradient-accumulation-steps 4`
4. Disable Flash Attention: `export MARKGPT_USE_FLASH_ATTENTION=false`
5. Use mixed precision: `export MARKGPT_MIXED_PRECISION=true`

### Q: Why is my training loss NaN?
**A:** Common causes and fixes:
| Cause | Fix |
|-------|-----|
| Learning rate too high | Reduce to `1e-5` to `1e-4` |
| Gradient explosion | Enable gradient clipping: `--max-grad-norm 1.0` |
| Bad data | Validate data with `scripts/verify_data.py` |
| Precision issue | Use `TORCH_CUDA_LAUNCH_BLOCKING=1` for debugging |
| CUDA compute mismatch | Check GPU compatibility with model |

### Q: How long should training take for each model size?
**A:** On NVIDIA A100 (1 GPU, batch_size=64):
- **Nano**: ~2 hours for 1B tokens
- **Small**: ~8 hours for 1B tokens
- **Base**: ~20 hours for 1B tokens
- **Medium**: ~60+ hours for 1B tokens

### Q: Can I resume training from a checkpoint?
**A:** Yes! Checkpoints save model, optimizer, and scheduler state:
```bash
python src/training/train.py --resume-from checkpoints/ckpt_epoch_5.pt
```

### Q: How do I use mixed precision training?
**A:** Enable automatically with:
```bash
export MARKGPT_MIXED_PRECISION=true
python src/training/train.py
```

## Performance & Prediction

### Q: What's the inference speed for different models?
**A:** On NVIDIA A100:
- **Nano**: 3ms per forward pass → ~330 tokens/sec
- **Small**: 9ms per forward pass → ~110 tokens/sec
- **Base**: 24ms per forward pass → ~42 tokens/sec

### Q: How do I generate text with MarkGPT?
**A:** Simple inference:
```python
from src.model.markgpt import MarkGPT
from src.tokenizer.tokenizer import Tokenizer

model = MarkGPT.load_pretrained('markgpt-small')
tokenizer = Tokenizer.load('vocab.pkl')

prompt = "<en> John 3:16"
tokens = tokenizer.encode(prompt)
output = model.generate(tokens, max_length=100)
print(tokenizer.decode(output[0]))
```

### Q: How do I control generation temperature?
**A:** Temperature controls randomness (0=deterministic, 2=very random):
```python
output = model.generate(
    tokens,
    max_length=100,
    temperature=0.7  # Lower = more focused
)
```

### Q: How do I measure model quality (perplexity)?
**A:** See [PERFORMANCE_BENCHMARKS.md](PERFORMANCE_BENCHMARKS.md) for full details:
```python
import math
from src.utils.eval import evaluate_model

metrics = evaluate_model(model, test_loader)
perplexity = math.exp(metrics['loss'])
print(f"Test Perplexity: {perplexity:.2f}")
```

## Multilingual & Banso

### Q: How do I train a bilingual English/Banso model?
**A:** Use multilingual dataset:
```python
from src.utils.datasets import MultilingualDataset

dataset = MultilingualDataset(
    en_path='data/bible_en.txt',
    banso_path='data/bible_banso.txt',
    tokenizer=tokenizer,
    mix_ratio=0.7  # 70% English, 30% Banso
)
```

See [MULTILINGUAL_GUIDE.md](MULTILINGUAL_GUIDE.md) for detailed instructions.

### Q: How much Banso data do I need?
**A:** Recommended amounts:
- **Minimum**: 10k tokens (barely works)
- **Good**: 100k-500k tokens (reasonable quality)
- **Excellent**: 1M+ tokens (strong performance)

Current MarkGPT Banso corpus: ~300k tokens

### Q: Will Banso performance regress English quality?
**A:** Minimal regression with good curriculum:
```
English-only perplexity: 2.6
70% EN / 30% Banso: 2.8 (+7.7% regression)
50% EN / 50% Banso: 3.1 (+19% regression)
```

Mitigate with curriculum learning (see [MULTILINGUAL_GUIDE.md](MULTILINGUAL_GUIDE.md)).

## Evaluation & Benchmarking

### Q: How do I benchmark a model?
**A:** Run performance tests:
```bash
# Full benchmark
python scripts/benchmark.py --model markgpt-small --batch-size 64

# Specific metric
python scripts/benchmark_throughput.py --model-size small
```

See [PERFORMANCE_BENCHMARKS.md](PERFORMANCE_BENCHMARKS.md) for detailed numbers.

### Q: How do I compare MarkGPT to other models?
**A:** Use a common benchmark (e.g., LAMBADA, PTB):
```python
# MarkGPT Small on English Bible
perplexity = 3.8

# Comparable models:
# - GPT-2 Small: 20.5 (different dataset)
# - Pythia 70M: 15.2
# - TinyLLaMA 1.1B: ~3.0
```

**Note**: Direct comparison difficult due to different training data.

## Deployment

### Q: How do I deploy MarkGPT in production?
**A:** Multiple options:

1. **REST API** (recommended):
   ```bash
   python -m src.inference.api --port 8000
   curl -X POST http://localhost:8000/v1/completions
   ```

2. **Docker**:
   ```bash
   docker build -t markgpt:latest .
   docker run -p 8000:8000 markgpt:latest
   ```

3. **Quantized (edge)**:
   ```python
   model_int8 = quantize_model(model)  # 4x smaller
   model_int8.save('model_quantized.pt')
   ```

See [DEPLOYMENT_GUIDE.md](../capstone/DEPLOYMENT_GUIDE.md).

### Q: How do I reduce model size for mobile?
**A:** Use quantization:
```bash
# FP32 → INT8 conversion
python scripts/quantize_model.py markgpt-small --output-dtype int8
# Resulting size: 125MB → 31MB (4x reduction)
```

## Data & Preprocessing

### Q: How do I prepare my own data for training?
**A:** 
1. Create text file (one sentence per line)
2. Tokenize: `python src/tokenizer/train.py --data my_data.txt`
3. Use in training: `--data-path my_data.txt`

### Q: How do I handle different languages in preprocessing?
**A:** Use language tags:
```python
en_text = "<en> Hello world"
banso_text = "<banso> Njòbɔ́ kɩ́"

tokens_en = tokenizer.encode(en_text)
tokens_banso = tokenizer.encode(banso_text)
```

### Q: How large should my training corpus be?
**A:** Recommended sizes:
- **Minimum**: 50M tokens (barely functional)
- **Good**: 500M-1B tokens (decent quality)
- **Optimal**: 5B+ tokens (Chinchilla-optimal for 125M model)

## Community & Support

### Q: How do I report a bug?
**A:** Create an issue on [GitHub](https://github.com/iwstechnical/MarkGPT-LLM-Curriculum/issues) with:
- Title describing the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python, CUDA versions)

### Q: How do I contribute code?
**A:** See [COMMUNITY_GUIDELINES.md](COMMUNITY_GUIDELINES.md) for:
- Code style requirements
- Pre-commit hook setup
- Pull request process
- Testing guidelines

### Q: Where can I get help?
**A:** 
- **Issues**: GitHub Issues for bugs
- **Discussions**: GitHub Discussions for questions
- **Email**: iwstechnical@gmail.com for general inquiries
- **Notebook Examples**: `/notebooks/` for tutorials

### Q: Is MarkGPT production-ready?
**A:** MarkGPT v1.0 is suitable for:
- ✅ Educational use and learning
- ✅ Research projects
- ✅ Small-scale production (internal tools)
- ⚠️ Public-facing applications (use quantized, add safeguards)
- ❌ Mission-critical systems (use established providers)

---

**FAQ Version**: 2.0
**Last Updated**: 2024
**Maintained by**: MarkGPT Community
