# MarkGPT FAQ

## Getting Started

### Q: Do I need a GPU to run MarkGPT?

**A:** No, but it helps significantly. MarkGPT works on CPU, but training is ~50x slower. For the curriculum:
- **Nano model**: Trains in 2-5 minutes on GPU, ~2 hours on CPU
- **Small model**: Trains in 30 minutes on GPU, ~24 hours on CPU

If you don't have a GPU:
- Use Google Colab (free GPU)
- Run smaller models (Nano instead of Small)
- Reduce training steps in configs

### Q: Which GPU do I need?

**A:** Any NVIDIA GPU with 2GB+ VRAM works:
- **RTX 3060** (12GB): Perfect
- **RTX 2060** (6GB): Works, batch size limited
- **GTX 1660Ti** (6GB): Works
- Mac M1/M2: Works with MPS (torch.device('mps'))

Check your GPU:
```python
import torch
print(torch.cuda.get_device_name())
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Q: What if my training keeps failing?

**A:** Common issues:

1. **Out of Memory**: Reduce batch size
   ```bash
   python train.py --batch_size 16  # Instead of 32
   ```

2. **CUDA Errors**: Update PyTorch
   ```bash
   pip install --upgrade torch torchvision torchaudio
   ```

3. **Loss goes to NaN**: Learning rate too high
   ```bash
   python train.py --learning_rate 0.0001  # Lower it
   ```

4. **Module not found**: Run from root directory
   ```bash
   cd MarkGPT-LLM-Curriculum
   python src/training/train.py  # NOT from src/ subdirectory
   ```

## Models

### Q: What's the difference between Nano, Small, and Base?

| Model | Params | d_model | L | Heads | Speed | Quality |
|-------|--------|---------|---|-------|-------|---------|
| **Nano** | 2M | 128 | 4 | 2 | ~2 min | Basic |
| **Small** | 10M | 256 | 6 | 4 | ~30 min | Good |
| **Base** | 85M | 512 | 12 | 8 | ~4 hours | Excellent |

- **Start with Nano** to verify setup
- **Use Small** for balance of speed/quality
- **Use Base** for best results (if GPU available)

### Q: Can I use a pre-trained GPT-2?

**A:** Yes! MarkGPT's architecture is GPT-2 compatible:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Now fine-tune with LoRA on Bible data
```

But the curriculum is designed around MarkGPT to teach every detail.

## Data

### Q: Can I add my own data?

**A:** Yes! Create a .txt file with one sentence per line:
```
In the beginning was the Word.
And the Word was with God.
And the Word was God.
...
```

Then preprocess:
```bash
python scripts/preprocess_bible.py \
    --input my_data.txt \
    --vocab_size 4096
```

### Q: How much data do I need?

**A:**
- Minimum: 1M tokens (~5K short documents)
- Good: 10M tokens (~50K documents)  
- Excellent: 100M+ tokens (entire corpus)

The Bible has ~780K tokens—enough for educational purposes, but much less than production models.

### Q: How do I handle non-English text?

**A:** Use `BansoPreprocessor`:
```python
from src.tokenizer.banso_preprocess import BansoPreprocessor

prep = BansoPreprocessor()
normalized_text = prep.normalize("Fə́ gǔ lɯ́m ñvæ wæ")
```

Or combine English + Banso:
```python
from src.training.mixed_language_dataset import MixedLanguageDataset

mixed_data = MixedLanguageDataset(
    english_tokens, 
    banso_tokens,
    english_ratio=0.8  # 80% English, 20% Banso
)
```

## Training

### Q: My model generates the same word repeatedly—what's wrong?

**A:** This happens if:
1. **Model is undertrained**: Train for more steps
2. **Temperature too low**: Use temperature=0.7-1.0 instead of 0.0
3. **Vocabulary too small**: Use vocab_size=4096 instead of 256

### Q: How do I know if my model is learning?

**A:** Monitor training loss:
- ✅ **Good**: Loss smoothly decreases each epoch
- ⚠️ **Caution**: Loss plateaus (learning rate too low)
- ❌ **Bad**: Loss increases or becomes NaN (learning rate too high)

```python
import matplotlib.pyplot as plt

# After training
losses = model_training_results['losses']
plt.plot(losses)
plt.xlabel('Training step')
plt.ylabel('Loss')
plt.show()
```

### Q: Should I use LoRA or full fine-tuning?

**A:** Use LoRA unless:
- ✅ Use LoRA if: Limited resources, multiple tasks, adapter sharing
- ✅ Use full fine-tuning if: Unlimited compute, single task, maximum quality needed

LoRA gives ~95% of full fine-tuning performance with 100x fewer parameters.

## Inference

### Q: How do I generate longer text?

**A:** Increase `max_length`:
```python
model.generate(tokens, max_length=500)  # Instead of 50
```

But note:
- Longer generation = slower
- Quality degrades for very long sequences (~1000+ tokens)
- Use text summarization for repeated content filtering

### Q: Can I export the model to ONNX/TensorRT?

**A:** Yes, use HuggingFace's conversion tools:
```python
from transformers.onnx import export_pytorch_model

export_pytorch_model(model, "model.onnx", model_name="markgpt")
```

Then deploy with:
- **ONNX Runtime**: Cross-platform inference
- **TensorRT**: NVIDIA optimization (faster on GPU)
- **TensorFlow Lite**: Mobile deployment

## Language & Culture

### Q: Why focus on the Bible and Banso?

**A:** Because:
1. **Parallel texts** → Learn multilingual models
2. **Cultural significance** → Balance Western bias in NLP
3. **Educational** → Teach both English and low-resource language techniques
4. **Respect** → Partner with Banso community, share benefits

### Q: Can I use other language pairs?

**A:** Absolutely! Replace with any languages:
```python
# Arabic-French, Swahili-English, Mandarin-Spanish, etc.
```

The curriculum generalizes. The choice of Bible + Banso is illustrative.

### Q: Am I helping or harming indigenous language communities?

**A:** Good question! LLMs are a **tool**, not a solution:
- ✅ They create digital heritage and accessibility tools
- ❌ They can't replace live language practice or community transmission
- ❌ Language decline is driven by economics/policy, not tech
- ✅ Most important: **Share benefits** with community, don't profit alone

Read: Nettle & Romaine (2000), *Vanishing Voices*.

## Troubleshooting

### Q: I'm getting "ModuleNotFoundError: No module named 'src'"

**A:** Run from repo root:
```bash
cd MarkGPT-LLM-Curriculum
python src/training/train.py  # ✓
```

NOT:
```bash
cd MarkGPT-LLM-Curriculum/src
python training/train.py  # ✗
```

### Q: "RuntimeError: CUDA out of memory"

**A:** Solutions in order:

```python
# 1. Clear cache
torch.cuda.empty_cache()

# 2. Reduce batch size
# 3. Use gradient checkpoint (trade speed for memory)
model = torch.utils.checkpoint.checkpoint(model)
# 4. Use mixed precision
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model(inputs)
```

### Q: My Jupyter notebook kernel keeps dying

**A:** Usually memory. Try:
```python
# Reduce batch size
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Or use CPU for exploration
device = 'cpu'
```

## Contributing

### Q: How do I submit improvements to MarkGPT?

**A:** See CONTRIBUTING.md:

1. Fork repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Commit with conventional messages: `feat(model): add flash attention`
4. Push and create PR
5. All tests must pass

### Q: I found a bug!

**A:** Create an issue on GitHub with:
- [ ] Minimal reproducible example
- [ ] Expected vs. actual behavior
- [ ] Your environment (Python 3.10, PyTorch 2.0, CUDA 11.8, etc.)
- [ ] Error traceback

## References

- MarkGPT Docs: [to be created]
- PyTorch Tutorials: https://pytorch.org/tutorials/
- Hugging Face Course: https://huggingface.co/course
- Attention is All You Need: https://arxiv.org/abs/1706.03762

## Still Stuck?

- [ ] Check troubleshooting guide
- [ ] Search GitHub issues
- [ ] Post on discussions: https://github.com/yourusername/MarkGPT-LLM-Curriculum/discussions
- [ ] Email: support@markgpt.org (see README)
