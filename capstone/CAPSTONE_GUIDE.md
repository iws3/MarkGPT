# MarkGPT Capstone: Step-by-Step Implementation

## Phase 1: Setup (Day 1)

### Objectives
- [ ] Clone MarkGPT repository
- [ ] Set up Python environment
- [ ] Verify installation
- [ ] Run first test

### Steps

```bash
# 1. Clone repository
git clone https://github.com/yourusername/MarkGPT-LLM-Curriculum.git
cd MarkGPT-LLM-Curriculum

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python scripts/verify_setup.py
# Expected output: "✓ Setup OK: Python 3.10+, PyTorch, CUDA available, ..."
```

## Phase 2: Data Preparation (Days 2-3)

### Objectives
- [ ] Download Bible and Banso data
- [ ] Preprocess text into tokenized format
- [ ] Inspect dataset structure

### Steps

```bash
# 1. Download data
python scripts/download_data.py
# Downloads: KJV Bible, WEB Bible, Lamnso' corpus (~500MB)

# 2. Preprocess to binary format
python scripts/preprocess_bible.py \
    --input data/raw/bible_kjv.txt \
    --output data/processed/bible_tokens.bin \
    --vocab_size 4096

# 3. Verify dataset
python -c "
import numpy as np
data = np.memmap('data/processed/bible_tokens.bin', dtype=np.uint16, mode='r')
print(f'Dataset size: {len(data):,} tokens')
print(f'Sample tokens: {data[:50]}')
"
```

## Phase 3: Train MarkGPT (Days 4-7)

### Objectives
- [ ] Understand model config
- [ ] Train Nano model
- [ ] Monitor training
- [ ] Evaluate perplexity

### Steps

```bash
# 1. Train Nano model (small, fast ~2 minutes on GPU)
python src/training/train.py \
    --config configs/markgpt_nano.yaml \
    --data_path data/processed/bible_tokens.bin \
    --output_dir checkpoints/nano \
    --num_steps 1000 \
    --batch_size 32 \
    --learning_rate 0.001

# 2. Monitor with tensorboard
tensorboard --logdir checkpoints/nano

# 3. Evaluate
python src/utils/evaluation.py \
    --model_path checkpoints/nano/final.pt \
    --test_path data/processed/bible_tokens_test.bin
```

## Phase 4: Fine-tuning (Days 8-9)

### Objectives
- [ ] Apply LoRA for efficient fine-tuning
- [ ] Fine-tune on Banso text
- [ ] Compare English vs. Banso performance

### Steps

```bash
# 1. Fine-tune with LoRA on Banso
python src/training/train.py \
    --config configs/markgpt_nano.yaml \
    --checkpoint checkpoints/nano/final.pt \
    --use_lora \
    --lora_rank 8 \
    --data_path data/processed/banso_tokens.bin \
    --output_dir checkpoints/nano_banso_lora \
    --num_steps 500

# 2. Merge LoRA weights
python -c "
from src.model.lora import merge_lora_weights
merge_lora_weights('checkpoints/nano_banso_lora/final.pt', 
                   'checkpoints/nano_banso_merged.pt')
"
```

## Phase 5: Inference and Generation (Day 10)

### Objectives
- [ ] Load trained model
- [ ] Generate text
- [ ] Create inference API

### Steps

```python
# inference_test.py
from src.model.markgpt import MarkGPT, MarkGPTConfig
from src.tokenizer.tokenizer import Tokenizer

# Load model
config = MarkGPTConfig.from_pretrained("configs/markgpt_nano.yaml")
model = MarkGPT(config)
model.load_state_dict(torch.load("checkpoints/nano/final.pt"))
model.eval()

# Load tokenizer
tokenizer = Tokenizer.load("data/tokenizer.model")

# Generate
prompt = "In the beginning was"
tokens = tokenizer.encode(prompt)
generated = model.generate(
    tokens=tokens,
    max_length=50,
    temperature=0.7
)
text = tokenizer.decode(generated)
print(text)
```

## Results Expected

- **MarkGPT Nano** (~2M params):
  - Training time: 2-5 minutes (GPU)
  - Validation loss: 4.5-5.0 (on Bible)
  - Perplexity: 90-150
  - Generated text: Coherent but repetitive

- **Fine-tuned on Banso** (LoRA, 8 adapters):
  - Fine-tuning time: 1 minute (GPU)
  - Additional parameters: ~65K (vs 2M)
  - Language switching: Instant adapter swap

## Checkpoint: Self-Assessment

- [ ] Model trains without errors
- [ ] Loss decreases during training
- [ ] Generated text is coherent (not random)
- [ ] LoRA merging reduces model size (checkpoint file smaller after merge)
- [ ] Can switch between English/Banso with LoRA loading

## References

- MarkGPT source: https://github.com/yourusername/MarkGPT-LLM-Curriculum
- LoRA paper: https://arxiv.org/abs/2106.09714
- Hugging Face Transformers: https://huggingface.co/transformers/

## Next Steps

1. Deploy model as REST API (Flask/FastAPI)
2. Build web UI for text generation
3. Fine-tune on additional languages
4. Submit to HuggingFace Model Hub
