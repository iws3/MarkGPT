# Quick Start

Get up and running in 5 minutes.

## 1. Install (2 min)

```bash
git clone https://github.com/yourusername/MarkGPT-LLM-Curriculum.git
cd MarkGPT-LLM-Curriculum
pip install -r requirements.txt
python scripts/verify_setup.py
```

## 2. Download Data (1 min)

```bash
python scripts/download_data.py
```

## 3. Train Nano Model (2 min on GPU)

```bash
python src/training/train.py --config configs/markgpt_nano.yaml
```

## 4. Generate Text

```python
from src.model.markgpt import MarkGPT, MarkGPTConfig
import torch

model = MarkGPT(MarkGPTConfig.nano())
model.load_state_dict(torch.load("checkpoints/latest.pt"))
model.eval()

tokens = [1, 2, 3]  # "In the beginning"
generated = model.generate(tensor, max_length=50)
print(generated)
```

Done! í¾‰
