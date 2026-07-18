# Quick Reference

## Common Commands

```bash
# Setup
pip install -r requirements.txt
python scripts/verify_setup.py

# Data
python scripts/download_data.py
python scripts/preprocess_bible.py

# Training
python src/training/train.py --config configs/markgpt_nano.yaml
python src/training/train.py --config configs/markgpt_small.yaml

# Testing
make test
pytest tests/ -v --tb=short

# Lint & Format
make lint
make format
```

## Key Files

- `src/model/markgpt.py` - Core model
- `src/training/train.py` - Training script
- `configs/markgpt_nano.yaml` - Nano config (2M params)
- `configs/markgpt_small.yaml` - Small config (10M params)
- `tests/` - Unit tests

## Configs

| Model | Parameters | d_model | Layers | Speed |
|-------|-----------|---------|--------|-------|
| Nano | 2M | 128 | 4 | 2 min |
| Small | 10M | 256 | 6 | 30 min |
| Base | 85M | 512 | 12 | 4 hrs |

