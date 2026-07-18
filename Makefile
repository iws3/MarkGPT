.PHONY: help install test lint format train-nano train-small generate eval clean

help:
	@echo "MarkGPT Development Commands"
	@echo "============================"
	@echo ""
	@echo "  make install       Install project and dependencies"
	@echo "  make test          Run test suite"
	@echo "  make lint          Run linters (ruff + mypy)"
	@echo "  make format        Format code with black + ruff"
	@echo "  make train-nano    Train MarkGPT-Nano model"
	@echo "  make train-small   Train MarkGPT-Small model"
	@echo "  make generate      Generate text from trained model"
	@echo "  make eval          Run evaluation on test set"
	@echo "  make download-data Download Bible and Banso corpus"
	@echo "  make preprocess    Preprocess corpus into binary format"
	@echo "  make verify        Verify environment setup"
	@echo "  make clean         Remove build artifacts and cache"
	@echo ""

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

train-nano:
	python src/training/train.py --config configs/markgpt_nano.yaml --model-size nano

train-small:
	python src/training/train.py --config configs/markgpt_small.yaml --model-size small

generate:
	python src/inference/generate.py \
		--checkpoint checkpoints/markgpt_small_latest.pt \
		--prompt "In the beginning" \
		--max-length 100 \
		--temperature 0.8

eval:
	python capstone/evaluation/evaluate_markgpt.py

download-data:
	python scripts/download_data.py --all --verify

preprocess:
	python scripts/preprocess_bible.py \
		--input data/raw/kjv_bible.txt \
		--output data/processed \
		--vocab-size 8000

verify:
	python scripts/verify_setup.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
	rm -rf dist/ build/
	@echo "✓ Clean complete"
