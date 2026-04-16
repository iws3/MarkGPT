# Changelog

All notable changes to the MarkGPT LLM Curriculum project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-15

### Added

#### Infrastructure
- Comprehensive `.gitignore` for Python, PyTorch, data files, and IDE-specific directories
- MIT License with data attribution and community acknowledgments
- Initial CHANGELOG.md documenting project evolution

#### Documentation
- Contributing guide with git workflow, conventional commits, and code standards
- Pre-commit hooks configuration for black, ruff, and mypy
- Model architecture documentation with tensor shape diagrams
- Training stability guide and troubleshooting documentation
- Complete module README files with lesson indices
- Project roadmap for future curriculum directions
- Standardised lesson and exercise templates for contributors

#### Data Processing
- Python script for downloading Bible corpora (KJV, WEB) and Lamnso' texts
- Environment verification script with detailed environment checks
- Bible corpus preprocessing pipeline with binary serialization
- Banso linguistic preprocessing with dialect detection

#### Model Architecture
- MarkGPT base implementation with configurable model variants
- CausalSelfAttention with Flash Attention support (PyTorch 2.0+)
- LoRA layer implementation for parameter-efficient fine-tuning
- Rotary Position Embedding (RoPE) as alternative to learned embeddings
- Model factory functions with parameter count reporting
- Gradient flow monitoring utilities for training diagnostics

#### Tokenization
- Complete BPE tokenizer with merge history logging
- Tokenizer comparison utility with fertility benchmarking
- Extended Banso text preprocessor with dialect detection
- Vocabulary analysis notebook with cross-language comparisons
- 50 annotated Nso' proverbs dataset with cultural context

#### Training Pipeline
- Gradient accumulation support for larger effective batch sizes
- Learning rate finder utility based on Smith (2017) method
- Mixed-language pre-training pipeline (English + Banso)
- Comprehensive wandb logging with experiment tracking
- Early stopping callback with best model restoration
- Distributed training stub for multi-GPU reference
- Training resumption with complete state serialization
- Training diagnostic notebook for loss curve analysis

#### Inference & Deployment
- Multiple generation strategies: greedy, top-k, top-p (nucleus), beam search
- Sample generation with temperature control
- Gradio web interface for interactive MarkGPT demo
- HuggingFace model export functionality

#### Capstone & Evaluation
- Banso fine-tuning data preparation script
- LoRA fine-tuning pipeline on vernacular corpus
- Full evaluation suite: perplexity, BLEU, self-BLEU
- Model card template with complete sections
- Training report template for learner documentation

#### Modules & Lessons (39 lessons across Modules 2-9)
- Module 02: Python for ML, data manipulation, linear algebra, calculus for optimization
- Module 03: Neural networks, backpropagation, loss functions, optimizers
- Module 04: RNNs, LSTMs, GRUs, attention mechanisms
- Module 05: Word embeddings, sequence labeling, tokenization deep dive
- Module 06: Scaled attention, multi-head attention, transformer blocks
- Module 07: Pre-training objectives, training stability, scaling laws
- Module 08: Transfer learning, parameter-efficient fine-tuning (LoRA)
- Module 09: Banso language and culture, biblical text sourcing, dataset documentation

#### Exercises
- 18 comprehensive exercise files across Modules 2-9
- From-scratch implementations: backprop, Adam, BPE, tokenizers, attention, Trigram model (Day 5)
- Ablation studies: regularization, learning rate schedules, model variants, Laplace smoothing (Day 5)
- Analysis notebooks with visualisation and interpretation tools, including "Colorless green ideas" reflection (Day 5)
- Solution directories with complete implementations

#### Notebooks
- Getting started guide with Colab-friendly setup
- N-gram models interactive explorer
- Word embeddings with nearest-neighbour search and analogy arithmetic
- Attention mechanism step-by-step visualiser
- Training monitor with live loss curves and diagnostics
- Banso dataset explorer with corpus statistics
- MarkGPT demo with temperature sampling
- Full evaluation notebook with human evaluation rubric

#### Quality Standards
- PEP 8 code style throughout
- Type annotations on all functions
- Google-style docstrings with Args, Returns, Raises, References
- Comprehensive unit tests for data loading, tokenization, training
- All 100 commits follow conventional commit format

### Changed
- Expanded .gitignore to cover all development tools and data types
- Enhanced existing model implementations with advanced features

## [0.1.0] - 2026-01-01

### Initial Commit
- Project structure established
- README, GETTING_STARTED, SYLLABUS documents
- Initial Python package structure with src/, tests/, notebooks/
- Basic configuration templates
