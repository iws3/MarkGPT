# Community and Contributing Guidelines

## Welcome to MarkGPT!

MarkGPT is an open-source curriculum for building LLMs with a focus on multilingual support, especially for underrepresented languages like Lamnso' (Banso).

## Core Values

- **Open**: Free and openly available code, models, and curriculum
- **Inclusive**: Welcoming contributors from all backgrounds and experience levels
- **Educational**: "Why before how"—emphasizing understanding over rote implementation
- **Equitable**: Revenue sharing with language communities
- **Transparent**: Clear documentation of limitations, biases, and design decisions

## Getting Involved

### Report Issues

Found a bug? Have a feature request? Use GitHub Issues:

```
Title: [BUG] Training fails on M1 Mac
Body:
Error: RuntimeError: Unsupported operation: MPS
Steps to reproduce:
1. python -m venv venv
2. pip install -r requirements.txt
3. python src/training/train.py --model-size nano

Expected: Training begins on CPU fallback
Actual: RuntimeError
```

### Submit a Pull Request

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/my-feature`
3. **Make changes** with tests and docstrings
4. **Run linting**: `make lint`
5. **Push**: `git push origin feature/my-feature`
6. **Open PR** with description of changes

### Join the Community

- **Discussions**: GitHub Discussions for questions and ideas
- **Discord** (coming soon): Real-time chat and collaboration
- **Community calls**: Regular (monthly) video calls for contributors

## Code Style

### Python Style Guide

MarkGPT follows **PEP 8** with Black formatting:

```python
# Good
def train_model(
    model: MarkGPT,
    train_loader: DataLoader,
    num_epochs: int = 10
) -> Dict[str, float]:
    """Train transformer model on dataset."""
    metrics = {}
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader)
        metrics[f"epoch_{epoch}_loss"] = loss
    return metrics

# Bad
def train_model(model,loader,epochs=10):
    # Train without type hints
    ...
```

### Type Hints

All functions must include type hints:

```python
from typing import Dict, List, Tuple, Optional
import torch

def calculate_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """Calculate perplexity from logits."""
    loss = cross_entropy(logits, targets)
    return math.exp(loss.item())
```

### Docstrings

Use Google-style docstrings:

```python
def evaluate_model(model: MarkGPT, loader: DataLoader) -> Dict[str, float]:
    """Evaluate model on validation set.
    
    Args:
        model: Transformer model to evaluate
        loader: DataLoader with validation data
        
    Returns:
        Dictionary with metrics:
            - 'loss': Cross-entropy loss
            - 'perplexity': Perplexity score
            - 'accuracy': Token accuracy (if applicable)
            
    Example:
        >>> model = MarkGPT.load_pretrained('markgpt-small')
        >>> metrics = evaluate_model(model, val_loader)
        >>> print(f"Perplexity: {metrics['perplexity']:.2f}")
    """
    pass
```

### Setup Pre-commit Hooks

```bash
# Install dependencies
pip install pre-commit black ruff mypy

# Install hooks
pre-commit install

# Now linting runs on every git commit!
```

## Testing

### Writing Tests

```python
# tests/test_markgpt.py
import unittest
import torch
from src.model.markgpt import MarkGPT, MarkGPTConfig

class TestMarkGPT(unittest.TestCase):
    
    def setUp(self):
        self.config = MarkGPTConfig(vocab_size=1000, d_model=64)
        self.model = MarkGPT(self.config)
    
    def test_forward_pass(self):
        batch = torch.randint(0, 1000, (2, 128))  # (batch, seq_len)
        logits = self.model(batch)
        self.assertEqual(logits.shape, (2, 128, 1000))
    
    def test_device_movement(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            batch = torch.randint(0, 1000, (2, 128)).cuda()
            logits = self.model(batch)
            self.assertTrue(logits.is_cuda)
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_markgpt.py::TestMarkGPT::test_forward_pass

# With coverage
pytest --cov=src --cov-report=html
```

## Documentation

### Adding Documentation

1. **Docstrings**: In every function/class
2. **Module docs**: At top of `.py` files
3. **README**: Per-directory overview
4. **Examples**: Runnable code in `/docs` or notebooks

### Building Documentation

```bash
# Generate HTML docs (requires Sphinx)
cd docs/
make html
# Open build/html/index.html
```

## Commit Message Guidelines

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (black, ruff)
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `test`: Tests
- `chore`: Tooling, dependencies

### Examples

```
feat(training): add gradient accumulation support

- Implement GradientAccumulation context manager
- Add accumulation_steps configuration
- Update training loop to use accumulation
- Add tests for gradient accumulation

Closes #42
```

```
fix(tokenizer): correct BPE merge order

Previously, merges were applied in random order, causing
non-deterministic tokenization. Now sorted by frequency.

Fixes #18
```

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (1.1.0): New features (backward compatible)
- **PATCH** (1.1.1): Bug fixes

### Creating a Release

```bash
# Tag release
git tag -a v1.0.0 -m "MarkGPT 1.0.0: Initial release"

# Push tag
git push origin v1.0.0

# Create GitHub Release with changelog
# (automated via CI/CD)
```

## Community Roles

### Contributor
- Submits code, documentation, or tests
- First PR? Welcome! We're here to help.

### Maintainer
- Reviews and merges PRs
- Manages issues
- Coordinates larger features
- If interested, reach out to: iwstechnical@gmail.com

### Language Advocate
- Represents language community (e.g., Banso speakers)
- Provides cultural/linguistic feedback
- Ensures equitable treatment of languages
- Participates in bi-annual advisory board meetings

## Code of Conduct

We're committed to providing a safe, respectful environment for everyone.

### Our Pledge
We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone.

### Expectations
- Be respectful and inclusive
- Provide constructive feedback
- Accept criticism gracefully
- Focus on what's best for the community

### Consequences
Violations of the code of conduct will result in:
1. Warning and private discussion
2. Temporary suspension from participation
3. Permanent removal if necessary

**Report Code of Conduct violations to**: conduct@markgpt.dev

## Roadmap

**Q1 2024**
- [x] Release MarkGPT v1.0
- [ ] 100+ curriculum lessons complete

**Q2 2024**
- [ ] Mobile deployment guide
- [ ] Advanced quantization techniques
- [ ] Community ambassadors program

**Q3 2024**
- [ ] Support for 3rd language
- [ ] Production deployment case studies

**Q4 2024**
- [ ] MarkGPT 1.1 with model improvements
- [ ] Research paper submission

## Contact Us

- **Email**: iwstechnical@gmail.com
- **GitHub Issues**: For bugs and features
- **Discussions**: For questions and ideas
- **Discord**: (invite TBA) Real-time chat

---

**Guide Version**: 1.0
**Last Updated**: 2024
**Maintained by**: MarkGPT Community Team
