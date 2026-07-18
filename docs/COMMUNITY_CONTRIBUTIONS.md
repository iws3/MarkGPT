# Community Contribution Guide

## Getting Started as a Contributor

### 1. Prerequisites

- Python 3.10+
- Git & GitHub account
- Familiarity with PyTorch/Transformers (for code contributions)
- Understanding of MarkGPT architecture (see ARCHITECTURE.md)

### 2. Fork & Clone

```bash
# Fork on GitHub: https://github.com/iwstechnical/markgpt

# Clone your fork
git clone https://github.com/YOUR_USERNAME/markgpt.git
cd markgpt

# Add upstream remote
git remote add upstream https://github.com/iwstechnical/markgpt.git

# Fetch latest
git fetch upstream
```

### 3. Create Feature Branch

```bash
# Create branch from latest main
git fetch upstream
git checkout -b feature/your-feature-name upstream/main

# Branch naming:
# - feature/short-description (new feature)
# - fix/bug-short-description (bug fix)
# - docs/topic (documentation)
# - perf/optimization-description (performance)
```

---

## Contribution Types

### A. Code Contributions

**Target**: Core library improvements (model, training, tokenizer)

#### Example: Add New Loss Function

1. **Create file**: `src/training/losses.py`

```python
import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """Contrastive loss for bilingual embeddings."""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings_a, embeddings_b, labels):
        """
        Args:
            embeddings_a: (batch_size, hidden_dim)
            embeddings_b: (batch_size, hidden_dim)
            labels: (batch_size,) - 1 if same, 0 if different
        """
        
        # Normalize
        embeddings_a = F.normalize(embeddings_a, dim=1)
        embeddings_b = F.normalize(embeddings_b, dim=1)
        
        # Similarity
        logits = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature
        
        # Loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
```

2. **Add tests**: `tests/test_losses.py`

```python
import pytest
import torch
from src.training.losses import ContrastiveLoss

def test_contrastive_loss_shape():
    """Test loss output shape."""
    
    loss_fn = ContrastiveLoss()
    
    embeddings_a = torch.randn(32, 768)
    embeddings_b = torch.randn(32, 768)
    labels = torch.randint(0, 2, (32,))
    
    loss = loss_fn(embeddings_a, embeddings_b, labels)
    
    assert loss.shape == torch.Size([])  # Scalar

def test_contrastive_loss_backward():
    """Test gradient flow."""
    
    loss_fn = ContrastiveLoss()
    
    embeddings_a = torch.randn(32, 768, requires_grad=True)
    embeddings_b = torch.randn(32, 768, requires_grad=True)
    labels = torch.randint(0, 2, (32,))
    
    loss = loss_fn(embeddings_a, embeddings_b, labels)
    loss.backward()
    
    assert embeddings_a.grad is not None
    assert embeddings_b.grad is not None
```

3. **Run tests**:

```bash
pytest tests/test_losses.py -v
```

4. **Commit**:

```bash
git add src/training/losses.py tests/test_losses.py
git commit -m "feat: add contrastive loss for bilingual embeddings"
```

### B. Documentation Contributions

**Target**: Guide improvements, tutorial content, examples

#### Example: Create Fine-Tuning Tutorial

1. **Create**: `docs/tutorials/finetuning_sentiment.md`

```markdown
# Fine-tuning MarkGPT for Sentiment Analysis

## Overview
Learn how to adapt MarkGPT-Small for bilingual sentiment classification.

## Dataset Preparation

```python
# Load sentiment data
en_data = load_dataset('sst2', split='train')
banso_data = load_banso_sentiment_dataset()

# Combine
mixed_data = combine_and_balance(en_data, banso_data, ratio=0.7)
```

## Training

```python
from markgpt_finetuning import SentimentTuner

tuner = SentimentTuner(
    model='markgpt-small',
    learning_rate=2e-5,
    batch_size=16
)

metrics = tuner.train(mixed_data, epochs=3)
print(f"Final Accuracy: {metrics['accuracy']:.2f}")
```

## Evaluation

```python
# Test on held-out set
results = tuner.evaluate(test_data)
```
```

2. **Commit**:

```bash
git add docs/tutorials/finetuning_sentiment.md
git commit -m "docs: add sentiment fine-tuning tutorial"
```

### C. Research Contributions

**Target**: Novel experiments, extensions, applications

#### Example: Publish Multilingual Evaluation Results

1. **Create**: `research/multilingual_evaluation.md`

```markdown
# Multilingual Capabilities of MarkGPT

## Abstract
Evaluation of MarkGPT's English-Banso bilingual performance on 5 downstream tasks.

## Methodology

### Models Tested
- MarkGPT-Nano (70M)
- MarkGPT-Small (200M)
- MarkGPT-Base (500M)

### Benchmark Composition
- Translation (500 parallel EN-BANSO pairs)
- Sentiment (200 labeled examples)
- Q&A (150 context-question-answer triplets)

## Results

| Task | Nano | Small | Base |
|------|------|-------|------|
| Translation BLEU | 18.2 | 24.5 | 31.1 |
| Sentiment Acc | 68% | 75% | 82% |
| QA F1 | 42.1 | 55.3 | 68.7 |

## Conclusion
MarkGPT-Base demonstrates strong bilingual capabilities...
```

2. **Include code**: `research/scripts/eval_multilingual.py`

3. **Commit**:

```bash
git add research/multilingual_evaluation.md research/scripts/eval_multilingual.py
git commit -m "research: evaluate multilingual capabilities"
```

---

## Pull Request Process

### Step 1: Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### Step 2: Create Pull Request

- Title: Clear, concise (e.g., "Add contrastive loss for embeddings")
- Description:
  - What problem does this solve?
  - Implementation details
  - Testing performed
  - Related issues (if any)

### Step 3: PR Template

```markdown
## Description
Brief description of changes.

## Related Issues
Fixes #123

## Changes
- Added contrastive loss function
- Added tests with 95% coverage
- Updated documentation

## Testing
- Unit tests: ✅ All pass
- Integration tests: ✅ Pass
- Manual testing: ✅ Verified on V100

## Checklist
- [x] Code follows style guide
- [x] Self-review completed
- [x] Comments added for complex logic
- [x] Documentation updated
- [x] Tests added/updated
- [x] No new warnings generated
```

### Step 4: Review Process

- Maintainers review within 2-5 days
- May request changes (in review comments)
- Address feedback and commit additional fixes
- Re-request review

### Step 5: Merge

- Squash commits for clean history (usually)
- PR merged to main
- Feature branch deleted

---

## Code Style & Standards

### Python

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) + MarkGPT conventions:

```python
# Good ✓
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward pass with docstring.
    
    Args:
        input_ids: Token IDs (batch_size, seq_length)
        attention_mask: Attention mask (batch_size, seq_length)
    
    Returns:
        Output logits (batch_size, seq_length, vocab_size)
    """
    ...

# Run formatter
black src/ --line-length=100

# Type check
mypy src/
```

### Documentation

```markdown
# Clear Heading

Paragraph with context.

## Subheading

- Bullet point
- Another point

```python
# Code block with syntax highlight
code_example = "formatted nicely"
```
```

---

## Testing Guidelines

### Unit Tests

```python
# tests/test_module.py

import pytest
import torch

class TestMarkGPTComponent:
    def test_initialization(self):
        """Test component initializes correctly."""
        component = ComponentClass()
        assert component is not None
    
    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        component = ComponentClass()
        dummy_input = torch.randn(2, 512)
        output = component(dummy_input)
        assert output.shape == (2, 512)
    
    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
    def test_various_batch_sizes(self, batch_size):
        """Test with different batch sizes."""
        # ...
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_module.py -v

# With coverage
pytest tests/ --cov=src/ --cov-report=html
```

---

## Reporting Issues

### Bug Report Template

```markdown
## Description
Brief description of the bug.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS: Ubuntu 20.04
- Python: 3.10
- PyTorch: 2.0
- MarkGPT: 1.0-dev

## Error Log
```
<error message here>
```

## Minimal Code Example
```python
# Code that reproduces the issue
```
```

### Feature Request Template

```markdown
## Description
What new capability would you like?

## Use Case
Why is this useful? Real-world application?

## Proposed Solution
How should this work?

## Alternatives Considered
Other approaches?
```

---

## Recognition

### Contribution Tiers

**Tier 1** (1-10 lines): Git commit mention
**Tier 2** (11-100 lines): CONTRIBUTORS.md entry
**Tier 3** (100+ lines / ongoing): Core team consideration
**Tier 4** (Research publication): Co-authorship offer

### Hall of Fame

Significant contributors featured in:
- README.md "Credits" section
- Annual release announcements
- Community blog posts

---

## Support Channels

- **GitHub Discussions**: General Q&A
- **Issues**: Bug reports & feature requests
- **Discord** (coming soon): Real-time community chat
- **Email**: contributors@markgpt.dev

---

**Community Contribution Guide v1.0**
**Last Updated**: 2024
