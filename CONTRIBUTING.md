# Contributing to MarkGPT LLM Curriculum

Thank you for your interest in contributing to the MarkGPT LLM Curriculum! This guide covers everything you need to know about contributing, whether you're fixing bugs, adding exercises, or improving documentation.

## Table of Contents

1. [Quick Links for Contributors](#quick-links-for-contributors)
2. [Code of Conduct](#code-of-conduct)
3. [Getting Started](#getting-started)
4. [Fork-and-Branch Workflow](#fork-and-branch-workflow)
5. [Conventional Commits](#conventional-commits)
6. [Code Style](#code-style)
7. [Docstring Standards](#docstring-standards)
8. [Pull Request Checklist](#pull-request-checklist)
9. [For Non-Technical Contributors](#for-non-technical-contributors)

---

## 🚀 Quick Links for Contributors

### New to Contributing? Start Here!

1. **[contributors/CONTRIBUTORS_GUIDE.md](contributors/CONTRIBUTORS_GUIDE.md)** ⭐ **START HERE FOR BEGINNERS**
   - Complete step-by-step guide with no assumptions
   - Perfect for first-time open source contributors
   - Includes troubleshooting for common Git issues

2. **[BEST_PRACTICES.md](BEST_PRACTICES.md)** — Quality standards and best practices
   - Code style guidelines
   - Jupyter notebook best practices
   - Educational content guidelines
   - Module-specific advice

3. **[CONTRIBUTORS.md](CONTRIBUTORS.md)** — Hall of fame and contribution tiers
   - See previous contributors
   - Get recognized for your work
   - Learn about contribution levels

### Different Types of Contributors

**Student or First-Timer?**
→ Read [contributors/CONTRIBUTORS_GUIDE.md](contributors/CONTRIBUTORS_GUIDE.md) (designed for you!)

**Experienced Developer?**
→ Jump to [Fork-and-Branch Workflow](#fork-and-branch-workflow)

**Educator or Language Expert?**
→ See [For Non-Technical Contributors](#for-non-technical-contributors)

**Want Best Practices?**
→ Check [BEST_PRACTICES.md](BEST_PRACTICES.md)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors are expected to:

- Be respectful and constructive in all interactions
- Welcome learners and educators from all backgrounds
- Respect intellectual property and cultural knowledge, especially regarding Lamnso' language and Cameroon heritage
- Follow all local and international laws regarding data and language use

Violations can be reported to iwstechnical@gmail.com.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git installed and configured
- Basic familiarity with GitHub/GitLab workflow

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/MarkGPT-LLM-Curriculum.git
cd MarkGPT-LLM-Curriculum

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify setup
python scripts/verify_setup.py
```

---

## Fork-and-Branch Workflow

### Step 1: Create a Fork

Fork the repository on GitHub (top-right "Fork" button). This creates your own copy where you have write access.

### Step 2: Clone Your Fork

```bash
git clone https://github.com/YOUR-USERNAME/MarkGPT-LLM-Curriculum.git
cd MarkGPT-LLM-Curriculum
```

### Step 3: Add Upstream Remote

Keep your fork in sync with the main repository:

```bash
git remote add upstream https://github.com/iwstechnical/MarkGPT-LLM-Curriculum.git
git fetch upstream
```

### Step 4: Create a Feature Branch

Always create a new branch for your changes. Use a descriptive name:

```bash
# Sync with upstream first
git fetch upstream
git checkout upstream/main

# Create your feature branch
git checkout -b feat/module-05-tokenization-lesson
```

**Branch naming conventions:**
- `feat/` — new feature or lesson
- `fix/` — bug fix
- `docs/` — documentation update
- `test/` — test addition
- `refactor/` — code refactoring
- `chore/` — maintenance, config updates

### Step 5: Make Your Changes

Edit files, add tests, update documentation. See the [Code Style](#code-style) section below.

### Step 6: Commit Your Changes

Use conventional commits (detailed below). Make atomic commits — one logical unit per commit.

```bash
git add .
git commit -m "feat(module-05): add tokenization deep-dive lesson with BPE worked example"
```

### Step 7: Push to Your Fork

```bash
git push origin feat/module-05-tokenization-lesson
```

### Step 8: Create a Pull Request

Go to the main repository on GitHub and click "New Pull Request". Select your branch. Fill in:

- **Title:** Matches your commit message format
- **Description:** Explains what changed and why (reference any issues with #123)
- **Tests:** Confirm you've run the test suite
- **Breaking Changes:** Note any if applicable

---

## Conventional Commits

All commits must follow the conventional commit format. This enables automated versioning and generates readable changelogs.

### Format

```
<type>(scope): <short imperative description>

<optional longer body explaining why and impact>
```

### Type

- **feat** — A new feature (e.g., new lesson, new exercise file)
- **fix** — A bug fix
- **docs** — Documentation changes (not code)
- **test** — Adding or updating tests
- **refactor** — Code restructuring without behavior change
- **style** — Formatting, missing semicolons, etc. (no logic change)
- **chore** — Build, dependency, config updates

### Scope

The scope identifies which part of the project is affected:

- `model` — Core model architecture (src/model/)
- `tokenizer` — Tokenization pipeline (src/tokenizer/)
- `training` — Training loop and related utilities (src/training/)
- `module-XX` — Specific module lesson/exercise content
- `docs` — Documentation files
- `test` — Test files
- `scripts` — Standalone scripts (scripts/)

### Examples

```
feat(model): add LoRA layer implementation for parameter-efficient fine-tuning

fix(tokenizer): handle edge case when vocabulary is empty

docs(module-03): add comprehensive backpropagation lesson with visuals

test(training): add smoke test for distributed training stub

refactor(scripts): unify data downloading and verification logic

chore: update pre-commit hook versions to latest stable
```

### Body Guidelines

The commit body is optional but encouraged for non-trivial changes:

```
feat(module-06): add RoPE positional encoding implementation

Implement RotaryEmbedding class following Su et al. (2021). This provides
superior extrapolation compared to absolute positional embeddings.

- Add RotaryEmbedding class with precomputed frequency matrix
- Add apply_rotary_emb() utility function
- Add config option to use_rope for experiments
- Include verification test that rotation is linear in position difference

Fixes #42
```

---

## Code Style

### Python Code Formatting

All Python code must be formatted with **Black** and pass **Ruff** linting. Pre-commit hooks will enforce this automatically.

#### Black Configuration

- Line length: 100 characters
- String quotes: Double quotes preferred
- Indentation: 4 spaces

#### Ruff Configuration

Run:
```bash
ruff check src/
ruff check tests/
```

Common issues:
- Unused imports: Use `ruff check --fix` to remove them
- Undefined names: Ensure proper imports
- Line too long: Break into multiple lines or use Black

### Type Hints

All functions should have type annotations:

```python
from typing import Tuple, Optional, List
import torch

def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass of the model.
    
    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)
        attention_mask: Optional mask of shape (batch_size, seq_len)
    
    Returns:
        Tuple of (logits, loss) tensors
    """
    ...
    return logits, loss
```

### Naming Conventions

- **Variables:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private methods:** `_leading_underscore`

### Comments

- Use comments to explain **why**, not what
- Avoid obvious comments like `# increment i`
- Use docstrings for all public functions and classes (see next section)

---

## Docstring Standards

All public functions, classes, and modules must have docstrings in **Google style**.

### Function Template

```python
def compute_perplexity(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> float:
    """Compute per-token perplexity on a dataset.
    
    This metric measures how "surprised" the model is by the test data. Lower
    is better. Perplexity of exp(1.0) means the model assigns ~37% probability
    to the actual next token.
    
    Args:
        model: A language model with a forward() method that returns logits
        dataloader: A DataLoader yielding (input_ids, target_ids) batches
        device: Device to run inference on ("cpu" or "cuda")
    
    Returns:
        The average perplexity across all batches
    
    Raises:
        ValueError: If dataloader is empty
        RuntimeError: If model fails forward pass
    
    Example:
        >>> ppl = compute_perplexity(model, val_loader, device="cuda")
        >>> print(f"Validation perplexity: {ppl:.2f}")
    
    References:
        - Jurafsky & Martin (2023): Speech and Language Processing, Ch. 3
        - Merity et al. (2017): "Regularizing and Optimizing LSTM Language Models"
    """
    ...
```

### Class Template

```python
class BibleDataset(torch.utils.data.Dataset):
    """Memory-mapped dataset for Bible corpus text.
    
    Loads pre-tokenized Bible text from a binary .bin file and yields
    (input_ids, target_ids) pairs of configurable length.
    
    Attributes:
        data_file: Path to the .bin file
        block_size: Context length (input sequence length)
        data: Memory-mapped file handle
        length: Total number of sequences available
    
    Example:
        >>> dataset = BibleDataset("data/processed/kjv_train.bin", block_size=512)
        >>> loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        >>> for x, y in loader:
        ...     print(x.shape, y.shape)  # (32, 512), (32, 512)
    """
    ...
```

### Module Template

```python
"""Bible corpus data loading and processing utilities.

This module provides:
- BibleDataset: Memory-mapped dataset for efficient loading of large corpora
- create_dataloaders(): Factory function to create train/val/test splits
- compute_fertility(): Tokenizer efficiency metric

The Bible corpus is stored as binary files of uint16 token IDs for fast loading.
"""
```

---

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] **Branch is up to date** with upstream/main
- [ ] **Tests pass**: `pytest tests/ -v`
- [ ] **Code is formatted**: `black src/ tests/` and `ruff check --fix src/ tests/`
- [ ] **Type checking passes**: `mypy src/` (if applicable)
- [ ] **Docstrings are complete** (Google style, with References section)
- [ ] **Commits are conventional** (proper type/scope/message)
- [ ] **No large files committed** (check .gitignore)
- [ ] **Documentation updated**: docs/, module README, docstrings
- [ ] **For lessons**: Learning objectives clear, prerequisites listed
- [ ] **For exercises**: TODO sections clear, solutions/ directory included
- [ ] **CHANGELOG.md updated** with your changes

### Before Final Submission

```bash
# Run all checks
pytest tests/ -v
black src/ tests/ --check
ruff check src/ tests/
mypy src/

# List commits in your branch
git log upstream/main..HEAD --oneline
```

---

## For Non-Technical Contributors

### Lamnso' Language Experts

We're actively seeking native speakers of Lamnso' (Nso') to:

1. **Review linguistic content** in Module 09
2. **Verify corpus correctness** — flag any mistransliterated or incorrect passages  
3. **Add cultural context** — contribute essays on language history and usage
4. **Record audio samples** — for future voice module
5. **Collect additional texts** — proverbs, songs, contemporary writing

**Your contribution process:**

- Create an issue titled "Community contribution: [topic]"
- Discuss scope and timeline with maintainers
- Receive guidance on file format and attribution
- Receive equal credit in CONTRIBUTORS.md

### Educators

Help expand the curriculum by:

1. **Adapting for your context** — translate into local languages, adjust for your student population, create culturally-relevant examples
2. **Adding examples** — real-world applications of tokenization, attention, etc.
3. **Creating assessment rubrics** — for evaluating learner projects
4. **Writing reflection prompts** — thought-provoking questions for modules

### Community Contributors

We welcome contributions in:

- **Lesson design** — following the template in docs/LESSON_TEMPLATE.md
- **Exercise creation** — coding problems with solutions, difficulty levels
- **Jupyter notebooks** — interactive tutorials, visualisations
- **Data sources** — documentation of additional Bible translations, Banso texts
- **Translations** — translate lessons or documentation into other languages

---

## Review Process

All contributions go through code review:

1. **Automated checks** — tests, linting, type checking must pass
2. **Community review** — at least one maintainer reviews the PR
3. **Feedback cycle** — responding courteously to review comments
4. **Final approval** — maintainer approves and merges

We aim to review PRs within 1 week. 

---

## Questions?

- **GitHub Issues**: For bug reports and feature requests
- **Discussion Threads**: For general questions and ideas  
- **Email**: iwstechnical@gmail.com for sensitive topics or direct mentorship

Thank you for being part of the MarkGPT community! 🙏
