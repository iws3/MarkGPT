# Contributing to MarkGPT

We welcome contributions! This document explains how to contribute.

## Code Style

- **Python**: PEP 8, Google-style docstrings
- **Type hints**: All functions must have type annotations
- **Tests**: Every new module must have tests (pytest)
- **Documentation**: Docstrings explain "why", not just "what"

## Development Workflow

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/MarkGPT-LLM-Curriculum.git`
3. **Create branch**: `git checkout -b feature/your-feature`
4. **Make changes** (see code style guidelines above)
5. **Test**: `make test` - all tests must pass
6. **Commit**: Use conventional messages (see below)
7. **Push**: `git push origin feature/your-feature`
8. **Pull Request**: Submit PR with description

## Conventional Commit Format

All commits must follow this format:

```
<type>(scope): <message>

<optional longer explanation>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (no logic change)
- `refactor`: Code restructuring
- `test`: Adding/fixing tests
- `chore`: Build, dependencies, etc.

**Scope:** Module or component affected
- `(model)`: src/model/
- `(training)`: src/training/
- `(tokenizer)`: src/tokenizer/
- `(lessons)`: modules/*/lessons/
- `(exercises)`: modules/*/exercises/
- etc.

**Examples:**
```
feat(model): add flash attention support
fix(training): correct gradient accumulation step calculation
docs(lessons): improve attention explanation with diagrams
test(tokenizer): add Unicode handling tests
```

## Running Tests

```bash
# All tests
make test
# or
python -m pytest tests/ -v

# Specific test file
pytest tests/test_markgpt.py -v

# Specific test
pytest tests/test_markgpt.py::TestMarkGPT::test_forward_pass -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Pre-commit Hooks

We use black, ruff, and mypy for code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install

# Now every commit automatically:
# - Formats with black
# - Lints with ruff
# - Type checks with mypy
```

## Documentation

### Module Documentation

Each module should have:
- `lessons/*.md` - Conceptual explanations
- `exercises/*.md` - Practical coding tasks  
- `README.md` - Module overview

### Docstring Example

```python
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention (Vaswani et al., 2017).
    
    Attention weights are computed as softmax(Q @ K^T / sqrt(d_k)) @ V.
    This mechanism allows each position to attend to all previous positions,
    which is crucial for language modeling (causal attention).
    
    Args:
        query: Queries, shape (batch, seq_len, d_model)
        key: Keys, shape (batch, seq_len, d_model)
        value: Values, shape (batch, seq_len, d_model)
        mask: Binary mask for attention weights (optional).
              1 = attend, 0 = mask out. Shape (seq_len, seq_len).
    
    Returns:
        output: Attention-weighted values, shape (batch, seq_len, d_model)
        attention_weights: Softmax weights, shape (batch, seq_len, seq_len)
    
    Raises:
        ValueError: If query and key shapes don't match
    
    References:
        - Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS.
          https://arxiv.org/abs/1706.03762
    """
```

## Pull Request Process

1. **Title**: Use conventional format (same as commits)
2. **Description**: Explain what and why
3. **Link issues**: "Fixes #123" or "Relates to #456"
4. **Tests**: Ensure all tests pass
5. **Review**: Address reviewer comments

### PR Template

```markdown
## Description
Brief explanation of changes

## Type of Change
- [ ] New feature
- [ ] Bug fix  
- [ ] Documentation
- [ ] Other

## Testing
- [ ] Added unit tests
- [ ] All tests pass (`make test`)
- [ ] Manual testing: [describe]

## Checklist
- [ ] Follows code style guidelines
- [ ] Updated documentation
- [ ] No breaking changes
```

## Non-Code Contributions

- **Lessons**: Write new Module lessons
- **Exercises**: Create coding challenges
- **Datasets**: Build Banso corpus, fix data quality
- **Translation**: Translate lessons to other languages
- **Examples**: Create notebooks demonstrating use cases
- **Feedback**: Test curriculum, report issues

## Community Note

We follow the [Contributor Covenant](https://www.contributor-covenant.org/):
- Be respectful and inclusive
- Welcome diverse perspectives
- Address conflicts constructively
- Report harassment to maintainers

## License

By contributing, you agree your work is licensed under MIT License.
