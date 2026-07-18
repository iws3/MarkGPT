# Contribution Templates

## Issue Report Template

```markdown
## Description
Brief description of the issue.

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.0]
- PyTorch: [e.g., 2.0.1]
- CUDA: [e.g., 11.8]

## Steps to Reproduce
1. ...
2. ...
3. ...

## Expected Behavior
What should happen.

## Actual Behavior
What actually happened.

## Error Message
```
Paste full error/traceback here
```

## Additional Context
Any other relevant information.
```

## Pull Request Template

```markdown
## Description
Describe your changes. Link to related issues: Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code style/refactoring

## Testing
- [ ] Added unit tests
- [ ] Existing tests pass locally
- [ ] Tested on GPU (if applicable)

## Documentation
- [ ] Updated docstrings
- [ ] Updated README if needed
- [ ] Added example code

## Checklist
- [ ] Code follows project style (black, ruff)
- [ ] Type hints included
- [ ] No breaking changes (or documented)
- [ ] Commit messages are clear

## Related Issues
Closes: #...
Relates to: #...
```

## Code Review Checklist

**For Reviewers**:

- [ ] Code is clear and well-commented
- [ ] Type hints present and correct
- [ ] Docstrings complete (Google style)
- [ ] No unused imports or variables
- [ ] Tests cover new functionality
- [ ] Performance impact acceptable
- [ ] Documentation updated
- [ ] No security issues
- [ ] Follows project conventions

**For Authors** (before submitting):

```bash
# Run linting
black src/
ruff check src/ --fix

# Run type checking  
mypy src/ --strict

# Run tests
pytest tests/ -v

# Manual review
git diff --check
```

## Documentation Style Guide

### Example Lesson

```markdown
# L05.1 Embeddings

## Learning Objectives
- [ ] Understand word embeddings as learned representations
- [ ] Compare Word2Vec, GloVe, and transformer embeddings
- [ ] Implement embedding lookup and training

## Concepts

### Word Embeddings
Brief explanation with intuition.

### Skip-gram Model
Detailed explanation with math.

## Code Examples

\`\`\`python
# Practical example here
\`\`\`

## Exercises
1. Exercise 1 with scaffold
2. Exercise 2 with challenge

## Key Takeaways
- Summary point 1
- Summary point 2

## References
- Mikolov et al. (2013): Word2Vec - https://arxiv.org/abs/1301.3781
```

### Example Exercise

```markdown
# day05_embeddings.md

## Objective
Learn to implement and visualize word embeddings.

## Part 1: Basics (30 min)

### Task 1.1: Load Pre-trained Embeddings
\`\`\`python
# TODO: Load GloVe embeddings from data/glove.100d.txt
# Expected output: embedding matrix (vocab_size, 100)
\`\`\`

## Part 2: Challenge (60 min)

### Challenge: Train Your Own Embeddings
Implement skip-gram training...

## Solution Hints
- Hint 1
- Hint 2

## Evaluation Rubric
- Code correctness: /10
- Visualization quality: /10
```

## Commit Message Style

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code formatting
- `refactor`: Code restructuring
- `perf`: Performance
- `test`: Tests
- `chore`: Tooling

### Examples

**Good**:
```
feat(training): add gradient accumulation

- Implement GradientAccumulation context manager
- Support arbitrary accumulation steps
- Add tests for correctness

Closes #42
```

**Bad**:
```
fixed stuff

updated code to make it better
```

---

**Templates Version**: 1.0
**Last Updated**: 2024
