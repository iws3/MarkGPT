# Contribution Best Practices

This document outlines best practices for contributing to the MarkGPT LLM Curriculum. These guidelines help maintain quality, consistency, and clarity across the entire project.

---

## Table of Contents

1. [Code Quality](#code-quality)
2. [Documentation Standards](#documentation-standards)
3. [Jupyter Notebook Guidelines](#jupyter-notebook-guidelines)
4. [Commit Message Standards](#commit-message-standards)
5. [Testing & Validation](#testing--validation)
6. [Pull Request Standards](#pull-request-standards)
7. [Educational Content Guidelines](#educational-content-guidelines)
8. [Specific Module Guidelines](#specific-module-guidelines)

---

## Code Quality

### Python Code Style

Follow **PEP 8** with these specific guidelines:

```python
# ✅ Good: Clear variable names, proper spacing
def calculate_attention_scores(query, key, dimension):
    """Compute scaled dot-product attention scores."""
    scores = np.dot(query, key.T) / np.sqrt(dimension)
    return scores

# ❌ Avoid: Cryptic names, bad formatting
def calc(q, k, d):
    s = np.dot(q, k.T) / np.sqrt(d)
    return s
```

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Variables | `snake_case` | `input_data`, `model_params` |
| Functions | `snake_case` | `compute_loss()`, `train_epoch()` |
| Classes | `PascalCase` | `TransformerBlock`, `DataLoader` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_SEQUENCE_LENGTH` |
| Notebooks | Descriptive | `numpy_arrays_basics.ipynb` |

### Comments & Readability

```python
# ✅ Good: Comments explain WHY, not WHAT
def softmax(x):
    # Subtract max for numerical stability (prevents overflow)
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# ❌ Avoid: Comments that restate the code
def softmax(x):
    # Subtract max from x
    x = x - np.max(x, axis=-1, keepdims=True)
    # Compute softmax
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
```

### No Hardcoded Paths

```python
# ✅ Good: Relative paths
data_path = Path(__file__).parent / "data" / "dataset.csv"
df = pd.read_csv(data_path)

# ❌ Avoid: Absolute paths
df = pd.read_csv("C:/Users/YourName/data.csv")
df = pd.read_csv("/Users/yourname/data.csv")
```

---

## Documentation Standards

### README Files

Every module should have a clear README that includes:

1. **Module Overview** — What will you learn?
2. **Prerequisites** — What should you know first?
3. **Learning Outcomes** — What can you do after?
4. **Structure** — What's in each subfolder?
5. **Estimated Time** — How long does it take?

```markdown
# Module 05: NLP Foundations — Text as Data

## Overview
This module introduces how to represent text as numerical data that computers can process.

## Prerequisites
- Python basics (Module 02)
- NumPy operations (Module 02)

## Learning Outcomes
After this module, you will:
- [ ] Understand tokenization strategies
- [ ] Create and visualize word embeddings
- [ ] Build text preprocessing pipelines

## Estimated Time: 6-8 hours
```

### Docstrings

Use **Google-style docstrings** for all functions:

```python
def create_embeddings(text, embedding_dim=300):
    """
    Create word embeddings from text using Word2Vec.
    
    Args:
        text (str): Input text to embed
        embedding_dim (int): Dimension of embeddings (default: 300)
    
    Returns:
        np.ndarray: Array of shape (vocab_size, embedding_dim)
    
    Raises:
        ValueError: If embedding_dim < 10
    
    Example:
        >>> text = "hello world"
        >>> embeddings = create_embeddings(text, embedding_dim=100)
        >>> embeddings.shape
        (2, 100)
    """
    if embedding_dim < 10:
        raise ValueError("embedding_dim must be at least 10")
    # Implementation...
```

---

## Jupyter Notebook Guidelines

### Cell Structure

Follow this pattern for educational notebooks:

```
[1] Markdown: Title & Overview
[2] Markdown: Learning Objectives
[3] Markdown: Prerequisite Review
[4] Code: Import Libraries
[5] Code: Load/Generate Data
[6] Markdown: Explanation & Theory
[7] Code: Implementation
[8] Markdown: Interpretation & Visualization
[9] Code: Exercises
[10] Markdown: Summary & Next Steps
```

### Code Cell Best Practices

```python
# ✅ Good: Clear, well-commented code
# Load data
df = pd.read_csv('data.csv')

# Remove missing values (we'll handle them properly in Module 09)
df_clean = df.dropna()

# Create feature: word length
df_clean['word_length'] = df_clean['text'].str.len()

print(f"Dataset shape: {df_clean.shape}")
print(f"Missing values: {df_clean.isnull().sum()}")

# ❌ Avoid: Long, confusing cells
df = pd.read_csv('data.csv'); df_clean = df.dropna(); df_clean['wl'] = df_clean['text'].str.len(); print(df_clean.shape)
```

### Output & Visualization

```python
# ✅ Good: Informative plots with labels
plt.figure(figsize=(10, 6))
plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Word Embeddings Visualization (First 2 Dimensions)')
plt.legend()
plt.show()

# ❌ Avoid: Plots without context
plt.scatter(embeddings[:, 0], embeddings[:, 1])
plt.show()
```

### Markdown Cells

Use Markdown cells to:
- Explain concepts before code
- Break up code logically
- Provide mathematical notation using LaTeX
- Link to external resources

```markdown
## Attention Mechanism

The attention mechanism computes a weighted sum of values based on their similarity to a query:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ is the query matrix
- $K$ is the key matrix
- $V$ is the value matrix
- $d_k$ is the key dimension

Let's implement this step by step:
```

---

## Commit Message Standards

Follow **Conventional Commits** format for clear, organized history:

```bash
# Format: type(scope): subject

# Examples:
git commit -m "feat(module-05): add word embedding visualization"
git commit -m "fix(module-02): correct numpy array indexing explanation"
git commit -m "docs(contributors): expand troubleshooting guide"
git commit -m "refactor(module-06): improve transformer implementation clarity"
git commit -m "test(module-03): add unit tests for backpropagation"
```

### Types

- `feat:` — New feature or lesson content
- `fix:` — Bug fix or error correction
- `docs:` — Documentation improvements
- `refactor:` — Code reorganization without behavior changes
- `test:` — Tests or test-related changes
- `style:` — Formatting (spaces, quotes, etc.)
- `perf:` — Performance improvements

### Scope

- `module-01` — Module 01 related changes
- `module-02` — Module 02 related changes
- `contributors` — Contributor guide changes
- `docs` — General documentation
- `build` — Build system changes

### Subject Line

- Use imperative mood: "add" not "added"
- Don't capitalize the first letter
- No period at the end
- Keep it under 50 characters when possible

```bash
# ✅ Good
git commit -m "feat(module-02): add 5 practice problems for pandas operations"

# ❌ Avoid
git commit -m "Fixed module 2 stuff and added some exercises for students"
```

---

## Testing & Validation

### Before Submitting a Notebook

- [ ] Run all cells in order: `Restart & Run All`
- [ ] Check for any error messages
- [ ] Verify all output is visible
- [ ] Test with fresh data (ensure no side effects)
- [ ] Check imports are at the top
- [ ] Verify file paths are relative (not absolute)

### Before Submitting Code

```bash
# If you've added Python code, run basic checks:
python -m py_compile your_file.py  # Check syntax
python -m pylint your_file.py      # Check style (optional)
python your_file.py                # Run it
```

### Performance Expectations

For notebooks:
- Code cells should run in < 30 seconds
- Visualizations should render in < 5 seconds
- Total notebook should complete in < 10 minutes

For large computations:
- Document expected runtime
- Provide progress indicators
- Consider using smaller datasets for examples

---

## Pull Request Standards

### PR Title Format

```
type: brief description

# Examples:
feat: add attention visualization for module-06
fix: correct softmax implementation in module-04
docs: expand contribution guide with examples
```

### PR Description Template

```markdown
## Description
Clear explanation of what you changed and why.

## Type of Change
- [ ] New lesson or exercise
- [ ] Bug fix
- [ ] Documentation improvement
- [ ] Code refactor
- [ ] Other: ___

## Related Module(s)
- Module 02
- Module 05

## Testing
How did you test this? What should reviewers test?
- Ran all notebook cells successfully
- Verified with 3 different inputs
- Tested on Python 3.10

## Checklist
- [ ] Code works without errors
- [ ] Comments and docstrings added
- [ ] No hardcoded paths
- [ ] Follows naming conventions
- [ ] Commit messages are clear

## Screenshots (if applicable)
Include before/after visualizations if relevant.
```

---

## Educational Content Guidelines

### Explaining Complex Concepts

Use the **Feynman Technique**:

1. **Explain Simply** — Use plain language, avoid jargon
2. **Identify Gaps** — Ask yourself what you don't understand
3. **Simplify Further** — Refine your explanation
4. **Use Analogies** — Compare to familiar concepts

**Example: Explaining Backpropagation**

```markdown
## Backpropagation: A Simple Explanation

Imagine you're navigating a dark mountain trail trying to reach the bottom.
You can't see the path, but you feel the slope under your feet.

- **Forward Pass**: You walk forward and reach a point with elevation E (error)
- **Backward Pass**: You turn around, measure the slope, find the steepest descent
- **Update**: You take a step downhill (gradient descent)
- **Repeat**: You keep doing this until you reach the valley (low error)

In a neural network:
- "Elevation" = Loss (how wrong the prediction is)
- "Slope" = Gradient (how much each weight contributed to the error)
- "Steps downhill" = Weight updates

The math just formalizes this intuition!
```

### Progression & Scaffolding

Build complexity gradually:

```
Lesson 1: Simple example with 2 features
Lesson 2: Slightly larger example (10 features)
Lesson 3: Real dataset (100+ features)
Exercise 1: Modify existing code
Exercise 2: Write code from scratch
Project: Combine multiple concepts
```

### Real-World Applications

For every concept, include an application:

```python
# Concept: Softmax

# Basic example (education)
scores = np.array([1, 2, 3])
probs = softmax(scores)

# Real application: Language model prediction
logits = model(input_text)  # Raw model output
probabilities = softmax(logits)  # Convert to probabilities
next_word = sample_from(probabilities)  # Generate next word
```

---

## Specific Module Guidelines

### Modules 01-03: Foundations

- Use lots of **visualizations**
- Explain **intuition before equations**
- Include pseudocode and diagrams
- Assume minimal background knowledge

### Modules 04-06: Intermediate

- Balance **theory and practice**
- Provide mathematical derivations
- Include implementation details
- Reference papers and resources

### Modules 07-10: Advanced

- Assume stronger background
- Focus on **cutting-edge techniques**
- Include research-level explanations
- Discuss trade-offs and design decisions

### Banso Language Content

- Include **linguistic notes** about Banso grammar/phonetics
- Provide **context for cultural references**
- Use **proper orthography** and pronunciation guides
- **Cite sources** for language materials

---

## Quick Checklist for Contributors

Before submitting, ask yourself:

- [ ] Is the explanation clear to someone encountering this for the first time?
- [ ] Would a complete beginner understand the code?
- [ ] Are there comments explaining *why*, not just *what*?
- [ ] Do all examples run without errors?
- [ ] Are hardcoded paths replaced with relative paths?
- [ ] Is the commit message clear and follows conventions?
- [ ] Did I test everything thoroughly?
- [ ] Is the contribution focused (or should I split it)?
- [ ] Does it fit the style and structure of the existing content?
- [ ] Would I be proud to show this to a professor or hiring manager?

---

## Questions or Clarifications?

See [contributors/CONTRIBUTORS_GUIDE.md](../contributors/CONTRIBUTORS_GUIDE.md) or email `iwstechnical@gmail.com`.

Happy contributing! 🚀
