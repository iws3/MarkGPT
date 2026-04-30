# Module 2 Exercise: NumPy Vectorization

## Objective
Master NumPy array operations and vectorization to write efficient numerical code.

## Background
The curse of Python is loops—they're slow. NumPy's power is **vectorization**: writing operations that apply to entire arrays at once, executed in optimized C code.

## Exercise 1: Matrix Operations

```python
import numpy as np
import time

# TODO: Create a (1000, 500) matrix of random values
X = None  # Replace with np.random.randn(...)

# TODO: Create a (500, 100) weight matrix
W = None  # Replace with np.random.randn(...)

# TODO: Compute matrix multiplication X @ W using NumPy (no loops!)
Z = None

# Verify shape
assert Z.shape == (1000, 100), f"Expected (1000, 100), got {Z.shape}"
```

## Exercise 2: Broadcasting Subtlety

```python
# Data normalization using broadcasting
X = np.random.randn(100, 784)  # 100 images, 784 pixels

# TODO: Compute mean of each column
mean = None  # Hint: np.mean(X, axis=0)

# TODO: Compute standard deviation of each column  
std = None

# TODO: Normalize: (X - mean) / std using broadcasting (no explicit loops)
X_normalized = None

# Verify: Each column should have mean ≈ 0, std ≈ 1
assert np.allclose(X_normalized.mean(axis=0), 0, atol=1e-6)
assert np.allclose(X_normalized.std(axis=0), 1, atol=1e-6)
```

## Exercise 3: Masking and Indexing

```python
# Filtering data with boolean masks
texts = np.array(['the', 'bible', 'speaks', 'truth', 'god', 'light'])
lengths = np.array([3, 5, 6, 5, 3, 5])

# TODO: Create a boolean mask for words with length >= 5
mask = None

# TODO: Use mask to select only words with length >= 5
long_words = None

# Verify
assert len(long_words) == 4
assert 'the' not in long_words
```

## Exercise 4: Aggregation

```python
# Compute statistics on batch of embeddings
embeddings = np.random.randn(1000, 100)  # 1000 samples, 100-dim embeddings

# TODO: Compute mean embedding (average across all samples)
mean_embedding = None

# TODO: Compute covariance matrix (100, 100) capturing correlations
# Hint: np.cov(embeddings.T)
covariance = None

# TODO: Find most similar pair of embeddings using cosine distance
# Hint: Compute correlations between all pairs
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# (This is harder; optional bonus)
```

## Challenge: Efficient Batch Processing

```python
# Problem: Process 100,000 Bible verses through a model
verses = [text1, text2, ..., text100000]  # Strings

# Naive way (SLOW):
# for verse in verses:
#     embedding = model(verse)  # One at a time!

# Better way (FAST):
def batch_process(verses, model, batch_size=100):
    """Process verses in batches using vectorized operations"""
    embeddings = []
    
    # TODO: Iterate over batches
    for i in range(0, len(verses), batch_size):
        batch = verses[i:i+batch_size]
        
        # TODO: Tokenize entire batch
        tokens = tokenize_batch(batch)
        
        # TODO: Forward through model (operates on entire batch at once)
        batch_embeddings = model(tokens)
        
        embeddings.append(batch_embeddings)
    
    # TODO: Concatenate all embeddings
    return np.concatenate(embeddings)
```

## Key Takeaways

- ✅ Use NumPy operations instead of Python loops
- ✅ Broadcasting lets you avoid explicit loops
- ✅ Batch processing is faster than one-at-a-time
- ✅ Vectorization makes code 100-1000x faster

## Submission

```python
# Test file (submit this)
def test_vectorization():
    # Your solutions should make these pass:
    assert Z.shape == (1000, 100)
    assert np.allclose(X_normalized.mean(axis=0), 0, atol=1e-6)
    assert len(long_words) == 4
    # etc.
```

## Resources

- NumPy Documentation: https://numpy.org/doc/stable/user/basics.broadcasting.html
- "A Gentle Introduction to NumPy": https://github.com/rougier/numpy-100
