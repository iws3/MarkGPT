# Module 02 — Python & Mathematics Essentials
## Days 7–12 | Beginner–Intermediate

---

## Module Overview

Now that you understand what language models are, let's build the mathematical and programming foundations you'll need. This module doesn't assume you know Python or linear algebra — only curiosity and willingness to work through examples.

By the end of Module 02, you will:
- Write vectorized code in NumPy
- Manipulate data with Pandas
- Plot and visualize loss curves
- Understand matrix operations, calculus, and probability
- Have a solid foundation for the neural networks in Module 03
- Relate theory to MarkGPT architecture
- Complete hands-on exercises

## Structure

```
lessons/       - Conceptual explanations with code examples
exercises/     - Practical implementation exercises
projects/      - Larger projects (optional)
resources/     - Additional readings and links
```

## Time Estimate

- Lessons: 4-6 hours
- Exercises: 4-6 hours
- **Total: 8-12 hours per module**

## Key Concepts

[See lesson files for detailed content]

## Completion Checklist

- [ ] Read all lessons (L*_*.md files)
- [ ] Complete all exercises (day*_*.md files)
- [ ] Pass the module quiz (if provided)
- [ ] Understand connections to MarkGPT

## Resources

- Lesson references contain links to papers and tutorials
- http://markgpt-docs.com (forthcoming)
- GitHub discussions: https://github.com/yourusername/MarkGPT-LLM-Curriculum/discussions

## Next Module

See ../module-0$((i+1))/README.md for the next module.
## Python Essentials for Machine Learning

### Python Fundamentals Review
Python is the dominant language in ML due to its simplicity, readability, and ecosystem.

**Core Concepts**
- Variables: Store data in memory
- Data types: int, float, str, bool, list, dict, tuple, set
- Operators: Arithmetic, comparison, logical, membership
- Control flow: if/elif/else, loops (for, while)
- Functions: Reusable code blocks with parameters and returns
- Classes: Object-oriented programming foundation

### Data Structures

**Lists**
- Ordered, mutable collections
- Index: 0-based access
- Methods: append(), extend(), insert(), remove(), pop()
- List comprehension: Concise creation [x*2 for x in range(10)]

**Dictionaries**
- Key-value pairs, unordered (Python 3.7+ ordered)
- Efficient lookup by key
- Methods: keys(), values(), items(), get(), pop()
- Use case: Storing mapped data

**Tuples**
- Ordered, immutable sequences
- Hashable: Can be dict keys
- Unpacking: a, b, c = (1, 2, 3)
- Use case: Fixed collections, function returns

**Sets**
- Unordered, unique elements
- Operations: union, intersection, difference
- Use case: Removing duplicates, membership testing

### Functions and Scope

**Function Definition**
```python
def function_name(arg1, arg2=default_value, *args, **kwargs):
    '''Docstring explaining function'''
    # Function body
    return result
```

**Scope Rules (LEGB)**
- Local: Inside function
- Enclosing: In outer function
- Global: Module level
- Built-in: Python built-ins

**Lambda Functions**
- Anonymous functions: lambda x: x**2
- Use with map(), filter(), sorted()
- Avoid complex logic

### Error Handling

**Try-Except Pattern**
```python
try:
    # Code that might raise exception
except SpecificError as e:
    # Handle specific error
except Exception as e:
    # Catch all remaining
finally:
    # Always execute (cleanup)
```

**Common Exceptions**
- ValueError: Invalid value
- TypeError: Wrong type
- IndexError: Invalid index
- KeyError: Missing dictionary key
- ZeroDivisionError: Division by zero

### Modules and Packages

**Importing**
- import numpy: Full module namespace
- from numpy import array: Specific items
- from numpy import * : All items (avoid, causes conflicts)
- import numpy as np: Aliasing

**Creating Modules**
- File with .py extension is module
- Packages: Folders with __init__.py
- Relative imports: from . import sibling
- Absolute imports: from package.module import item

## NumPy: Numerical Computing

### Why NumPy?
- Speed: ~100x faster than Python lists
- Memory: Efficient data storage
- Broadcasting: Vectorized operations
- Integration: Foundation for pandas, scikit-learn, etc.

### Creating Arrays

```python
import numpy as np

# From Python lists
np.array([1, 2, 3])
np.array([[1, 2], [3, 4]])  # 2D array

# Special arrays
np.zeros((3, 4))  # 3x4 zeros
np.ones((2, 3))   # 2x3 ones
np.eye(3)         # 3x3 identity
np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5) # 5 points 0 to 1
np.random.randn(3, 4) # Normal distribution
```

### Array Operations

**Element-wise Operations**
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b  # [5, 7, 9]
a * b  # [4, 10, 18]
a ** 2 # [1, 4, 9]
np.sqrt(a)  # Square root
np.exp(a)   # Exponential
np.log(a)   # Natural log
```

**Matrix Operations**
- @ operator: Matrix multiplication
- np.dot(a, b): Inner product
- np.outer(a, b): Outer product
- np.transpose(a) or a.T: Transpose

### Indexing and Slicing

**Single Index**
```python
a = np.array([0, 1, 2, 3, 4, 5])
a[0]   # First element: 0
a[-1]  # Last element: 5
a[2:5] # Slice [2,3,4]
```

**Multi-dimensional**
```python
b = np.array([[1, 2, 3], [4, 5, 6]])
b[0, 1]    # Row 0, column 1: 2
b[0, :]    # First row: [1, 2, 3]
b[:, 1]    # Second column: [2, 5]
b[0:1, 1:2] # Subarray
```

**Boolean Indexing**
```python
mask = a > 2
a[mask]  # Elements > 2
```

### Broadcasting

**Why Broadcasting?**
Vectorized operations on arrays of different shapes

**Broadcasting Rules**
1. Dimensions align from right to left
2. Dimensions are compatible if equal or one is 1
3. Missing dimensions treated as 1

**Examples**
```python
a = np.array([[1, 2, 3]])  # Shape (1, 3)
b = np.array([[1], [2], [3]])  # Shape (3, 1)
a + b  # Shape (3, 3), broadcasts both

c = np.array([1, 2, 3])  # Shape (3,)
d = np.array([[1], [2]])  # Shape (2, 1)
c + d  # Shape (2, 3)
```

### Common NumPy Functions

**Aggregation**
- np.sum(): Sum all elements
- np.mean(): Average
- np.std(): Standard deviation
- np.min(), np.max(): Minimum, maximum
- np.argmin(), np.argmax(): Index of min/max

**Linear Algebra**
- np.linalg.inv(): Matrix inverse
- np.linalg.det(): Determinant
- np.linalg.eig(): Eigenvalues, eigenvectors
- np.linalg.solve(): System of equations
- np.linalg.norm(): Vector/matrix norm

**Random**
- np.random.rand(): Uniform [0, 1)
- np.random.randn(): Standard normal
- np.random.choice(): Sample from array
- np.random.shuffle(): In-place shuffle

### Performance Tips

**Avoid Loops**
Replace Python loops with vectorized NumPy code
```python
# Slow
result = []
for x in array:
    result.append(x ** 2)

# Fast
result = array ** 2
```

**Memory Efficiency**
- Use .astype() for appropriate dtypes
- Avoid unnecessary copies
- Use views when possible (slicing)

**Benchmarking**
```python
import timeit
timeit.timeit('x ** 2', 'x = np.array(range(1000))')
```

## Pandas: Data Manipulation

### Why Pandas?
- Tabular data: Rows and columns like Excel
- Missing data: Handles NaN gracefully
- Data alignment: Automatic alignment by index
- Flexibility: Mix numeric, string, categorical data

### Creating DataFrames

```python
import pandas as pd

# From dict
df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'],
                  'age': [25, 30, 35]})

# From lists
df = pd.DataFrame([['Alice', 25], ['Bob', 30]],
                 columns=['name', 'age'])

# From CSV
df = pd.read_csv('data.csv')
```

### Accessing Data

**Column Access**
```python
df['name']      # Series
df[['name', 'age']]  # DataFrame
df.name         # Attribute access (if no spaces)
```

**Row Access**
```python
df.loc[0]       # By label
df.iloc[0]      # By position
df.loc[0, 'name']  # Specific cell
```

**Conditional Selection**
```python
df[df['age'] > 25]
df[(df['age'] > 25) & (df['name'] == 'Bob')]
```

### Data Cleaning

**Missing Values**
```python
df.isnull()      # Check for NaN
df.dropna()      # Remove rows with NaN
df.fillna(0)     # Replace with value
df.fillna(method='ffill')  # Forward fill
```

**Duplicates**
```python
df.duplicated()  # Find duplicates
df.drop_duplicates()  # Remove duplicates
```

**Data Types**
```python
df.dtypes        # Check types
df.astype({'age': 'int32'})  # Convert
```

### Data Transformation

**Sorting**
```python
df.sort_values('age')  # Ascending
df.sort_values('age', ascending=False)  # Descending
```

**Grouping**
```python
df.groupby('department').sum()
df.groupby('department')['salary'].mean()
```

**Aggregation**
```python
df.agg({'age': 'mean', 'salary': 'sum'})
df.describe()  # Summary statistics
```

**Merging**
```python
pd.merge(df1, df2, on='key')  # Inner join
pd.merge(df1, df2, how='left', on='key')
```

## Linear Algebra Fundamentals

### Vectors and Matrices

**Vectors**
- 1D array of numbers
- Direction and magnitude
- Example: [1, 2, 3]
- Notation: **v** or v⃗

**Matrices**
- 2D array: m rows × n columns
- Example: [[1, 2], [3, 4]]
- Notation: **A** or A_ij

**Tensors**
- Generalization to n dimensions
- Images: 3D (height, width, channels)
- Batches: 4D (batch, height, width, channels)

### Vector Operations

**Magnitude (Norm)**
$$||v|| = \sqrt{v_1^2 + v_2^2 + ... + v_n^2}$$

**Dot Product (Inner Product)**
$$v \cdot w = v_1 w_1 + v_2 w_2 + ... + v_n w_n$$

**Geometric Interpretation**
$$v \cdot w = ||v|| ||w|| \cos(\theta)$$
- θ = 0: Parallel vectors
- θ = 90°: Orthogonal (perpendicular)
- θ = 180°: Opposite vectors

### Matrix Operations

**Addition and Subtraction**
- Element-wise: A + B
- Same shape required

**Scalar Multiplication**
- Each element multiplied by scalar
- c * A = [c*a_ij]

**Matrix Multiplication**
$$C = AB \text{ where } c_{ij} = \sum_k a_{ik} b_{kj}$$
- Not commutative: AB ≠ BA
- Associative: (AB)C = A(BC)
- A: shape (m, n), B: shape (n, p) → C: shape (m, p)

**Transpose**
$$A^T_{ij} = A_{ji}$$
- Switch rows and columns

### Matrix Decomposition

**Eigendecomposition**
$$Av = \lambda v$$
- v: Eigenvector
- λ: Eigenvalue
- A must be square

**Singular Value Decomposition (SVD)**
$$A = U \Sigma V^T$$
- U, V: Orthogonal matrices
- Σ: Diagonal matrix of singular values
- Works for any matrix

**QR Decomposition**
$$A = QR$$
- Q: Orthogonal matrix
- R: Upper triangular
- Used in least squares

### Solving Linear Systems

**System of Equations**
$$Ax = b$$
- A: Coefficient matrix
- x: Unknown variables
- b: Constants

**Solution Methods**
1. Matrix inverse: x = A^(-1)b (if invertible)
2. Gaussian elimination: Direct computation
3. Iterative methods: Gradient descent, conjugate gradient
4. LU decomposition: Factorize A

**Computational Considerations**
- Computational cost: O(n³) for direct methods
- Numerical stability: Avoid near-singular matrices
- Sparse systems: Use specialized algorithms

## Calculus for Machine Learning

### Derivatives and Gradients

**Derivative**
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$
- Rate of change
- Slope of tangent line

**Gradient (Multivariable)**
$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \end{bmatrix}$$
- Vector of partial derivatives
- Points in direction of steepest increase

### Chain Rule

**Single Variable**
$$\frac{df}{dx} = \frac{df}{du} \cdot \frac{du}{dx}$$

**Multivariable**
$$\frac{\partial f}{\partial x} = \sum_i \frac{\partial f}{\partial u_i} \frac{\partial u_i}{\partial x}$$

**Application: Backpropagation**
- Propagate gradients backward through network
- Compute ∂L/∂w for all weights
- Foundation of neural network training

### Partial Derivatives

**Definition**
- Derivative with respect to one variable
- Hold others constant

**Second Derivatives**
$$\frac{\partial^2 f}{\partial x^2}, \quad \frac{\partial^2 f}{\partial x \partial y}$$

**Jacobian and Hessian**
- Jacobian: Matrix of first derivatives
- Hessian: Matrix of second derivatives
- Used in optimization algorithms

### Optimization

**Finding Extrema**
$$\nabla f = 0 \text{ at critical points}$$

**Convexity**
- Convex function: Single global minimum
- Non-convex: Multiple local minima
- Hessian positive semi-definite → Convex

**Gradient Descent**
$$x_{n+1} = x_n - \alpha \nabla f(x_n)$$
- Iteratively move in direction of negative gradient
- Step size α: Learning rate
- Converges for convex, well-behaved functions

## Probability and Statistics

### Basic Probability

**Definitions**
- Probability P(A): Likelihood of event A
- Sample space: All possible outcomes
- Event: Subset of sample space

**Rules**
- Sum rule: P(A) = 1 - P(not A)
- Product rule: P(A and B) = P(A|B)P(B)
- Bayes theorem: $$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

### Probability Distributions

**Discrete Distributions**
- Bernoulli: Binary (success/failure)
  $$P(X=1) = p, \quad P(X=0) = 1-p$$
- Binomial: n independent Bernoulli trials
- Poisson: Events in fixed interval

**Continuous Distributions**
- Uniform: Constant probability over interval
- Gaussian (Normal): Bell curve
  $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
- Exponential: Waiting times

### Expectation and Variance

**Expected Value**
$$E[X] = \sum_i x_i P(x_i)$$ (discrete)
$$E[X] = \int x f(x) dx$$ (continuous)

**Variance**
$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$
- Measures spread of distribution
- Standard deviation: σ = √Var(X)

**Covariance**
$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])]$$
- Measures relationship between variables
- Correlation: Normalized covariance

### Statistical Inference

**Point Estimation**
- Mean: μ̂ = (1/n)Σx_i
- Variance: σ̂² = (1/n)Σ(x_i - μ̂)²
- Maximum Likelihood Estimation (MLE)

**Confidence Intervals**
- Range of plausible parameter values
- 95% CI: ±1.96σ for normal distribution

**Hypothesis Testing**
- Null hypothesis H₀: No effect
- Alternative H₁: Effect exists
- p-value: Probability of data given H₀

### Central Limit Theorem

**Statement**
Distribution of sample means approaches normal as n→∞
regardless of original distribution

**Implications**
- Normal approximation valid for large samples
- Foundation of many statistical tests
- z-scores and t-tests rely on this

**Practical Significance**
- Sample size ~30: Often sufficient
- Enables inference from samples
- Justifies assuming normality

## Connecting to MarkGPT

### How These Fundamentals Power LLMs

**Python & Numpy**
- Matrix operations: Token embeddings multiplied by weight matrices
- Broadcasting: Batch operations across multiple sequences
- Vectorization: Efficient GPU computation

**Linear Algebra**
- Attention mechanism: Q·K^T matrix multiplication
- Transformations: Embedding rotations (RoPE)
- Decompositions: Low-rank approximations for efficiency

**Calculus**
- Gradients: Backpropagation through layers
- Chain rule: Error signals through attention
- Optimization: ADAM updates parameters

**Probability**
- Softmax: Convert logits to probabilities
- Cross-entropy: Loss function for training
- Beam search: Probabilistic sequence decoding

## Common Pitfalls and Best Practices

### Python Pitfalls

**Pitfall 1: Mutable Default Arguments**
```python
# Bad
def append_to_list(elem, to=[]):
    to.append(elem)
    return to

# Good
def append_to_list(elem, to=None):
    if to is None:
        to = []
    to.append(elem)
    return to
```

**Pitfall 2: Integer Division**
```python
# Python 2: 3 / 2 = 1 (integer division)
# Python 3: 3 / 2 = 1.5 (float division)
# Always: 3 // 2 = 1 (integer division)
```

**Pitfall 3: Name Shadowing**
```python
# Avoid
sum = [1, 2, 3]  # Shadows built-in sum()
total = sum(sum)  # Error!

# Good
data = [1, 2, 3]
total = sum(data)
```

### NumPy Pitfalls

**Pitfall 1: Unintended Broadcasting**
```python
# Unexpected shape change
a = np.array([1, 2, 3])  # Shape (3,)
b = np.array([[1], [2]])  # Shape (2, 1)
c = a + b  # Shape (2, 3) - broadcasts both!
```

**Pitfall 2: View vs Copy**
```python
a = np.array([1, 2, 3, 4])
b = a[1:3]  # View, changes affect a
c = a[1:3].copy()  # Copy, changes don't affect a
```

**Pitfall 3: Data Type Mismatches**
```python
a = np.array([1, 2, 3])  # dtype int64
b = np.array([1.5, 2.5])  # dtype float64
c = a / b  # Result is float64

# Integer division unexpected
d = a / 2  # float division, results float64
```

### Pandas Pitfalls

**Pitfall 1: Chained Indexing**
```python
# Can trigger SettingWithCopyWarning
df[df['age'] > 25]['salary'] = 100000  # Don't do this

# Better
df.loc[df['age'] > 25, 'salary'] = 100000
```

**Pitfall 2: Index Alignment**
```python
ordering = pd.Series([3, 1, 2])
avg_price = pd.Series([100, 101, 102], index=[1, 2, 3])
ordering * avg_price  # Aligns by index!
```

**Pitfall 3: Modifying a View**
```python
df2 = df[df['age'] > 25]  # View or copy?
df2['salary'] = 50000  # Modifies df? Sometimes.

df2 = df[df['age'] > 25].copy()  # Force copy
```

## Python Essentials for Machine Learning

### Python Fundamentals Review
Python is the dominant language in ML due to its simplicity, readability, and ecosystem.

**Core Concepts**
- Variables: Store data in memory
- Data types: int, float, str, bool, list, dict, tuple, set
- Operators: Arithmetic, comparison, logical, membership
- Control flow: if/elif/else, loops (for, while)
- Functions: Reusable code blocks with parameters and returns
- Classes: Object-oriented programming foundation

### Data Structures

**Lists**
- Ordered, mutable collections
- Index: 0-based access
- Methods: append(), extend(), insert(), remove(), pop()
- List comprehension: Concise creation [x*2 for x in range(10)]

**Dictionaries**
- Key-value pairs, unordered (Python 3.7+ ordered)
- Efficient lookup by key
- Methods: keys(), values(), items(), get(), pop()
- Use case: Storing mapped data

**Tuples**
- Ordered, immutable sequences
- Hashable: Can be dict keys
- Unpacking: a, b, c = (1, 2, 3)
- Use case: Fixed collections, function returns

**Sets**
- Unordered, unique elements
- Operations: union, intersection, difference
- Use case: Removing duplicates, membership testing

### Functions and Scope

**Function Definition**
```python
def function_name(arg1, arg2=default_value, *args, **kwargs):
    '''Docstring explaining function'''
    # Function body
    return result
```

**Scope Rules (LEGB)**
- Local: Inside function
- Enclosing: In outer function
- Global: Module level
- Built-in: Python built-ins

**Lambda Functions**
- Anonymous functions: lambda x: x**2
- Use with map(), filter(), sorted()
- Avoid complex logic

### Error Handling

**Try-Except Pattern**
```python
try:
    # Code that might raise exception
except SpecificError as e:
    # Handle specific error
except Exception as e:
    # Catch all remaining
finally:
    # Always execute (cleanup)
```

**Common Exceptions**
- ValueError: Invalid value
- TypeError: Wrong type
- IndexError: Invalid index
- KeyError: Missing dictionary key
- ZeroDivisionError: Division by zero

### Modules and Packages

**Importing**
- import numpy: Full module namespace
- from numpy import array: Specific items
- from numpy import * : All items (avoid, causes conflicts)
- import numpy as np: Aliasing

**Creating Modules**
- File with .py extension is module
- Packages: Folders with __init__.py
- Relative imports: from . import sibling
- Absolute imports: from package.module import item

## NumPy: Numerical Computing

### Why NumPy?
- Speed: ~100x faster than Python lists
- Memory: Efficient data storage
- Broadcasting: Vectorized operations
- Integration: Foundation for pandas, scikit-learn, etc.

### Creating Arrays

```python
import numpy as np

# From Python lists
np.array([1, 2, 3])
np.array([[1, 2], [3, 4]])  # 2D array

# Special arrays
np.zeros((3, 4))  # 3x4 zeros
np.ones((2, 3))   # 2x3 ones
np.eye(3)         # 3x3 identity
np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5) # 5 points 0 to 1
np.random.randn(3, 4) # Normal distribution
```

### Array Operations

**Element-wise Operations**
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b  # [5, 7, 9]
a * b  # [4, 10, 18]
a ** 2 # [1, 4, 9]
np.sqrt(a)  # Square root
np.exp(a)   # Exponential
np.log(a)   # Natural log
```

**Matrix Operations**
- @ operator: Matrix multiplication
- np.dot(a, b): Inner product
- np.outer(a, b): Outer product
- np.transpose(a) or a.T: Transpose

### Indexing and Slicing

**Single Index**
```python
a = np.array([0, 1, 2, 3, 4, 5])
a[0]   # First element: 0
a[-1]  # Last element: 5
a[2:5] # Slice [2,3,4]
```

**Multi-dimensional**
```python
b = np.array([[1, 2, 3], [4, 5, 6]])
b[0, 1]    # Row 0, column 1: 2
b[0, :]    # First row: [1, 2, 3]
b[:, 1]    # Second column: [2, 5]
b[0:1, 1:2] # Subarray
```

**Boolean Indexing**
```python
mask = a > 2
a[mask]  # Elements > 2
```

### Broadcasting

**Why Broadcasting?**
Vectorized operations on arrays of different shapes

**Broadcasting Rules**
1. Dimensions align from right to left
2. Dimensions are compatible if equal or one is 1
3. Missing dimensions treated as 1

**Examples**
```python
a = np.array([[1, 2, 3]])  # Shape (1, 3)
b = np.array([[1], [2], [3]])  # Shape (3, 1)
a + b  # Shape (3, 3), broadcasts both

c = np.array([1, 2, 3])  # Shape (3,)
d = np.array([[1], [2]])  # Shape (2, 1)
c + d  # Shape (2, 3)
```

### Common NumPy Functions

**Aggregation**
- np.sum(): Sum all elements
- np.mean(): Average
- np.std(): Standard deviation
- np.min(), np.max(): Minimum, maximum
- np.argmin(), np.argmax(): Index of min/max

**Linear Algebra**
- np.linalg.inv(): Matrix inverse
- np.linalg.det(): Determinant
- np.linalg.eig(): Eigenvalues, eigenvectors
- np.linalg.solve(): System of equations
- np.linalg.norm(): Vector/matrix norm

**Random**
- np.random.rand(): Uniform [0, 1)
- np.random.randn(): Standard normal
- np.random.choice(): Sample from array
- np.random.shuffle(): In-place shuffle

### Performance Tips

**Avoid Loops**
Replace Python loops with vectorized NumPy code
```python
# Slow
result = []
for x in array:
    result.append(x ** 2)

# Fast
result = array ** 2
```

**Memory Efficiency**
- Use .astype() for appropriate dtypes
- Avoid unnecessary copies
- Use views when possible (slicing)

**Benchmarking**
```python
import timeit
timeit.timeit('x ** 2', 'x = np.array(range(1000))')
```

## Pandas: Data Manipulation

### Why Pandas?
- Tabular data: Rows and columns like Excel
- Missing data: Handles NaN gracefully
- Data alignment: Automatic alignment by index
- Flexibility: Mix numeric, string, categorical data

### Creating DataFrames

```python
import pandas as pd

# From dict
df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'],
                  'age': [25, 30, 35]})

# From lists
df = pd.DataFrame([['Alice', 25], ['Bob', 30]],
                 columns=['name', 'age'])

# From CSV
df = pd.read_csv('data.csv')
```

### Accessing Data

**Column Access**
```python
df['name']      # Series
df[['name', 'age']]  # DataFrame
df.name         # Attribute access (if no spaces)
```

**Row Access**
```python
df.loc[0]       # By label
df.iloc[0]      # By position
df.loc[0, 'name']  # Specific cell
```

**Conditional Selection**
```python
df[df['age'] > 25]
df[(df['age'] > 25) & (df['name'] == 'Bob')]
```

### Data Cleaning

**Missing Values**
```python
df.isnull()      # Check for NaN
df.dropna()      # Remove rows with NaN
df.fillna(0)     # Replace with value
df.fillna(method='ffill')  # Forward fill
```

**Duplicates**
```python
df.duplicated()  # Find duplicates
df.drop_duplicates()  # Remove duplicates
```

**Data Types**
```python
df.dtypes        # Check types
df.astype({'age': 'int32'})  # Convert
```

### Data Transformation

**Sorting**
```python
df.sort_values('age')  # Ascending
df.sort_values('age', ascending=False)  # Descending
```

**Grouping**
```python
df.groupby('department').sum()
df.groupby('department')['salary'].mean()
```

**Aggregation**
```python
df.agg({'age': 'mean', 'salary': 'sum'})
df.describe()  # Summary statistics
```

**Merging**
```python
pd.merge(df1, df2, on='key')  # Inner join
pd.merge(df1, df2, how='left', on='key')
```

## Linear Algebra Fundamentals

### Vectors and Matrices

**Vectors**
- 1D array of numbers
- Direction and magnitude
- Example: [1, 2, 3]
- Notation: **v** or v⃗

**Matrices**
- 2D array: m rows × n columns
- Example: [[1, 2], [3, 4]]
- Notation: **A** or A_ij

**Tensors**
- Generalization to n dimensions
- Images: 3D (height, width, channels)
- Batches: 4D (batch, height, width, channels)

### Vector Operations

**Magnitude (Norm)**
$$||v|| = \sqrt{v_1^2 + v_2^2 + ... + v_n^2}$$

**Dot Product (Inner Product)**
$$v \cdot w = v_1 w_1 + v_2 w_2 + ... + v_n w_n$$

**Geometric Interpretation**
$$v \cdot w = ||v|| ||w|| \cos(\theta)$$
- θ = 0: Parallel vectors
- θ = 90°: Orthogonal (perpendicular)
- θ = 180°: Opposite vectors

### Matrix Operations

**Addition and Subtraction**
- Element-wise: A + B
- Same shape required

**Scalar Multiplication**
- Each element multiplied by scalar
- c * A = [c*a_ij]

**Matrix Multiplication**
$$C = AB \text{ where } c_{ij} = \sum_k a_{ik} b_{kj}$$
- Not commutative: AB ≠ BA
- Associative: (AB)C = A(BC)
- A: shape (m, n), B: shape (n, p) → C: shape (m, p)

**Transpose**
$$A^T_{ij} = A_{ji}$$
- Switch rows and columns

### Matrix Decomposition

**Eigendecomposition**
$$Av = \lambda v$$
- v: Eigenvector
- λ: Eigenvalue
- A must be square

**Singular Value Decomposition (SVD)**
$$A = U \Sigma V^T$$
- U, V: Orthogonal matrices
- Σ: Diagonal matrix of singular values
- Works for any matrix

**QR Decomposition**
$$A = QR$$
- Q: Orthogonal matrix
- R: Upper triangular
- Used in least squares

### Solving Linear Systems

**System of Equations**
$$Ax = b$$
- A: Coefficient matrix
- x: Unknown variables
- b: Constants

**Solution Methods**
1. Matrix inverse: x = A^(-1)b (if invertible)
2. Gaussian elimination: Direct computation
3. Iterative methods: Gradient descent, conjugate gradient
4. LU decomposition: Factorize A

**Computational Considerations**
- Computational cost: O(n³) for direct methods
- Numerical stability: Avoid near-singular matrices
- Sparse systems: Use specialized algorithms

## Calculus for Machine Learning

### Derivatives and Gradients

**Derivative**
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$
- Rate of change
- Slope of tangent line

**Gradient (Multivariable)**
$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \end{bmatrix}$$
- Vector of partial derivatives
- Points in direction of steepest increase

