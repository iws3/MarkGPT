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

## Advanced NumPy Concepts

### Structured Arrays
```python
dt = np.dtype([('name', 'U10'), ('age', 'i4')])
data = np.array([('Alice', 25), ('Bob', 30)], dtype=dt)
data['name']  # Access by field
```

**Use Cases**
- Database-like records
- Mixed data types
- Memory-efficient storage

### Memory Layout

**C-Contiguous vs Fortran-Contiguous**
```python
a = np.array([[1, 2], [3, 4]])  # C-order (row-major)
f = np.asfortranarray(a)  # F-order (column-major)
```

**Performance Implications**
- Row-major: Faster row iteration
- Column-major: Faster column iteration
- NumPy default: C-contiguous

### Advanced Slicing

**Fancy Indexing**
```python
a = np.arange(10)
indices = np.array([0, 2, 4])
a[indices]  # [0, 2, 4]

a[[0, 2, 4]]  # Same result
```

**Multidimensional Indexing**
```python
a = np.arange(12).reshape(3, 4)
rows = np.array([0, 2])
cols = np.array([1, 3])
a[rows, cols]  # Elements at (0,1) and (2,3)
```

### Universal Functions (ufuncs)

**Built-in Ufuncs**
```python
# Trigonometric
np.sin, np.cos, np.tan

# Exponential/logarithm
np.exp, np.log, np.log10

# Rounding
np.floor, np.ceil, np.round

# Element-wise comparison
np.greater, np.less, np.equal
```

**Custom Ufuncs**
- Vectorize functions for arrays
- Broadcasting built-in

## Advanced Pandas

### MultiIndex DataFrames

```python
arrays = [['bar', 'bar', 'baz'],
          ['one', 'two', 'one']]
index = pd.MultiIndex.from_arrays(arrays)
df = pd.DataFrame(np.random.randn(3, 2), index=index)
```

**Advantages**
- Hierarchical indexing
- Flexible grouping
- Efficient storage

### Time Series

```python
df = pd.read_csv('data.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# Resampling
df.resample('D').sum()  # Daily
df.resample('W').mean()  # Weekly

# Time-based indexing
df.loc['2020-01-01':'2020-12-31']
df.loc['2020-01']  # All January 2020
```

### Categorical Data

```python
df['color'] = pd.Categorical(['red', 'blue', 'red', 'green'])
df['color'].cat.categories
df['color'].cat.codes  # Numeric representation
```

**Benefits**
- Memory efficient (especially for many repeated values)
- Enforces allowed values
- Useful for ordinal data

### Pivot and Reshape

```python
# Pivot table
df.pivot_table(values='sales', index='date', columns='product')

# Unpivot (melt)
pd.melt(df, id_vars=['id'], value_vars=['col1', 'col2'])

# Stack/Unstack
df.stack()    # Wide to long
df.unstack()  # Long to wide
```

## Linear Algebra Applications

### Least Squares

**Problem**
Solve Ax = b when no exact solution exists

**Solution**
$$x = (A^T A)^{-1} A^T b$$

**NumPy Implementation**
```python
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
```

### Principal Component Analysis (PCA)

**Mathematical Basis**
1. Compute covariance matrix: Σ = (1/n) X^T X
2. Find eigenvalues/eigenvectors of Σ
3. Project X onto principal components

**Python Using SVD**
```python
U, S, V = np.linalg.svd(X, full_matrices=False)
# S contains singular values (related to variance)
```

### Matrix Norms

**Frobenius Norm**
$$||A||_F = \sqrt{\sum_{ij} A_{ij}^2}$$

**Spectral Norm (L2)**
$$||A||_2 = \sigma_{max}(A)$$

**NumPy**
```python
np.linalg.norm(A)           # Frobenius
np.linalg.norm(A, ord=2)    # Spectral
np.linalg.norm(A, ord='fro') # Frobenius
```

### Condition Number

**Definition**
$$\kappa(A) = ||A|| \cdot ||A^{-1}||$$

**Interpretation**
- κ ≈ 1: Well-conditioned
- κ → ∞: Ill-conditioned, numerically unstable

**NumPy**
```python
np.linalg.cond(A)
```

## Advanced Calculus

### Numerical Differentiation

**Forward Difference**
$$f'(x) \approx \frac{f(x+h) - f(x)}{h}$$

**Central Difference**
$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

**Python Example**
```python
def numerical_gradient(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)
```

### Numerical Integration

**Trapezoidal Rule**
$$\int_a^b f(x)dx \approx \sum_i \frac{f(x_i) + f(x_{i+1})}{2} \cdot h$$

**Simpson's Rule**
Higher accuracy using parabolic approximation

**SciPy**
```python
from scipy.integrate import quad
result, error = quad(f, a, b)
```

### Taylor Series

**Definition**
$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + ...$$

**Applications**
- Function approximation
- Error analysis
- Asymptotic behavior

**Example**
- e^x ≈ 1 + x + x²/2! + x³/3! + ...
- sin(x) ≈ x - x³/3! + x⁵/5! - ...

## Statistics Applications

### Distribution Fitting

**Parametric Fitting**
```python
from scipy import stats

# Fit normal distribution
mu, sigma = stats.norm.fit(data)

# Fit exponential
lambda_param = stats.expon.fit(data)[1]
```

### Hypothesis Testing

**t-test: Compare Two Means**
```python
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(group1, group2)
```

**ANOVA: Multiple Groups**
```python
from scipy.stats import f_oneway
f_stat, p_value = f_oneway(group1, group2, group3)
```

**Chi-square: Categorical**
```python
from scipy.stats import chi2_contingency
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
```

### Correlation and Covariance

**Pearson Correlation**
$$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i(x_i-\bar{x})^2 \sum_i(y_i-\bar{y})^2}}$$

**Python**
```python
np.corrcoef(x, y)
pd.DataFrame(data).corr()
```

**Spearman Correlation**
- Rank-based (non-parametric)
- Robust to outliers

### Bayesian Statistics

**Bayes Theorem**
$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

**Prior, Likelihood, Posterior**
- Prior P(A): Belief before data
- Likelihood P(B|A): Data given hypothesis
- Posterior P(A|B): Updated belief

**Conjugate Priors**
- Simple closed-form updates
- Normal-normal, beta-binomial

## Optimization Algorithms

### Gradient Descent Variants

**Batch vs Stochastic**
- Batch: Update on full dataset
- Stochastic: Update on single sample
- Mini-batch: Update on small batch

**Convergence Analysis**
- Learning rate: α controls step size
- Too small: Slow convergence
- Too large: Divergence or oscillation

### Second-Order Methods

**Newton's Method**
$$x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$$

**Advantages**
- Quadratic convergence (faster)
- Uses Hessian information

**Disadvantages**
- Hessian expensive to compute
- May not converge if Hessian singular

**Quasi-Newton (BFGS)**
- Approximate Hessian
- Practical alternative

### Constrained Optimization

**Lagrange Multipliers**
$$\nabla f = \lambda \nabla g$$

**KKT Conditions**
- Generalization to inequalities
- Necessary conditions for optimality

**Penalty Methods**
- Add constraint penalties to objective
- Solve unconstrained problems

## Working with Real Data

### Data Loading and Format Conversions

**CSV Files**
```python
df = pd.read_csv('data.csv')
df.to_csv('output.csv', index=False)
```

**JSON**
```python
df = pd.read_json('data.json')
df.to_json('output.json')
```

**Excel**
```python
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df.to_excel('output.xlsx')
```

### Data Validation

**Checking Data Quality**
```python
# Duplicates
df[df.duplicated()].shape[0]

# Missing values
df.isnull().sum()

# Data types
df.dtypes

# Statistical summary
df.describe()
```

### Handling Missing Data

**Strategies**
1. Deletion: Remove rows/columns
2. Mean/median imputation: Simple
3. Forward/backward fill: Time series
4. KNN imputation: Use neighbors
5. Model-based: Learn imputation

**Python Examples**
```python
df.dropna()           # Delete
df.fillna(df.mean())  # Mean
df.fillna(method='ffill')  # FF
```

### Outlier Detection

**Statistical Methods**
- Z-score: > 3 or < -3
- IQR: Outside Q1 - 1.5*IQR to Q3 + 1.5*IQR

**Distance-based**
- Isolation Forest
- Local Outlier Factor (LOF)

**Python**
```python
from sklearn.preprocessing import StandardScaler
z_scores = np.abs(StandardScaler().fit_transform(df))
outliers = (z_scores > 3).any(axis=1)
```

## Visualization with Matplotlib

### Basic Plotting

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 4, 9])
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Title')
plt.show()

plt.scatter(x, y)
plt.hist(data, bins=20)
plt.boxplot([data1, data2])
```

### Styling and Customization

**Subplots**
```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
```

**Colors and Styles**
```python
plt.plot(x, y, 'r-', linewidth=2)  # Red line
plt.plot(x, y, 'b.', markersize=10)  # Blue dots
```

**Annotations**
```python
plt.annotate('Peak', xy=(1.5, 10), xytext=(2, 11),
             arrowprops=dict(arrowstyle='->'))
```

### 3D Visualization

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.show()
```

**Surface Plots**
```python
ax.plot_surface(X, Y, Z)
```

## Performance and Debugging

### Timing and Profiling

**Simple Timing**
```python
import time
start = time.time()
# Code
end = time.time()
print(f'Time: {end - start} seconds')
```

**Using timeit**
```python
import timeit
result = timeit.timeit('x**2', 'x=np.array(range(1000))', number=1000)
```

### Memory Profiling

**Line Profiler**
```python
# Install: pip install line_profiler
# Use: kernprof -l -v script.py
@profile
def my_function():
    # Function body
    pass
```

**Memory Usage**
```python
import tracemalloc
tracemalloc.start()
# Code
current, peak = tracemalloc.get_traced_memory()
```

### Debugging Techniques

**Print Debugging**
```python
print(f'Variable x: {x}, type: {type(x)}')
print(f'Array shape: {arr.shape}, dtype: {arr.dtype}')
```

**Assertion Checks**
```python
assert len(data) > 0, 'Data cannot be empty'
assert np.all(np.isfinite(X)), 'NaN or Inf in data'
```

**Using pdb**
```python
import pdb; pdb.set_trace()
# Now in debugger, can inspect variables
```

## Machine Learning Preprocessing Pipeline

### Feature Scaling

**Standardization**
$$x_{scaled} = \frac{x - \mu}{\sigma}$$

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)
```

**Normalization (Min-Max)**
$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X)
```

### Feature Engineering

**Polynomial Features**
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

**Interaction Terms**
- Important for capturing relationships
- Exponential growth in features
- Use domain knowledge to select

**Manual Feature Creation**
```python
df['log_income'] = np.log(df['income'])
df['age_squared'] = df['age'] ** 2
```

### Feature Selection

**Filter Methods**
- Correlation with target
- Chi-square for categorical
- Fast, independent of model

```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)
X_new = selector.fit_transform(X, y)
```

**Wrapper Methods**
- RFE (Recursive Feature Elimination)
- Forward/backward selection
- Model-dependent, computationally expensive

**Embedded Methods**
- Regularization (L1, L2)
- Tree feature importance

## Advanced Machine Learning Techniques

### Ensemble Methods

**Bagging (Bootstrap Aggregating)**
- Random samples with replacement
- Train independent models
- Aggregate predictions (average/vote)
- Reduces variance

```python
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(estimator=DecisionTreeClassifier())
bag.fit(X, y)
```

### Random Forests

**Algorithm**
1. Bootstrap samples
2. Train decision tree on random subset of features
3. Repeat many times
4. Aggregate predictions

**Advantages**
- Reduces overfitting
- Handles non-linearity well
- Feature importance built-in
- Parallelizable

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
```

### Gradient Boosting

**Process**
1. Start with initial prediction (mean/mode)
2. Fit model to residuals
3. Add scaled prediction to ensemble
4. Repeat on new residuals

**Key Differences from Bagging**
- Sequential (not parallel)
- Learns from errors (residuals)
- Often better performance
- More prone to overfitting

```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(learning_rate=0.1)
gb.fit(X, y)
```

### XGBoost and LightGBM

**XGBoost Advantages**
- Regularization built-in
- Handles missing values
- GPU acceleration available
- Highly optimized implementation

```python
import xgboost as xgb
model = xgb.XGBClassifier(learning_rate=0.1, max_depth=5)
model.fit(X, y)
```

**LightGBM**
- Faster training
- Lower memory usage
- Handles large datasets well

## Support Vector Machines

**Mathematical Foundation**
Maximize margin while minimizing misclassification:
$$\min_{w,b} \frac{1}{2}||w||^2 + C \sum_i \xi_i$$

Subject to: $y_i(w \cdot x_i + b) \geq 1 - \xi_i$

**Kernels**
- Linear: Simple, fast
- RBF (Radial Basis Function): Non-linear, default
- Polynomial: Degree control
- Custom: Domain-specific

```python
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X, y)
```

## Dimensionality Reduction

### Why Reduce Dimensions

- Curse of dimensionality
- Computational efficiency
- Visualization
- Remove noise and multicollinearity
- Better generalization

**Trade-off**
- Information loss
- Interpretability

### Linear Dimensionality Reduction

**Principal Component Analysis (PCA)**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f'Explained variance: {pca.explained_variance_ratio_}')
```

**Linear Discriminant Analysis (LDA)**
- Supervised (uses labels)
- Finds directions that maximize class separation

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_reduced = lda.fit_transform(X, y)
```

### Non-linear Dimensionality Reduction

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- Preserves local structure
- Great for visualization
- Computationally expensive

```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30)
X_reduced = tsne.fit_transform(X)
```

**UMAP (Uniform Manifold Approximation and Projection)**
- Faster than t-SNE
- Preserves more global structure
- Excellent for large datasets

## Clustering

### K-Means Clustering

**Algorithm**
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids as cluster mean
4. Repeat until convergence

**Choosing k**
- Elbow method: Plot inertia vs k
- Silhouette score: Measure cluster quality
- Domain knowledge

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
```

### Hierarchical Clustering

**Agglomerative (Bottom-up)**
1. Start with each point as cluster
2. Merge closest clusters
3. Repeat until one cluster

**Linkage Methods**
- Single: Minimum distance
- Complete: Maximum distance
- Average: Mean distance
- Ward: Minimize within-cluster variance

```python
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X, method='ward')
dendrogram(Z)
```

### DBSCAN

**Algorithm**
- Density-based
- No need to specify k
- Finds arbitrary shapes
- Identifies outliers as noise points

**Parameters**
- eps: Neighborhood radius
- min_samples: Min points in neighborhood

```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
```

## Model Evaluation

### Classification Metrics

**Confusion Matrix**
- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

**Accuracy**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision**
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall**
$$\text{Recall} = \frac{TP}{TP + FN}$$

### F1-Score and ROC

**F1-Score (Harmonic Mean of Precision and Recall)**
$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**ROC Curve**
- Plot TPR vs FPR
- AUC (Area Under Curve): Probability of correct ranking
- Higher AUC = Better discrimination

```python
from sklearn.metrics import roc_auc_score, roc_curve
auc = roc_auc_score(y_true, y_proba)
fpr, tpr, thresholds = roc_curve(y_true, y_proba)
```

### Regression Metrics

**Mean Squared Error (MSE)**
$$MSE = \frac{1}{n} \sum_i (y_i - \hat{y}_i)^2$$

**Root Mean Squared Error (RMSE)**
$$RMSE = \sqrt{MSE}$$

**Mean Absolute Error (MAE)**
$$MAE = \frac{1}{n} \sum_i |y_i - \hat{y}_i|$$

**R² (Coefficient of Determination)**
$$R^2 = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2}$$

### Cross-Validation

**Why Cross-Validation**
- Better use of data
- More reliable performance estimate
- Detect overfitting

**K-Fold Cross-Validation**
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f'Mean score: {scores.mean():.3f} (+- {scores.std():.3f})')
```

**Stratified K-Fold**
- Maintains class distribution
- Important for imbalanced datasets

### Hyperparameter Tuning

**Grid Search**
```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X, y)
print(f'Best params: {grid.best_params_}')
```

**Random Search**
- More efficient for large parameter spaces
- Sample random combinations

## Text Processing Fundamentals

### Tokenization and Vectorization

**Word Tokenization**
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)
vocab = vectorizer.get_feature_names_out()
```

**TF-IDF (Term Frequency-Inverse Document Frequency)**
$$TF\text{-}IDF(t,d) = TF(t,d) \cdot IDF(t)$$
$$IDF(t) = \log\frac{N}{n_t}$$

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(texts)
```

### Word Embeddings

**Dense Representations**
- Capture semantic meaning
- Lower dimensionality than one-hot
- Can transfer across tasks

**Word2Vec**
- Skip-gram or CBOW architecture
- Learned from context windows
- Pre-trained models available

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5)
vector = model.wv['word']
```

## Time Series Analysis

### Decomposition

$$Y_t = T_t + S_t + R_t$$

- Trend (T): Long-term direction
- Seasonality (S): Regular patterns
- Residual (R): Random noise

```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(series, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
```

### Autoregressive Methods

**AR (Autoregressive)**
$$y_t = c + \phi_1 y_{t-1} + ... + \phi_p y_{t-p} + \epsilon_t$$

**MA (Moving Average)**
$$y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q}$$

**ARIMA (AutoRegressive Integrated Moving Average)**
- I: Differencing for stationarity
- ARIMA(p,d,q)

```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(series, order=(1, 1, 1))
results = model.fit()
```

### Forecasting and Uncertainty

**Point Forecasts vs Intervals**
- Point forecast: Single predicted value
- Confidence intervals: Range of likely values
- Prediction intervals: Wider (include model error)

**Evaluation**
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

## Image Processing Basics

### Loading and Manipulating Images

```python
from PIL import Image
import cv2

# PIL
img = Image.open('image.jpg')
img_array = np.array(img)

# OpenCV
img = cv2.imread('image.jpg')
# Returns BGR (not RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

**Image Properties**
- Shape: (height, width, channels)
- Dtype: uint8 (0-255) or float (0-1)
- Channels: Grayscale (1), RGB (3), RGBA (4)

### Filters and Transformations

**Convolution (2D Filtering)**
- Blur: Average neighboring pixels
- Edge detection: Sobel, Canny
- Sharpening: Enhance edges

```python
# Blur
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Edge detection
edges = cv2.Canny(img, 100, 200)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(img, kernel)
eroded = cv2.erode(img, kernel)
```

## Video Processing

### Reading and Writing Videos

```python
import cv2

# Read video
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Process frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

# Write video
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
while # condition:
    # get or create frame
    out.write(frame)
out.release()
```

## Natural Language Processing Basics

### Preprocessing

**Lowercasing and Punctuation**
```python
text = text.lower()
import string
text = text.translate(str.maketrans('', '', string.punctuation))
```

**Stopword Removal**
```python
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in tokens if w not in stop_words]
```

**Stemming and Lemmatization**
- Stemming: Reduce to root (running → run)
- Lemmatization: Reduce to base form (better but slower)

### Named Entity Recognition (NER)

**Task: Identify entities**
- PERSON, ORG, LOCATION, DATE, etc.

```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp('Apple is looking to hire in San Francisco.')
for ent in doc.ents:
    print(ent.text, ent.label_)
```

**Library Options**
- spaCy: Fast, production-ready
- NLTK: Educational
- Transformers: State-of-the-art but slower

### Sentiment Analysis

**Task: Classify text emotion/sentiment**

**Approaches**
1. Rule-based: Lexicon of positive/negative words
2. Machine learning: Trained on labeled data
3. Deep learning: Neural networks

```python
from textblob import TextBlob
text = 'This movie is absolutely fantastic!'
blob = TextBlob(text)
print(f'Polarity: {blob.sentiment.polarity}')
```

## Working with APIs and Web Data

### HTTP Requests

```python
import requests

# GET request
response = requests.get('https://api.example.com/data')
if response.status_code == 200:
    data = response.json()
else:
    print(f'Error: {response.status_code}')

# POST request
response = requests.post('https://api.example.com/submit',
                        json={'key': 'value'})
```

### JSON and Data Formats

**JSON (JavaScript Object Notation)**
```python
import json

# Parse JSON
data = json.loads(json_string)

# Create JSON
json_string = json.dumps(data, indent=2)
```

**XML**
```python
import xml.etree.ElementTree as ET

tree = ET.parse('file.xml')
root = tree.getroot()
for child in root:
    print(child.tag, child.attrib)
```

### Web Scraping

**BeautifulSoup**
```python
from bs4 import BeautifulSoup

response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find elements
titles = soup.find_all('h1')
for title in titles:
    print(title.text)
```

**Ethical Considerations**
- Check robots.txt
- Respect rate limits
- Follow terms of service
- Use APIs when available

## Case Study 1: Iris Dataset Classification

**Dataset Overview**
- 150 samples, 4 features (sepal length, width, petal length, width)
- 3 classes (Setosa, Versicolor, Virginica)
- Classic machine learning dataset

**Pipeline**
1. Load and explore data
2. Train-test split
3. Scale features
4. Train classifier (e.g., SVM, Random Forest)
5. Evaluate with cross-validation
6. Visualize decision boundaries

### Iris: Exploration and Visualization

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Summary statistics
print(df.describe())

# Visualization
import matplotlib.pyplot as plt
plt.scatter(df['sepal length'], df['sepal width'],
           c=df['target'], cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
```

### Iris: Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multiple models
svm = SVC(kernel='rbf')
rf = RandomForestClassifier()

svm.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluate
print('SVM:', svm.score(X_test, y_test))
print('RF:', rf.score(X_test, y_test))
print(classification_report(y_test, svm.predict(X_test)))
```

## Case Study 2: Housing Price Prediction

**Problem Setup**
- Regression task: Predict house prices
- Multiple features: Size, location, age, etc.
- Real-world data challenges:
  - Missing values
  - Outliers
  - Multicollinearity
  - Categorical features

**Typical Data Source**
- Boston Housing, Ames Housing, or Kaggle datasets

### Housing: Feature Engineering

```python
import pandas as pd
import numpy as np

df = pd.read_csv('housing.csv')

# Handle missing data
df = df.dropna(subset=['critical_column'])
df['numeric_col'].fillna(df['numeric_col'].mean(), inplace=True)

# Categorical encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
```

### Housing: Model Selection

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score

X = df.drop('price', axis=1)
y = df['price']

models = {
    'Linear': LinearRegression(),
    'GBM': GradientBoostingRegressor(),
    'XGB': xgb.XGBRegressor()
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f'{name}: {scores.mean():.4f} (+- {scores.std():.4f})')
```

## Case Study 3: Customer Segmentation

**Business Problem**
- Identify customer groups
- Tailor marketing strategies
- Improve customer retention

**Data Features**
- Demographics: Age, income, location
- Behavior: Purchase history, frequency
- RFM: Recency, Frequency, Monetary

**Unsupervised Learning Approach**
- No predefined customer categories
- K-means or hierarchical clustering

### Segmentation: Clustering

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Prepare data
X = df[['income', 'spending', 'frequency']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(K, inertias)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
```

### Segmentation: Analysis

```python
# Train final model
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze segments
for i in range(3):
    segment = df[df['cluster'] == i]
    print(f'Segment {i}:')
    print(segment[['income', 'spending', 'frequency']].describe())
    print()

# Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    segment = df[df['cluster'] == i]
    ax.scatter(segment['income'], segment['spending'],
              segment['frequency'], label=f'Segment {i}')
ax.legend()
plt.show()
```

## Appendix A: Python Ecosystem Overview

**Numerical Computing**
- NumPy: Arrays, linear algebra
- SciPy: Scientific computing, optimization

**Data Manipulation**
- Pandas: DataFrames, data cleaning
- Dask: Parallel processing

**Machine Learning**
- scikit-learn: Classical ML
- XGBoost, LightGBM: Gradient boosting
- statsmodels: Statistical modeling

**Deep Learning**
- TensorFlow/Keras: Neural networks
- PyTorch: Research-friendly
- JAX: NumPy-like with autodiff

## Appendix B: Mathematics Quick Reference

**Calculus**
- Derivative: Rate of change
- Chain rule: d/dx f(g(x)) = f'(g(x)) · g'(x)
- Gradient: Vector of partial derivatives

**Linear Algebra**
- Determinant: Volume scaling factor
- Eigenvalue: Scaling factor under transformation
- Rank: Dimension of column/row space

**Probability**
- PDF: Probability density
- CDF: Cumulative probability
- Expectation: Mean/average

## Appendix C: Common Pitfalls Summary

1. Data leakage: Fitting scaler on test set
2. Class imbalance: Not using stratified split
3. Feature scaling: Forgetting for distance-based models
4. Overfitting: Too complex models on small data
5. Wrong metric: Using accuracy for imbalanced data
6. NaN handling: Not checking for missing values
7. Broadcasting: Unexpected behavior with shapes
8. Mutable defaults: List comprehensions in loops
9. Memorization: Testing on training data
10. Correlation ≠ causation: Confounding variables

## Appendix D: Performance Optimization Checklist

**Code Level**
- Use vectorized operations (NumPy, Pandas)
- Avoid Python loops when possible
- Use list comprehensions over loops
- Profile before optimizing

**Data Level**
- Subsample for exploration
- Select relevant features
- Use appropriate data types
- Index large DataFrames

**Computation Level**
- Parallelize (joblib, Dask)
- Use GPU acceleration (CuPy, RAPIDS)
- Batch processing for large datasets
- Cache expensive computations

## Appendix E: Hyperparameter Ranges

**Decision Trees**
- max_depth: 3-30
- min_samples_split: 2-20
- min_samples_leaf: 1-10

**Random Forest**
- n_estimators: 100-1000
- max_depth: 5-50
- max_features: 'auto', 'sqrt', 'log2'

**SVM**
- C: 0.001-1000 (log scale)
- kernel: 'linear', 'rbf', 'poly'
- gamma: 0.0001-10 (for rbf)

## Appendix F: Useful Command Line Tools

**Jupyter**
```bash
jupyter notebook              # Start server
jupyter lab                   # Modern interface
jupyter nbconvert --help      # Convert notebooks
```

**Git**
```bash
git log --oneline             # View history
git diff                      # Show changes
git stash                     # Temporary save
```

**Python Development**
```bash
python -m pip install -r requirements.txt
python -m pytest tests/        # Run tests
python -m black .             # Format code
```

## Appendix G: Glossary

**Broadcasting**: Automatically expanding dimensions for operations
**Epoch**: One complete pass through training data
**Gradient**: Vector of partial derivatives
**Hyperparameter**: Parameter set before training
**Regularization**: Technique to prevent overfitting
**Residual**: Difference between predicted and actual
**Scalability**: Ability to handle growing data
**Stationarity**: Statistical properties don't change over time
**Vectorization**: Operating on arrays without explicit loops
**Whitening**: Normalizing to zero mean and unit variance

## Appendix H: Learning Resources

**Online Courses**
- Coursera: ML Specialization
- Fast.ai: Practical DL
- Andrew Ng's courses: Fundamentals

**Books**
- 'Hands-On ML' by Aurélien Géron
- 'The Hundred-Page ML Book'
- 'ISLR' for statistics

**Documentation**
- scikit-learn docs: Examples for every algorithm
- NumPy/Pandas tutorials: Official guides
- TensorFlow/PyTorch guides: DL frameworks

## Appendix I: Setting Up Your Python Environment

**Virtual Environment**
```bash
python -m venv ml_env
source ml_env/bin/activate      # Linux/Mac
ml_env\Scripts\activate.bat    # Windows
```

**Essential Packages**
```bash
pip install numpy pandas matplotlib
pip install scikit-learn scipy
pip install jupyter notebook
pip install tensorflow keras  # or pytorch
```

**Environment Management**
```bash
conda create -n myenv python=3.9
conda activate myenv
conda install -c conda-forge numpy pandas
```

## Appendix J: Module 02 Completion References

**Topics Covered**
- Python fundamentals and data structures
- Advanced NumPy operations and memory management
- Pandas data manipulation and time series
- Linear algebra and matrix operations
- Calculus and optimization fundamentals
- Probability theory and statistics
- Machine learning algorithms and techniques
- Model evaluation and hyperparameter tuning
- Data preprocessing and feature engineering
- Practical applications: NLP, vision, time series

**Next Steps**
Apply these fundamentals to real-world projects in modules 1.2+
Combine concepts: e.g., feature engineering + ensemble methods
Deepen specialization: NLP, CV, timeseries, or RL

## Advanced NumPy Patterns

### Masking and Boolean Indexing

```python
a = np.array([1, 2, 3, 4, 5])
mask = a > 2
print(a[mask])  # [3 4 5]

a[a > 2] = 0  # Modify in place
print(a)  # [1 2 0 0 0]
```

**Complex Masks**
```python
# Multiple conditions
mask = (a > 2) & (a < 5)
a[mask] = -1

# Using np.where
result = np.where(a > 2, a, 0)
```

### Einsum: Einstein Summation

**Matrix Multiplication**
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Traditional
C = A @ B

# Using einsum
C = np.einsum('ij,jk->ik', A, B)
```

**Trace and Diagonal**
```python
trace = np.einsum('ii->', A)  # Sum of diagonal
diag = np.einsum('ii->i', A)  # Diagonal elements
```

**Benefits**
- More readable for complex operations
- Better memory efficiency
- Automatic optimization

### Strides and Strided Arrays

```python
a = np.arange(12).reshape(3, 4)
print(f'Strides: {a.strides}')  # (16, 4) bytes

# Create stride view without copying
every_other = a[::2, ::2]
print(every_other.strides)  # (32, 8)

# Rolling window
from numpy.lib.stride_tricks import as_strided
shape = (10, 3)
strides = (8, 8)  # 8 bytes apart
windows = as_strided(np.arange(12), shape=shape, strides=strides)
```

## Advanced Pandas Strategies

### GroupBy Operations

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B'],
    'value': [10, 20, 30, 40]
})

# Aggregate multiple functions
df.groupby('category').agg({
    'value': ['sum', 'mean', 'std']
})

# Custom aggregation
def range_val(x):
    return x.max() - x.min()

df.groupby('category')['value'].agg(range_val)

# Transform
df['normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()
```

### Join and Merge Operations

**Inner Join**
```python
left = pd.DataFrame({'key': ['a', 'b', 'c'], 'x': [1, 2, 3]})
right = pd.DataFrame({'key': ['a', 'b', 'd'], 'y': [4, 5, 6]})

result = pd.merge(left, right, on='key', how='inner')
# Only 'a', 'b' matched
```

**Outer Join**
```python
result = pd.merge(left, right, on='key', how='outer')
# All keys: 'a', 'b', 'c', 'd'
# NaN where missing
```

**Concatenation**
```python
df1 = pd.DataFrame({'A': [1, 2]})
df2 = pd.DataFrame({'A': [3, 4]})
result = pd.concat([df1, df2], ignore_index=True)
```

