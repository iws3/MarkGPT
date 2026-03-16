# Lesson L03.2: Matrices as Transformations
## Day 3: Mathematics You Already Know | How Linear Transformations Work

### Lesson Overview
A matrix is a rectangular array of numbers. But what does it *do*? In this lesson, you will learn to see matrices not as collections of data, but as machines that transform vectors. A matrix encodes a geometric transformation: rotation, reflection, scaling, or combinations thereof. In neural networks, every layer is fundamentally a matrix multiplication—a transformation of one vector space into another. By the end of this lesson, you will have geometric intuition for matrix operations, understand how multiplying a vector by a matrix changes it, and see why matrix multiplication is not commutative but associative. You will implement matrix operations and visualize transformations in Python. This is the foundation for understanding how deep networks transform information from input to output.

## Table of Contents
- What Is a Matrix?
- Matrices as Transformations
- Matrix-Vector Multiplication
- Matrix-Matrix Multiplication
- Special Matrices: Identity, Diagonal, Transpose
- Determinant and Invertibility
- Eigenvalues and Eigenvectors (Intuition)
- Matrices in Python with NumPy
- Practical Examples from Neural Networks
- Closing Reflection

---

## What Is a Matrix?

A matrix is a rectangular array of numbers arranged in rows and columns.

**Example:**
```
    [1   2   3]
A = [4   5   6]
    [7   8   9]
```

This is a 3×3 matrix (3 rows, 3 columns). We write it as **A** ∈ ℝ³×³.

### Why Matrices Matter
- **Data storage**: Images are matrices (height × width × 3 channels)
- **Transformations**: Matrices encode operations that change vectors
- **Neural networks**: Every layer is a matrix applied to a vector

---

## Matrices as Transformations

This is the key insight: **a matrix is a function that transforms one vector into another.**

### Example: Scaling Transformation

```
S = [2   0]
    [0   2]
```

This matrix scales vectors by a factor of 2. If you apply it to **v** = [1, 1]:

**S · v** = [2, 2]

Geometrically: the vector is twice as long.

### Example: Rotation Transformation

```
R = [cos(θ)   -sin(θ)]
    [sin(θ)    cos(θ)]
```

This matrix rotates vectors by angle θ. For θ = 90°:

```
R = [0   -1]
    [1    0]
```

Apply to **v** = [1, 0]: Result = [0, 1] (rotated 90° counterclockwise).

### Example: Shear Transformation

```
H = [1   1]
    [0   1]
```

This shifts the x-component based on the y-component. The space is "sheared."

---

## Matrix-Vector Multiplication

How do you multiply a matrix by a vector?

For **A** ∈ ℝᵐ×ⁿ and **v** ∈ ℝⁿ, the result **Av** ∈ ℝᵐ.

**Definition**: The i-th component of **Av** is the dot product of the i-th row of **A** with **v**.

### Example

```
      [1   2]   [3]       [1·3 + 2·4]       [11]
  A = [4   5] , v = [4] => [4·3 + 5·4]   =  [32]
      [7   8]              [7·3 + 8·4]      [53]
```

### Geometric Interpretation
- **A** encodes a transformation
- **v** is a vector in the input space
- **Av** is that vector transformed into the output space

In neural networks:
- **v** = input to a layer (e.g., word embedding)
- **A** = weight matrix learned during training
- **Av** = transformed representation sent to the next layer

---

## Matrix-Matrix Multiplication

Multiplying two matrices combines their transformations.

For **A** ∈ ℝᵖ×ʳ and **B** ∈ ℝʳ×ᵍ, the result **AB** ∈ ℝᵖ×ᵍ.

**Definition**: The (i, j) component of **AB** is the dot product of the i-th row of **A** with the j-th column of **B**.

### Example

```
    [1   2]   [5   6]   
AB= [3   4] · [7   8]   = ?

First component: 1*5 + 2*7 = 19
[1·5 + 2·7    1·6 + 2·8]   [19   22]
[3·5 + 4·7    3·6 + 4·8] = [43   50]
```

### Important: Non-Commutativity
**AB ≠ BA** in general. Matrix multiplication order matters. This reflects the geometric fact that applying transformation A then B is different from applying B then A.

### Important: Associativity
**(AB)C = A(BC)** even if the three matrices are not square. This matters for efficient neural network computation—you can group operations differently to minimize computation.

---

## Special Matrices

### Identity Matrix **I**
```
I = [1   0]
    [0   1]
```

**I · v = v** for any vector **v**. It's the "do nothing" transformation.

### Diagonal Matrix **D**
```
D = [2   0]
    [0   3]
```

Multiplying by **D** scales the first component by 2 and the second by 3. Efficient to compute.

### Transpose **Aᵀ**
Swapping rows and columns:

```
    [1   2   3]         [1   4]
A = [4   5   6]    =>   [2   5]
                        [3   6]
```

Used in: computing distances, attention mechanisms, solving systems of equations.

---

## Determinant and Invertibility

The **determinant** of a matrix is a single number that encodes important properties.

**For a 2×2 matrix:**

```
det(A) = ad - bc
         [a   b]
         [c   d]
```

### What Determinant Means
- **det(A) ≠ 0**: Matrix is invertible; transformation is reversible
- **det(A) = 0**: Matrix is singular; information is lost; transformation is not reversible
- **|det(A)|**: Factor by which the transformation scales volume/area

### In Neural Networks
If a layer has a determinant near zero, gradients will vanish during backpropagation, making training difficult. This is why careful initialization of weights matters.

---

## Eigenvalues and Eigenvectors (Intuition)

An **eigenvector** of a matrix **A** is a special vector **v** such that multiplying by **A** just scales it:

**A · v** = λ**v**

where λ (lambda) is the **eigenvalue**.

### Geometric Interpretation
- **v** is a direction that the transformation **A** doesn't rotate, only scale
- λ is the scaling factor
- If λ > 1, the transformation stretches in that direction
- If λ < 1, it shrinks
- If λ < 0, it reverses and scales

### Why It Matters
- **Principal Component Analysis (PCA)**: Eigenvectors point to directions of maximum variance
- **Stability**: Large eigenvalues can cause gradients to explode
- **Convergence**: The largest eigenvalue bounds how fast gradient descent converges

We won't compute eigenvalues from scratch here, but knowing the intuition helps you debug neural networks later.

---

## Matrices in Python with NumPy

```python
import numpy as np

# Create a matrix
A = np.array([[1, 2],
              [3, 4]])

# Create a vector
v = np.array([5, 6])

# Matrix-vector multiplication
result = np.dot(A, v)  # or A @ v

# Matrix-matrix multiplication
B = np.array([[7, 8],
              [9, 10]])
C = np.dot(A, B)  # or A @ B

# Transpose
A_T = A.T

# Determinant
det_A = np.linalg.det(A)

# Inverse
A_inv = np.linalg.inv(A)  # If det(A) ≠ 0

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
```

---

## Practical Examples from Neural Networks

### Example 1: Single Layer Forward Pass

```
x ∈ ℝ^d      Input vector (e.g., word embedding)
W ∈ ℝ^(h×d)  Weight matrix (learned)
b ∈ ℝ^h      Bias vector (learned)
h ∈ ℝ^h      Hidden layer output

h = W · x + b
```

The matrix **W** transforms the input from d dimensions to h dimensions.

### Example 2: Attention Mechanism

```
Q, K, V are all matrices (queries, keys, values)
scores = Q · K^T  (matrix of dot products)
```

The transpose is essential for getting the dimensions right.

### Example 3: Regularization

```
Total Loss = L(model) + λ · ||W||²
```

The norm ‖W‖² depends on the eigenvalues of W. If eigenvalues are large, the model is "stretched" in certain directions and may overfit.

---

## Closing Reflection: Abstract Algebra Is Concrete Geometry

Matrix algebra can seem abstract and mechanical when first learned—formulas to memorize, rules to follow. Everything changes when you see matrices geometrically.

A weight matrix in a neural network is not just a list of numbers. It is a transformation that stretches, rotates, and reshapes the representation of information as it flows through the network.

When you train MarkGPT (starting in Module 06), you will be learning billions of matrix coefficients. Each one adjusts a tiny piece of a global transformation that converts input tokens into predicted outputs. That is not magic—it is mathematics. But it is mathematics with real geometric meaning.

Hold onto that intuition. When debugging your model, when trying to understand why a training run failed, when optimizing for speed: come back to the geometry. It will guide you.

*Next: Module 01, Day 4 — Probability and Information Theory (Lesson L04.1)*
