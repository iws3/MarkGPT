# Lesson L03.1: Vectors and What They Really Mean
## Day 3: Mathematics You Already Know | Understanding the Foundation of Machine Learning

### Lesson Overview
You now understand AI history, what language models are, and how computers represent text. Today, we step into mathematics—but not the abstract, intimidating kind. This lesson is about *intuition*. A vector is not a mysterious symbol from a textbook; it is a way of describing location, direction, and magnitude in space. In deep learning, vectors are everywhere: they represent words, images, states of a neural network, attention scores. By the end of this lesson, you will have visual and geometric intuition for vectors: what addition means (combining movements), what dot product means (alignment), zero why magnitude matters (distance). You will implement vector operations from scratch in NumPy and see how they underlie everything MarkGPT does.

## Table of Contents
- What Is a Vector?
- Vectors as Arrows, Lists, and Movements
- Vector Operations: Addition and Subtraction
- Dot Product and Angles
- Vector Magnitude (Norm)
- Orthogonality and Linear Independence
- Vectors in Python with NumPy
- Intuition Before Formulas
- Practical Examples
- Closing Reflection

---

## What Is a Vector?

A vector is fundamentally simple: a list of numbers arranged in a particular order.

**Examples:**
- In 2D: **v** = [3, 4] represents a point or direction in 2D space
- In 3D: **v** = [1, 2, 3] represents a point or direction in 3D space
- In 768D: **v** = [0.2, -0.5, 0.1, ..., -0.3] could represent a word embedding

But numbers alone don't convey meaning. The power comes from the *interpretation* and the *operations* you can perform on vectors.

---

## Vectors as Arrows, Lists, and Movements

### Geometric View: Arrow in Space
Imagine a vector as an arrow starting at the origin (0, 0) and pointing to a location.

- **v** = [3, 4] draws an arrow 3 units right and 4 units up
- Length (magnitude) of this arrow: √(3² + 4²) = 5
- Direction: roughly northeast

### List View: Sequence of Numbers
The same vector as data:

- **v** = [3, 4]
- First component: 3
- Second component: 4

### Movement View: Displacement
The vector [3, 4] can represent:

- "Move 3 steps east, then 4 steps north"
- A sequence of instructions or transformations
- A pathway through space

All three views are the same mathematical object, just interpreted differently.

---

## Vector Operations: Addition and Subtraction

### Vector Addition
If **u** = [1, 2] and **v** = [3, 1], then **u + v** = [4, 3].

**Geometric interpretation**: "First follow **u**'s arrow, then follow **v**'s arrow. The result is the total displacement."

```python
import numpy as np

u = np.array([1, 2])
v = np.array([3, 1])
result = u + v  # [4, 3]
```

### Vector Subtraction
**u - v** = [1, 2] - [3, 1] = [-2, 1].

**Geometric interpretation**: "The difference vector points from the tip of **v** to the tip of **u**."

---

## Dot Product and Angles

The dot product is one of the most important operations in machine learning.

### Definition
For vectors **u** = [u₁, u₂, ...] and **v** = [v₁, v₂, ...]

**u · v** = u₁v₁ + u₂v₂ + ...

### Geometric Interpretation
**u · v** = ‖**u**‖ ‖**v**‖ cos(θ)

where θ is the angle between the vectors.

- If θ = 0° (same direction): cos(θ) = 1, dot product is maximum
- If θ = 90° (perpendicular): cos(θ) = 0, dot product is zero
- If θ = 180° (opposite): cos(θ) = -1, dot product is minimized

### In Machine Learning
The dot product measures **alignment** or **similarity**:

- Large positive dot product: vectors point in similar directions
- Zero dot product: vectors are orthogonal (independent)
- Large negative dot product: vectors point in opposite directions

In attention mechanisms (Module 06), queries and keys are compared using dot products to determine which words to focus on.

---

## Vector Magnitude (Norm)

The magnitude (or norm) of a vector is its length.

### L2 Norm (Euclidean)
‖**v**‖ = √(v₁² + v₂² + ...)

For **v** = [3, 4]: ‖**v**‖ = √(9 + 16) = 5

### L1 Norm (Manhattan)
‖**v**‖₁ = |v₁| + |v₂| + ...

For **v** = [3, 4]: ‖**v**‖₁ = 3 + 4 = 7

### Why Magnitude Matters
- **Normalization**: Dividing by magnitude creates unit vectors (length 1)
- **Distance**: The magnitude of **u - v** is the distance between points
- **Stability**: Normalized vectors prevent gradient explosions in neural networks

---

## Orthogonality and Linear Independence

Two vectors are **orthogonal** if their dot product is zero.

**Example**: **u** = [1, 0] and **v** = [0, 1]

**u · v** = 0 → orthogonal

### Why It Matters
- **Independence**: Orthogonal vectors are independent; neither can be expressed as a multiple of the other
- **Basis**: A set of orthogonal vectors can form a coordinate system (basis)
- **Efficiency**: In embeddings, orthogonal features carry non-redundant information

---

## Vectors in Python with NumPy

```python
import numpy as np

# Create vectors
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Addition
w = u + v  # [5, 7, 9]

# Subtraction
diff = u - v  # [-3, -3, -3]

# Dot product
dot_product = np.dot(u, v)  # 1*4 + 2*5 + 3*6 = 32

# Magnitude (L2 norm)
magnitude = np.linalg.norm(u)  # √(1 + 4 + 9) = √14 ≈ 3.74

# Normalized vector (unit vector)
unit_u = u / np.linalg.norm(u)

# Element-wise multiplication (not dot product)
element_wise = u * v  # [4, 10, 18]
```

---

## Intuition Before Formulas

Here is the key insight: *understand the geometry first, then learn the formulas.*

- **Magnitude**: How far is this point from the origin?
- **Dot product**: How aligned are these two vectors?
- **Orthogonal**: Do these vectors point in completely independent directions?

If you can answer these questions geometrically, the formulas follow naturally.

---

## Practical Examples

### Example 1: Word Embeddings as Vectors

Suppose we have two word embeddings:

- **"king"** = [0.2, 0.5, -0.3, ...]  (768-dimensional)
- **"queen"** = [0.19, 0.48, -0.32, ...]  (768-dimensional)

These vectors are close (high dot product), so the model will treat them as similar words.

### Example 2: Cosine Similarity

To compare word embeddings, we often use **cosine similarity**:

sim(**u**, **v**) = (**u · v**) / (‖**u**‖ ‖**v**‖)

This is always between -1 and 1 and measures the angle between vectors, regardless of length.

```python
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
```

---

## Closing Reflection: Geometry Is Your Friend

Many students find linear algebra intimidating because they are taught formulas without geometry. But vectors, matrices, and operations on them are *geometric*.

They have shapes. They have directions. They have distances. Learning to *see* them geometrically transforms algebra from mechanical symbol manipulation into intuitive spatial reasoning.

When you implement attention in Module 06, you will compute dot products between query and key vectors. When you improve embeddings in Module 08, you will measure similarity between vectors. All of this is geometric reasoning clothed in vector notation.

Build the geometric intuition now. The formulas will follow naturally.

*Next: Lesson L03.2 — Matrices as Transformations*
