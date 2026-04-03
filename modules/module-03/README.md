# Module 03 — Neural Networks from Scratch
## Days 13–18 | Intermediate

---

## Module Overview

This module builds your intuition for how neural networks learn. You will implement a neuron, then a multi-layer perceptron, then the magic of backpropagation — all from scratch, in plain Python and NumPy.

By the end of Module 03, you will:
- Understand the neuron as a mathematical object
- Implement forward and backward passes
- Solve non-linear problems (XOR) that n-gram models cannot
- Grasp why depth matters in neural networks

## Learning Objectives

- Understand core ML concepts
- Implement algorithms from scratch
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


## Neuron Architecture and Fundamentals

### The Biological Perspective

Artificial neurons are inspired by biological neurons:
- **Dendrites**: Receive signals (inputs)
- **Cell Body**: Process information (weighted sum)
- **Axon**: Send output signal
- **Synapse**: Connection strength (weights)

**Firing Mechanism**
Biological neuron fires when activation exceeds threshold.
Artificial neuron: Apply activation function to weighted sum.

### Mathematical Definition of a Neuron

**Linear Combination**
$$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$

Where:
- $x_i$: Input features
- $w_i$: Weights (synaptic strengths)
- $b$: Bias (threshold shift)
- $z$: Pre-activation (logit)

**Activation**
$$a = \sigma(z)$$

Where $\sigma$ is activation function (ReLU, sigmoid, tanh, etc.)

### Activation Functions Deep Dive

**Sigmoid Function**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- Output: (0, 1)
- Smooth gradient
- Vanishing gradient problem

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```

**Hyperbolic Tangent**
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
- Output: (-1, 1)
- Centered at zero
- Also has vanishing gradient issue

### ReLU and Modern Activations

**Rectified Linear Unit (ReLU)**
$$f(z) = \max(0, z)$$
- Advantages:
  - Simple computation
  - No vanishing gradient for positive values
  - Speeds up convergence
- Disadvantages:
  - Dying ReLU (zero output for negative inputs)
  - Not smooth at z=0

```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

**Leaky ReLU**
$$f(z) = \max(\alpha z, z)$$
- $\alpha$ small (0.01): Allows small negative gradient
- Prevents dying ReLU

**ELU (Exponential Linear Unit)**
$$f(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$$
- Closer to zero-mean outputs
- Smoother near origin

### Weight Initialization

**Zero Initialization (Bad!)**
- All neurons identical
- Cannot break symmetry
- Network remains linear

**Xavier/Glorot Initialization**
$$W \sim \text{Uniform}\left(-\sqrt{\frac{6}{n + m}}, \sqrt{\frac{6}{n + m}}\right)$$
- For sigmoid/tanh
- Maintains gradient magnitude

```python
fanin = prev_layer_size
fanout = current_layer_size
limit = np.sqrt(6 / (fanin + fanout))
W = np.random.uniform(-limit, limit, size=(fanin, fanout))
```

**He Initialization**
$$W \sim \text{Normal}\left(0, \sqrt{\frac{2}{n}}\right)$$
- For ReLU networks
- Works better with ReLU activation

## Perceptron Learning Rule

### Single Neuron Classification

**Perceptron**
- Simplest neural network unit
- Separates linearly separable data
- History: Rosenblatt (1958)

**Algorithm**
1. Initialize weights randomly
2. For each training example:
   - Compute prediction: $\hat{y} = \text{sign}(w \cdot x + b)$
   - If error: Update $w \leftarrow w + y \cdot x$
3. Repeat until convergence

```python
class Perceptron:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.w = None
        self.b = 0
    
    def predict(self, X):
        return np.sign(X @ self.w + self.b)
    
    def fit(self, X, y, epochs=100):
        self.w = np.zeros(X.shape[1])
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                pred = np.sign(xi @ self.w + self.b)
                if pred != yi:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
```

### Perceptron Limitations

**XOR Problem**
Single perceptron cannot solve XOR:
- XOR not linearly separable
- Minsky and Papert (1969) proved this
- Led to "AI Winter"

**Visualization**
```python
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([0, 1, 1, 0])  # XOR labels

# Single perceptron fails
percep = Perceptron()
percep.fit(X_xor, y_xor)
print(percep.predict(X_xor))  # Cannot get all correct
```

**Solution: Hidden Layers**
- Multi-layer perceptron (MLP) needed
- Hidden layers create non-linear decision boundaries

