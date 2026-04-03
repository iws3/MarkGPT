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

## Multi-Layer Perceptron (MLP)

### Architecture

**Layer Composition**
- Input layer: Raw features
- Hidden layers: Learn representations
- Output layer: Final predictions

**Forward Pass Example (3-layer network)**
$$h^{(1)} = \sigma(W^{(1)} x + b^{(1)})$$
$$h^{(2)} = \sigma(W^{(2)} h^{(1)} + b^{(2)})$$
$$\hat{y} = f(W^{(3)} h^{(2)} + b^{(3)})$$

Where:
- $h^{(i)}$: Hidden layer activation
- $W^{(i)}$: Weight matrix for layer i
- $b^{(i)}$: Bias vector for layer i
- $\sigma$: Activation function (hidden)
- $f$: Output activation (sigmoid for binary, softmax for multi-class)

### Universal Approximation Theorem

**Key Theorem**
A feedforward network with single hidden layer can approximate:
- Any continuous function on compact domain
- With sufficient hidden units
- Using non-linear activation functions

**Implications**
- Hidden layers provide expressiveness
- One hidden layer theoretically sufficient
- In practice: Deeper networks generalize better
- Empirical evidence: Deep > shallow for many tasks

**Mathematical Intuition**
- Each hidden unit learns a feature
- Combinations create complex decision regions
- Non-linearity essential (linear layers compose to linear)

## Gradient Descent and Backpropagation

### Why Gradient Descent?

**Optimization Problem**
Minimize: $$L(W, b) = \frac{1}{m} \sum_{i=1}^m \ell(\hat{y}^{(i)}, y^{(i)})$$

Where:
- $L$: Total loss
- $\ell$: Loss per sample
- $m$: Number of training samples
- Dimensions: Thousands to billions of parameters
- Cannot solve analytically

**Gradient Direction**
$$\nabla L = \left[\frac{\partial L}{\partial w_1}, ..., \frac{\partial L}{\partial w_n}\right]$$
- Points toward steepest increase
- Move opposite direction to decrease loss

### Backpropagation Algorithm

**Chain Rule Foundation**
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \frac{\partial a}{\partial z} \frac{\partial z}{\partial w}$$

**Forward Pass**
- Compute all activations layer by layer
- Store intermediate values for backward pass

**Backward Pass**
1. Compute output layer error: $\delta^{(L)} = \nabla_a L \odot \sigma'(z^{(L)})$
2. Propagate backwards:
   $$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$
3. Compute gradients:
   $$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$
   $$\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$
4. Update parameters:
   $$W^{(l)} \leftarrow W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}$$

### Computational Complexity of Backprop

**Forward Pass Cost**
- Deep network: O(L × N²) where L=layers, N=units per layer
- Must compute all activations

**Backward Pass Cost**
- Similar to forward pass
- But includes transpose operations
- Typically 2x forward pass cost

**Efficiency**
```python
n_params = sum(layers)  # E.g., millions
forward_cost = O(n_params)
backward_cost = 2 * O(n_params)
total_per_step = 3 * O(n_params)

# 1000 training steps
total = 3000 * O(n_params)
```

**Tricks to Speed Up**
- Batch processing (vectorization)
- GPU acceleration
- Mixed precision (float32 instead of float64)

## Loss Functions for Different Tasks

### Mean Squared Error (MSE)

**For Regression**
$$\text{MSE} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2$$

**Gradient**
$$\frac{\partial \text{MSE}}{\partial \hat{y}} = -2(y - \hat{y})$$

**Properties**
- Convex (single global minimum)
- Larger errors penalized more
- Sensitive to outliers

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(y_true, y_pred):
    return -2 * (y_true - y_pred) / len(y_true)
```

### Cross-Entropy Loss

**For Classification**
$$\text{CE} = -\frac{1}{m} \sum_{i=1}^m \sum_{c=1}^C y_c^{(i)} \log(\hat{y}_c^{(i)})$$

Where:
- $C$: Number of classes
- $y_c^{(i)}$: True label (one-hot)
- $\hat{y}_c^{(i)}$: Predicted probability

**Binary Cross-Entropy**
$$\text{BCE} = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]$$

**Properties**
- Specific to probability outputs
- Penalizes confident mistakes heavily
- Natural for softmax output

```python
def cross_entropy_loss(y_true, y_pred, epsilon=1e-10):
    return -np.mean(y_true * np.log(y_pred + epsilon))
```

## Softmax and Output Layers

### Softmax Activation

**Definition**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Properties**
- Outputs sum to 1 (valid probability distribution)
- Differentiable everywhere
- Emphasizes maximum (winner-take-all)

```python
def softmax(z):
    # Numerical stability: subtract max
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)
```

**Numerical Stability**
- $e^z$ grows very fast
- Large $z$ values → overflow
- Solution: Subtract max from each row
- Mathematically equivalent, numerically stable

### Output Layer Design

**Binary Classification**
- Output: 1 neuron with sigmoid
- Loss: Binary cross-entropy
- Output interpretation: P(class=1)

**Multi-class Classification**
- Output: C neurons with softmax
- Loss: Cross-entropy
- Output interpretation: P(class=c) for each c

**Regression**
- Output: 1 or more neurons with linear
- Loss: MSE or MAE
- Output interpretation: Predicted value

**Multi-task Learning**
- Multiple output layers
- Each with appropriate activation/loss
- Shared hidden representations

## Forward Pass Implementation from Scratch

**Simple 3-Layer Network**
```python
class SimpleNN:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.lr = learning_rate
        self.params = {}
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            self.params[f'W{i+1}'] = np.random.randn(
                layer_sizes[i], layer_sizes[i+1]
            ) * 0.01
            self.params[f'b{i+1}'] = np.zeros((1, layer_sizes[i+1]))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, X):
        self.cache = {}
        A = X
        
        # Hidden layers
        for i in range(1, 3):
            Z = A @ self.params[f'W{i}'] + self.params[f'b{i}']
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i-1}'] = A
            A = self.relu(Z)
            self.cache[f'A{i}'] = A
        
        # Output layer
        Z3 = A @ self.params['W3'] + self.params['b3']
        self.cache['Z3'] = Z3
        self.cache['A2'] = A
        A3 = self.sigmoid(Z3)
        self.cache['A3'] = A3
        
        return A3
```

## Backward Pass Implementation from Scratch

**Computing Gradients**
```python
def backward(self, y):
    m = y.shape[0]
    grads = {}
    
    # Output layer error
    dA3 = self.cache['A3'] - y
    dZ3 = dA3  # Sigmoid derivative built into CE loss
    
    # Backprop through weights
    grads['W3'] = self.cache['A2'].T @ dZ3 / m
    grads['b3'] = np.sum(dZ3, axis=0) / m
    
    # Hidden layer 2
    dA2 = dZ3 @ self.params['W3'].T
    dZ2 = dA2 * (self.cache['Z2'] > 0)  # ReLU derivative
    
    grads['W2'] = self.cache['A1'].T @ dZ2 / m
    grads['b2'] = np.sum(dZ2, axis=0) / m
    
    # Hidden layer 1
    dA1 = dZ2 @ self.params['W2'].T
    dZ1 = dA1 * (self.cache['Z1'] > 0)  # ReLU derivative
    
    grads['W1'] = self.cache['A0'].T @ dZ1 / m
    grads['b1'] = np.sum(dZ1, axis=0) / m
    
    return grads

def update_params(self, grads):
    for key in self.params:
        self.params[key] -= self.lr * grads[key]
```

## Training Loop and Convergence

**Complete Training**
```python
def train(self, X, y, epochs=100, batch_size=32):
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = len(X) // batch_size
        
        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            
            X_batch = X[start:end]
            y_batch = y[start:end]
            
            # Forward pass
            output = self.forward(X_batch)
            
            # Compute loss
            loss = -np.mean(y_batch * np.log(output + 1e-10))
            epoch_loss += loss
            
            # Backward pass
            grads = self.backward(y_batch)
            
            # Update weights
            self.update_params(grads)
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return losses
```

## Debugging Neural Networks

### Gradient Checking

**Why Check?**
- Backprop has many operations
- Easy to make mistakes in implementation
- Off-by-one errors, transpose mistakes

**Numerical Gradient**
$$\frac{\partial f}{\partial w} \approx \frac{f(w + \epsilon) - f(w - \epsilon)}{2\epsilon}$$

**Verification**
```python
edsilon = 1e-5
numerical_grad = (loss(params + epsilon) - loss(params - epsilon)) / (2 * epsilon)
analytical_grad = compute_gradient(params)
diff = np.linalg.norm(numerical_grad - analytical_grad) / np.linalg.norm(numerical_grad + analytical_grad)
assert diff < 1e-7, 'Gradient check failed!'
```

### Common Issues and Fixes

**Issue: Loss doesn't decrease**
- Check 1: Learning rate (too high/low)
- Check 2: Gradient sign (should be negative)
- Check 3: Batch size effects

**Issue: Gradient explosion**
- Deep networks amplify gradients
- Solution: Gradient clipping
```python
clip_value = 1.0
for param in grads:
    grads[param] = np.clip(grads[param], -clip_value, clip_value)
```

**Issue: Gradient vanishing**
- Sigmoid derivative < 0.25
- Combines with chain rule → exponential decay
- Solution: ReLU activation, batch normalization

**Issue: Overfitting**
- Too many parameters
- Too many epochs
- Solution: Regularization, dropout, early stopping



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

## Multi-Layer Perceptron (MLP)

### Architecture

**Layer Composition**
- Input layer: Raw features
- Hidden layers: Learn representations
- Output layer: Final predictions

**Forward Pass Example (3-layer network)**
$$h^{(1)} = \sigma(W^{(1)} x + b^{(1)})$$
$$h^{(2)} = \sigma(W^{(2)} h^{(1)} + b^{(2)})$$
$$\hat{y} = f(W^{(3)} h^{(2)} + b^{(3)})$$

Where:
- $h^{(i)}$: Hidden layer activation
- $W^{(i)}$: Weight matrix for layer i
- $b^{(i)}$: Bias vector for layer i
- $\sigma$: Activation function (hidden)
- $f$: Output activation (sigmoid for binary, softmax for multi-class)

### Universal Approximation Theorem

**Key Theorem**
A feedforward network with single hidden layer can approximate:
- Any continuous function on compact domain
- With sufficient hidden units
- Using non-linear activation functions

**Implications**
- Hidden layers provide expressiveness
- One hidden layer theoretically sufficient
- In practice: Deeper networks generalize better
- Empirical evidence: Deep > shallow for many tasks

**Mathematical Intuition**
- Each hidden unit learns a feature
- Combinations create complex decision regions
- Non-linearity essential (linear layers compose to linear)

## Gradient Descent and Backpropagation

### Why Gradient Descent?

**Optimization Problem**
Minimize: $$L(W, b) = \frac{1}{m} \sum_{i=1}^m \ell(\hat{y}^{(i)}, y^{(i)})$$

Where:
- $L$: Total loss
- $\ell$: Loss per sample
- $m$: Number of training samples
- Dimensions: Thousands to billions of parameters
- Cannot solve analytically

**Gradient Direction**
$$\nabla L = \left[\frac{\partial L}{\partial w_1}, ..., \frac{\partial L}{\partial w_n}\right]$$
- Points toward steepest increase
- Move opposite direction to decrease loss

### Backpropagation Algorithm

**Chain Rule Foundation**
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \frac{\partial a}{\partial z} \frac{\partial z}{\partial w}$$

**Forward Pass**
- Compute all activations layer by layer
- Store intermediate values for backward pass

**Backward Pass**
1. Compute output layer error: $\delta^{(L)} = \nabla_a L \odot \sigma'(z^{(L)})$
2. Propagate backwards:
   $$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$
3. Compute gradients:
   $$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$
   $$\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$
4. Update parameters:
   $$W^{(l)} \leftarrow W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}$$

### Computational Complexity of Backprop

**Forward Pass Cost**
- Deep network: O(L × N²) where L=layers, N=units per layer
- Must compute all activations

**Backward Pass Cost**
- Similar to forward pass
- But includes transpose operations
- Typically 2x forward pass cost

**Efficiency**
```python
n_params = sum(layers)  # E.g., millions
forward_cost = O(n_params)
backward_cost = 2 * O(n_params)
total_per_step = 3 * O(n_params)

# 1000 training steps
total = 3000 * O(n_params)
```

**Tricks to Speed Up**
- Batch processing (vectorization)
- GPU acceleration
- Mixed precision (float32 instead of float64)

## Loss Functions for Different Tasks

### Mean Squared Error (MSE)

**For Regression**
$$\text{MSE} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2$$

**Gradient**
$$\frac{\partial \text{MSE}}{\partial \hat{y}} = -2(y - \hat{y})$$

**Properties**
- Convex (single global minimum)
- Larger errors penalized more
- Sensitive to outliers

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(y_true, y_pred):
    return -2 * (y_true - y_pred) / len(y_true)
```

### Cross-Entropy Loss

**For Classification**
$$\text{CE} = -\frac{1}{m} \sum_{i=1}^m \sum_{c=1}^C y_c^{(i)} \log(\hat{y}_c^{(i)})$$

Where:
- $C$: Number of classes
- $y_c^{(i)}$: True label (one-hot)
- $\hat{y}_c^{(i)}$: Predicted probability

**Binary Cross-Entropy**
$$\text{BCE} = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]$$

**Properties**
- Specific to probability outputs
- Penalizes confident mistakes heavily
- Natural for softmax output

```python
def cross_entropy_loss(y_true, y_pred, epsilon=1e-10):
    return -np.mean(y_true * np.log(y_pred + epsilon))
```

## Softmax and Output Layers

### Softmax Activation

**Definition**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Properties**
- Outputs sum to 1 (valid probability distribution)
- Differentiable everywhere
- Emphasizes maximum (winner-take-all)

```python
def softmax(z):
    # Numerical stability: subtract max
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)
```

**Numerical Stability**
- $e^z$ grows very fast
- Large $z$ values → overflow
- Solution: Subtract max from each row
- Mathematically equivalent, numerically stable

### Output Layer Design

**Binary Classification**
- Output: 1 neuron with sigmoid
- Loss: Binary cross-entropy
- Output interpretation: P(class=1)

**Multi-class Classification**
- Output: C neurons with softmax
- Loss: Cross-entropy
- Output interpretation: P(class=c) for each c

**Regression**
- Output: 1 or more neurons with linear
- Loss: MSE or MAE
- Output interpretation: Predicted value

**Multi-task Learning**
- Multiple output layers
- Each with appropriate activation/loss
- Shared hidden representations

## Forward Pass Implementation from Scratch

**Simple 3-Layer Network**
```python
class SimpleNN:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.lr = learning_rate
        self.params = {}
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            self.params[f'W{i+1}'] = np.random.randn(
                layer_sizes[i], layer_sizes[i+1]
            ) * 0.01
            self.params[f'b{i+1}'] = np.zeros((1, layer_sizes[i+1]))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, X):
        self.cache = {}
        A = X
        
        # Hidden layers
        for i in range(1, 3):
            Z = A @ self.params[f'W{i}'] + self.params[f'b{i}']
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i-1}'] = A
            A = self.relu(Z)
            self.cache[f'A{i}'] = A
        
        # Output layer
        Z3 = A @ self.params['W3'] + self.params['b3']
        self.cache['Z3'] = Z3
        self.cache['A2'] = A
        A3 = self.sigmoid(Z3)
        self.cache['A3'] = A3
        
        return A3
```

## Backward Pass Implementation from Scratch

**Computing Gradients**
```python
def backward(self, y):
    m = y.shape[0]
    grads = {}
    
    # Output layer error
    dA3 = self.cache['A3'] - y
    dZ3 = dA3  # Sigmoid derivative built into CE loss
    
    # Backprop through weights
    grads['W3'] = self.cache['A2'].T @ dZ3 / m
    grads['b3'] = np.sum(dZ3, axis=0) / m
    
    # Hidden layer 2
    dA2 = dZ3 @ self.params['W3'].T
    dZ2 = dA2 * (self.cache['Z2'] > 0)  # ReLU derivative
    
    grads['W2'] = self.cache['A1'].T @ dZ2 / m
    grads['b2'] = np.sum(dZ2, axis=0) / m
    
    # Hidden layer 1
    dA1 = dZ2 @ self.params['W2'].T
    dZ1 = dA1 * (self.cache['Z1'] > 0)  # ReLU derivative
    
    grads['W1'] = self.cache['A0'].T @ dZ1 / m
    grads['b1'] = np.sum(dZ1, axis=0) / m
    
    return grads

def update_params(self, grads):
    for key in self.params:
        self.params[key] -= self.lr * grads[key]
```

## Training Loop and Convergence

**Complete Training**
```python
def train(self, X, y, epochs=100, batch_size=32):
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = len(X) // batch_size
        
        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            
            X_batch = X[start:end]
            y_batch = y[start:end]
            
            # Forward pass
            output = self.forward(X_batch)
            
            # Compute loss
            loss = -np.mean(y_batch * np.log(output + 1e-10))
            epoch_loss += loss
            
            # Backward pass
            grads = self.backward(y_batch)
            
            # Update weights
            self.update_params(grads)
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return losses
```

## Debugging Neural Networks

### Gradient Checking

**Why Check?**
- Backprop has many operations
- Easy to make mistakes in implementation
- Off-by-one errors, transpose mistakes

**Numerical Gradient**
$$\frac{\partial f}{\partial w} \approx \frac{f(w + \epsilon) - f(w - \epsilon)}{2\epsilon}$$

**Verification**
```python
edsilon = 1e-5
numerical_grad = (loss(params + epsilon) - loss(params - epsilon)) / (2 * epsilon)
analytical_grad = compute_gradient(params)
diff = np.linalg.norm(numerical_grad - analytical_grad) / np.linalg.norm(numerical_grad + analytical_grad)
assert diff < 1e-7, 'Gradient check failed!'
```

### Common Issues and Fixes

**Issue: Loss doesn't decrease**
- Check 1: Learning rate (too high/low)
- Check 2: Gradient sign (should be negative)
- Check 3: Batch size effects

**Issue: Gradient explosion**
- Deep networks amplify gradients
- Solution: Gradient clipping
```python
clip_value = 1.0
for param in grads:
    grads[param] = np.clip(grads[param], -clip_value, clip_value)
```

**Issue: Gradient vanishing**
- Sigmoid derivative < 0.25
- Combines with chain rule → exponential decay
- Solution: ReLU activation, batch normalization

**Issue: Overfitting**
- Too many parameters
- Too many epochs
- Solution: Regularization, dropout, early stopping


## Convolutional Neural Networks (CNN)

### Motivation: Local Connectivity

**Problem with Dense Networks**
- Image with 224x224x3 pixels: 150K input features
- Dense hidden layer (1000 units): 150M weights
- Computationally expensive
- Doesn't leverage spatial structure

**CNN Solution**
- Local receptive fields (3x3, 5x5 kernels)
- Weight sharing (same filter across image)
- Sparse connectivity
- Orders of magnitude fewer parameters

### Convolution Operation

**Mathematical Definition**
$$\text{out}[i,j] = \sum_{a,b} \text{kernel}[a,b] \cdot \text{input}[i+a, j+b] + \text{bias}$$

**Kernel (Filter)**
- Small matrix (3x3, 5x5)
- Learnable weights
- Detects patterns (edges, corners, textures)

**Forward Pass Example**
```python
def convolve_2d(input_img, kernel, stride=1, padding=0):
    # Zero-pad if needed
    if padding > 0:
        input_img = np.pad(input_img, padding)
    
    h, w = input_img.shape
    k_h, k_w = kernel.shape
    
    out_h = (h - k_h) // stride + 1
    out_w = (w - k_w) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            window = input_img[i*stride:i*stride+k_h, 
                              j*stride:j*stride+k_w]
            output[i, j] = np.sum(window * kernel)
    
    return output
```

### Pooling Layers

**Max Pooling**
- Select maximum value in window
- Typical window: 2x2 or 3x3
- Provides translation invariance

```python
def max_pool_2d(input_img, pool_size=2, stride=2):
    h, w = input_img.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            window = input_img[i*stride:i*stride+pool_size,
                              j*stride:j*stride+pool_size]
            output[i, j] = np.max(window)
    
    return output
```

**Average Pooling**
- Mean instead of max
- Smoother downsampling
- Less common than max pooling

**Benefits**
- Reduces spatial dimensions
- Decreases computation
- Makes features robust to small shifts

### CNN Feature Extraction

**Layer Progression**
- Early layers: Low-level features (edges, colors)
- Middle layers: Mid-level features (shapes, textures)
- Deep layers: High-level features (objects, scenes)

**Visualization**
```python
# Early layer filters (edge detection)
fig, axes = plt.subplots(2, 3)
for i, ax in enumerate(axes.flat):
    ax.imshow(weights[:, :, 0, i], cmap='gray')
    ax.set_title(f'Filter {i}')
plt.show()
```

**Receptive Field Growth**
- Layer 1: 3x3 field
- Layer 2: 5x5 effective field
- Layer 3: 7x7 effective field
- Grows with depth

## Recurrent Neural Networks (RNN)

### Sequence Modeling Problem

**Why RNNs?**
- Sequential data: Text, audio, time series
- Variable length inputs
- Temporal dependencies
- Context from previous steps

**Standard NN Problem**
- Fixed input/output size
- No way to process sequences
- Loses temporal information

**RNN Solution**
- Recurrent connections
- Maintain hidden state
- Process one step at a time
- State carries information forward

### RNN Architecture

**Unfolding in Time**
$$h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

Where:
- $x_t$: Input at time t
- $h_t$: Hidden state at time t
- $y_t$: Output at time t
- $W_{hh}$: Recurrent weights
- $W_{xh}$: Input weights
- $W_{hy}$: Output weights

**Example: Text Processing**
```python
class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
    
    def forward(self, x_sequence):
        h = np.zeros((1, self.Whh.shape[0]))
        outputs = []
        
        for t in range(len(x_sequence)):
            h = np.tanh(x_sequence[t] @ self.Wxh + h @ self.Whh + self.bh)
            outputs.append(h)
        
        return outputs
```

### Vanishing Gradient in RNNs

**The Problem**
$$\frac{\partial h_t}{\partial h_0} = \prod_{i=0}^{t-1} \frac{\partial h_{i+1}}{\partial h_i}$$

When $\frac{\partial h_{i+1}}{\partial h_i} < 1$ (tanh derivative ≤ 1):
$$\frac{\partial h_t}{\partial h_0} \approx 0.9^t$$

For t=100: $0.9^{100} \approx 0.0000027$ (essentially zero)

**Consequences**
- Early layers don't learn
- Long-range dependencies lost
- Network only learns short-term patterns

**Solutions**
1. Better initialization (identity matrix for Whh)
2. Better activation (ReLU instead of tanh)
3. LSTM/GRU cells (explicit gates)
4. Gradient clipping

### Long Short-Term Memory (LSTM)

**The Cell State**
LSTM has explicit memory (cell state $C_t$):
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Where:
- $f_t$: Forget gate (what to discard)
- $i_t$: Input gate (what to add)
- $\tilde{C}_t$: Candidate values
- $\odot$: Element-wise multiplication

**Gate Equations**
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

**Key Insight**
- Cell state has straight connections (highway)
- Gradient flows without multiplication
- Avoids vanishing gradient

### Gated Recurrent Unit (GRU)

**Simplifed LSTM**
- 2 gates instead of 3
- No separate cell state
- Similar performance, fewer params

**Gate Equations**
$$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)  \text{ (reset gate)}$$
$$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)  \text{ (update gate)}$$
$$\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Comparison to LSTM**
- Fewer parameters (20% less)
- Faster computation
- Similar accuracy on most tasks
- Easier to understand

## Batch Normalization

### Internal Covariate Shift

**Problem**
- Parameter updates change input distribution to next layer
- Each layer must adapt to new distribution
- Slows learning
- Requires lower learning rate

**Visualization**
```python
# Without batch norm: Distributions shift
for epoch in range(100):
    # Layer 1 param update
    W1 -= alpha * grad_W1
    
    # Now layer 2 sees different input distribution!
    # Must retrain
    output = layer2(layer1(X))
```

### Batch Normalization Algorithm

**Training Time**
1. Compute mean: $\mu_B = \frac{1}{m} \sum_i x_i$
2. Compute variance: $\sigma_B^2 = \frac{1}{m} \sum_i (x_i - \mu_B)^2$
3. Normalize: $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
4. Scale and shift: $y_i = \gamma \hat{x}_i + \beta$

Where:
- $\gamma$: Learnable scale parameter
- $\beta$: Learnable offset parameter
- $\epsilon$: Small value for numerical stability

**Test Time**
- Use running statistics (moving average)
- Not per-mini-batch statistics
- Accumulated during training

```python
def batch_norm(x, gamma, beta, momentum=0.9, epsilon=1e-5):
    # Training
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    
    # Update running statistics
    running_mean = momentum * running_mean + (1 - momentum) * mean
    running_var = momentum * running_var + (1 - momentum) * var
    
    # Normalize
    x_norm = (x - mean) / np.sqrt(var + epsilon)
    
    # Scale and shift
    return gamma * x_norm + beta
```

### Benefits of Batch Normalization

**Advantages**
1. Allows higher learning rates
2. Reduces sensitivity to weight initialization
3. Acts as regularizer (slight noise from batch statistics)
4. Accelerates training
5. Can allow deeper networks

**Empirical Results**
- Training time reduced by 10-50%
- Final accuracy often improves by 1-5%
- More stable training

**When to Use**
- Deep networks (>10 layers)
- When convergence is slow
- GPU memory available (batch size important)
- Before ReLU or other activations

## Dropout Regularization

### How Dropout Works

**Training**
1. Randomly drop units with probability p
2. Forward pass with subset of units
3. Backpropagation only through active units
4. Different units dropped each iteration

**Test Time**
- Use all units
- Scale by (1-p) to match training


**Mathematical Intuition**
- Training ensemble of thinned networks
- Each presents obstacle to co-adaptation
- Final prediction: weighted average of subnetworks

```python
def dropout(x, p=0.5, training=True):
    if not training:
        return x
    
    # Create mask
    mask = np.random.binomial(1, 1-p, x.shape)
    
    # Apply mask and scale
    return x * mask / (1 - p)
```

### Regularization Techniques Comparison

**L1/L2 Regularization**
- Penalizes weight magnitude
- Simple but limited
- Works with any model

**Dropout**
- Kills neurons probabilistically
- Very effective for overfitting
- Efficient (slight computational cost)
- Works for any architecture

**Early Stopping**
- Monitor validation loss
- Stop when no improvement
- Simple, always works
- Requires separate validation set

**Data Augmentation**
- Increase dataset diversity
- Domain-specific
- Very effective
- Requires domain knowledge

**Best Practice**
- Combine multiple techniques
- Use validation set to monitor
- Adjust based on overfitting degree

## Learning Rate Schedules

### Fixed vs Adaptive Learning Rate

**Fixed Learning Rate**
- Simple: $\alpha = 0.001$
- Problem: Too high early, too low late
- May oscillate or not converge

**Decay Schedule**
$$\alpha_t = \alpha_0 \cdot (1 + \text{decay\_rate} \cdot t)^{-1}$$

**Step Decay**
$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / \text{steps} \rfloor}$$

Example: Reduce by 0.5 every 10 epochs

**Exponential Decay**
$$\alpha_t = \alpha_0 \cdot e^{-\text{decay\_rate} \cdot t}$$

```python
def lr_schedule(epoch, initial_lr=0.1):
    return initial_lr * (0.5 ** (epoch // 10))
```

### Adaptive Methods: Adam, RMSprop

**Momentum**
$$v_t = \beta v_{t-1} + (1 - \beta) \nabla L$$
$$\theta_t = \theta_{t-1} - \alpha v_t$$

- Accumulates gradient direction
- Accelerates convergence
- Dampens oscillations

**RMSprop**
$$m_t = \beta m_{t-1} + (1 - \beta) (\nabla L)^2$$
$$\theta_t = \theta_{t-1} - \alpha \frac{\nabla L}{\sqrt{m_t} + \epsilon}$$

- Divides by RMS of gradient
- Per-parameter learning rate
- Handles sparse gradients

**Adam (Adaptive Moment Estimation)**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L)^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

- Combines momentum and RMSprop
- Default β₁=0.9, β₂=0.999
- Robust across different problems
- Most used in practice

## Implementation: Complete Neural Network from Scratch

**Full Training Pipeline**
```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Xavier initialization
        limit = np.sqrt(6 / (input_size + hidden_size))
        self.W1 = np.random.uniform(-limit, limit, 
                                   (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        
        limit = np.sqrt(6 / (hidden_size + output_size))
        self.W2 = np.random.uniform(-limit, limit,
                                   (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)  # Hidden layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))  # Sigmoid
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Output layer
        dz2 = output - y
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0) / m
        
        # Hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - self.a1**2)  # tanh derivative
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0) / m
        
        return dW1, db1, dW2, db2
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            # Forward
            output = self.forward(X)
            
            # Loss
            loss = -np.mean(y * np.log(output) + 
                           (1-y) * np.log(1-output))
            
            # Backward
            dW1, db1, dW2, db2 = self.backward(X, y, output)
            
            # Update
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
    
    def predict(self, X):
        return self.forward(X) > 0.5
```


## ResNets and Skip Connections

### Why Skip Connections?

One of the key innovations: Residual networks allow very deep training.
Skip connections enable gradients to flow directly through layers.

$$y = F(x) + x$$

Benefits: Enables 152+ layer networks, better generalization.
## Convolutional Neural Networks - Advanced

### Depthwise Separable Convolution

Idea: Split convolution into spatial and channel dimensions.
- Depthwise: Apply per channel (h x w x 1)
- Pointwise: 1x1 conv to mix channels
- Result: 8-9x fewer parameters
- Used in MobileNet for efficiency

## Attention Mechanisms - Deep Dive

### Self-Attention

Query, Key, Value framework enables comparing all positions.
Each position attends to all other positions simultaneously.
Allows modeling long-range dependencies efficiently.

Mathematical foundation for transformers.
### Cross-Attention

Query from one source, Key/Value from another.
Connects encoder output to decoder.
Essential for sequence-to-sequence models.
Enables knowledge transfer between modalities.

## Transformer Architecture

### Encoder-Decoder Design

Encoder processes input without causal masking.
Decoder generates output with causal constraints.
Cross-attention connects both modules.
State-of-the-art foundation for NLP tasks.

### Vision Transformers

Apply transformer to images by treating as sequences.
Patch embeddings: Divide image into 16x16 patches.
Linear projection: Each patch to embedding.
Then apply standard transformer.
Competitive with CNNs on image classification.

## MarkGPT Architecture

### Model Scaling Laws

Empirically observed relationships:
- Loss decreases as O(N^-a) where N = model size
- a typically around 0.07-0.1
- Larger models learn faster
- Compute optimal: Not always biggest model

MarkGPT trained on 10T+ tokens to study scaling.
### Efficient Attention

Standard attention: O(N^2) complexity
Problematic for long sequences (N=4096)

Optimizations:
- Flash attention: Efficient CUDA kernels
- Grouped query: Share KV heads
- Sliding window: Fixed context size
- Linear attention: Kernel-based methods

## Recurrent vs Attention Trade-offs

### RNN Advantages
- Inherently handles sequences
- State size independent of sequence length
- Can process streaming data

### Attention Advantages
- Parallelizable (transformers)
- Better long-range modeling
- More expressiveness
- Interpretable (visualize weights)

Modern trend: Transformers dominate due to scale.
## Training Large Models

### Data Parallelism

Replicate model on N GPUs.
Each GPU processes different batch.
Synchronize gradients after backward pass.
Linear speedup (approximately).
Used for MarkGPT training.

### Model Parallelism

Split model layers across GPUs.
MarkGPT-Large: 32 layers on 8 GPUs.
Pipeline parallelism: Overlap computation.
Enables training larger models.
But introduces pipeline bubbles.

## Evaluation Metrics Deep Dive

### Perplexity

Average branching factor of next token.
Lower is better (model less confused).
Perplexity 10: Like choosing from 10 equally likely tokens.
MarkGPT achieves perplexity of 8-15 depending on domain.

### Human Evaluation Protocol

Standard procedure for language models:
1. Sample 100-500 examples
2. Multiple raters (3-5) per example
3. Blind evaluation (no model info)
4. Compute inter-rater agreement
5. Report mean and confidence intervals

Dimensions: Fluency, relevance, factuality, completeness.
## Prompt Engineering

### Few-Shot Learning

Providing examples to steer model behavior.
4 examples often as good as fine-tuning.
Order and quality of examples matter.
Can dramatically improve performance.

Example:
Task: Sentiment -> "+1: positive, 0: negative"
Examples given in prompt improve accuracy.
### Chain-of-Thought

Ask model to reason step-by-step.
Improves accuracy on math problems by 40%+.
Works even without examples (zero-shot).
Forces intermediate reasoning.
Increases inference time but improves quality.

## Generation Strategies

### Beam Search

Keep top-K hypotheses at each step.
K=1: Greedy (fast, mediocre)
K=5: Balance speed and quality
K=100: Slow but better quality
Finds better solutions than greedy.

### Sampling vs Greedy

Greedy: Always pick most likely next token.
Deterministic, boring, sometimes repetitive.

Top-K sampling: Sample from K most likely.
More diverse, sometimes nonsensical.

Temperature: Control randomness.
0: Greedy, 1: Original, >1: Very random.
MarkGPT uses T=0.8 for balance.

## Sentiment Classification: Case Study

### Dataset and Setup

50K movie reviews (binary classification)
Training: 40K, Validation: 5K, Test: 5K
Average length: 250 tokens
Class balanced (25K pos, 25K neg)

### Model and Results

Fine-tuned MarkGPT-Nano:
- Accuracy: 91% (vs LSTM baseline: 87%)
- Training time: 2 hours on single GPU
- Inference: 150 samples/sec on CPU

## Machine Translation: Case Study

### English-Spanish Translation

Dataset: 500K parallel sentences
Vocabulary: 50K both languages
Architecture: Encoder-Decoder Transformer

### Metrics

BLEU Score: 35.2 (competitive)
Human evaluation: 4.1/5 (good quality)
Inference speed: 80 tokens/sec
Outperforms SMT baseline by 15 BLEU points.

## Question Answering: Case Study

### SQuAD Dataset

100K questions on Wikipedia passages
Task: Extract answer span from context
Generative approach (generate answer text)

### Results

Exact Match: 78%
F1 Score: 85%
Inference: 200 questions/sec
10x more efficient than BERT-large.

## Overfitting Prevention

### Regularization Techniques

L1/L2 regularization: Penalize large weights
Dropout: Randomly drop neurons
Early stopping: Stop when val loss plateaus
Data augmentation: Expand training set
Batch normalization: Stabilize training

### Diagnosis

Train loss ≈ val loss: Good generalization
Train loss << val loss: Overfitting
Train loss >> val loss: Underfitting

## Debugging Neural Networks

### Common Issues

Loss diverges: Learning rate too high
Loss plateaus: Learning rate too low
NaN values: Numerical overflow
Gradient vanishing: Deep networks need fixes
Slow convergence: Bad initialization

### Systematic Approach

1. Check loss on tiny batch (N=32)
2. Verify data loading
3. Gradient checking (numerical vs analytical)
4. Monitor activation statistics
5. Visualize learned features

## Interpretability Techniques

### Attention Visualization

Heatmap: What each token attends to.
Early layers: Syntactic patterns
Late layers: Semantic relationships
Not always meaningful (some noise)

### Probing Tasks

Train binary classifier on hidden states
Task: Predict grammatical property
If classifier succeeds: Representation encodes information
Reveals what model learns at each layer

## Knowledge Distillation

### Idea

Large teacher model (7B params)
Small student model (100M params)
Transfer knowledge via soft targets

### Process

1. Train teacher normally
2. Get soft probabilities on unlabeled data
3. Student learns to match teacher
4. Use temperature to soften distributions
5. Combine with task loss

### Results

100M student with distillation: 90% of teacher
Without distillation: 75% of teacher
Dramatic improvement for small models.

## Model Quantization

### Motivation

7B model in FP32: 28 GB
7B model in INT8: 7 GB (4x smaller)
7B model in INT4: 3.5 GB (fits on GPU!)

### Methods

Post-training: Quantize after training
Quantization-aware: Simulate during training
INT8: ~5% accuracy drop
INT4: ~15% drop with techniques

### Impact

Faster inference on limited hardware
Reduced bandwidth requirements
Trade-off between size and quality

## Distributed Training

### Data Parallelism

Multiple GPUs, same model
Each GPU different batch
Synchronize gradients
Linear speedup (mostly)

### Pipeline Parallelism

Different layers on different GPUs
Fill pipeline with multiple batches
Reduces GPU idle time
Pipeline bubbles at boundaries

### Tensor Parallelism

Split single layer across GPUs
Highest communication overhead
Used for largest models (1T+ parameters)

## Appendix A: Activation Functions

Sigmoid: (0,1), vanishing gradient
Tanh: (-1,1), better centered
ReLU: [0,∞), no vanishing, simple
Leaky ReLU: No dying neurons
ELU: Smooth, better near zero
GELU: Smooth approximation of ReLU
Swish: Learned by AutoML, very effective

## Appendix B: Loss Functions

MSE: For regression, simple
MAE: Robust to outliers
Cross-entropy: Classification, natural choice
Focal loss: For imbalanced data
Contrastive loss: For similarity learning
Triplet loss: For metric learning
Info-NCE: For self-supervised learning

## Appendix C: Optimization Algorithms

SGD: Simple, effective baseline
SGD + Momentum: Accelerated convergence
AdaGrad: Per-parameter learning rate
RMSprop: Exponential moving average
Adam: Momentum + RMSprop (most used)
AdamW: Adam with weight decay
LAMB: For large batch training

Recommendation: Adam default, AdamW for fine-tuning.
## Module 03 Completion Summary

**Topics Covered**
- Neurons and MLPs
- Backpropagation algorithm
- CNNs and RNNs
- LSTMs and GRUs
- Attention mechanisms
- Transformers
- MarkGPT architecture
- Training techniques
- Evaluation metrics
- Debugging and interpretation
- Model compression
- Distributed training

**Skills Developed**
- Understand neural networks deeply
- Implement from scratch
- Train large models
- Debug effectively
- Interpret models
- Deploy efficiently

## Fine-tuning Large Language Models

### Transfer Learning Context

Pre-training: Learn general language on 10T tokens
Fine-tuning: Adapt to specific task on 10K tokens
Parameter efficiency: Only 0.1% of parameters as trainable
Result: State-of-the-art task performance

### Full Fine-tuning

Update all model parameters.
High quality but computationally expensive.
MarkGPT-Nano: 12 hours on 1 GPU for SQuAD.
Requires 30GB+ GPU for large models.
Not practical for most practitioners.

### Adapter Modules

Original: model size 7B
Adapter: 0.01B (0.15% parameters)
Add small bottleneck between layers.
Down-project to 64d, process, up-project back.
Achieves 99% of fine-tuning performance.
Only 10MB per task (stores cheaply).

### Prompt Tuning

Learn virtual prompt tokens (32 tokens).
Learnable parameters: 32 * 4096 ≈ 130K
Incredibly efficient (65x better than adapters).
Task performance: 98% of full fine-tuning.
Challenge: Less effective for very different tasks.

## Low-Rank Adaptation (LoRA)

### Motivation

Hypothesis: Weight updates are low-rank.
Instead of ΔW (huge matrix):
Use A @ B where A and B are small.
MarkGPT-7B LoRA: 4M params (0.06% of model).

### Implementation Details

For each weight matrix W:
Output = Wx + αABx
A: N x r (initialized N(0, std)
B: r x d (initialized zeros)
r: Rank (typically 8-64)
α: Scaling factor (16-32)

Benefits:
- Tiny parameters (stack modules)
- Merge with original weights (no inference overhead)
- Compatible with quantization

### LoRA Results

MarkGPT-7B with LoRA on GLUE:
- Training time: 1 hour vs 8 hours full
- Memory: 4GB vs 24GB
- Final accuracy: 87.2% vs 87.6% (0.4% gap)
- Inference: Identical speed (weights merged)

Scales to LoRA composition (multiple adapters).
## Interpretability Methods

### Attention Visualization

Heatmap: Query-key interactions.
Early layers: Attend to neighboring tokens.
Late layers: Semantic grouping.
Multi-head: Different attention patterns.

Insight: Models learn to structure information.
### Gradient-based Attribution

Input gradient ∂L/∂x shows feature importance.
Integrated gradients: Baseline-based method.
SmoothGrad: Average over noise samples.
DeepLIFT: Backpropagation with reference points.

Use: Explain why model made prediction.
### SHAP Values

Game theory approach to feature importance.
Contribution of each feature to prediction.
SHAP = Shapley value (from cooperative game theory).
Fair allocation of credit among features.
Computational cost: Expensive for large models.

## Knowledge Distillation - Advanced

### Conventional Distillation

Student learns from soft targets.
Temperature = 3-5 softens distribution.
Combine two losses:
- Task loss: Hard labels
- Distillation loss: Teacher soft targets
Ratio: 0.1 task + 0.9 distillation

### Feature-based Distillation

Match intermediate representations.
Compare hidden layers directly.
Better for compression (larger models).
Attention maps matching:
Loss = ||StudentAttention - TeacherAttention||^2

### Multi-Teacher Distillation

Combine knowledge from multiple teachers.
Diverse models capture different patterns.
Weight teacher outputs by confidence.
Significantly better student than single teacher.

## Model Quantization - Deep Dive

### Symmetric Quantization

Map [min, max] to [Qmin, Qmax]
INT8: [-128, 127] typically
Scale factor s = max(|min|, |max|) / 128
Q(x) = round(x / s)
Reverse: x' = Q(x) * s

### Asymmetric Quantization

Map [min, max] to [0, 255] for INT8
Scale: (max - min) / 255
Zero-point offset: min value
Better for skewed distributions.
3% better accuracy on typical weights.

### Quantization-Aware Training

Simulate quantization during training.
Gradients flow through fake quantization.
Model learns robust representations.
INT4 with QAT: 10% drop vs 25% without.
Training cost: 2x but worth the improvement.

### Calibration and Clipping

Min/max statistics from calibration set.
Method 1: Use actual min/max
Method 2: Percentile (0.1%, 99.9%)
Method 3: KL divergence alignment
Higher clip = info loss, lower = outlier issue.

## Advanced Training Techniques

### Mixed Precision Training

Master weights: FP32 (for stability)
Computation: FP16 (for speed)
Gradient scaling: Multiply by 2^16
Result: 2-3x speedup, minimal accuracy loss.
Essential for large models (7B+).

### Gradient Checkpointing

Re-compute activations instead of storing.
Memory: O(√N) instead of O(N)
Trade-off: Double computation, half memory.
Enable for largest models (13B+).
Slow but fits on single GPU.

### Gradient Accumulation

Effective batch = real batch * accumulation
Process N real batches, accumulate gradients.
Update after N steps.
Simulate larger batch on limited GPU.
Helps with batch-size optimization.

## Mixture of Experts

### Architecture

Multiple expert networks (12 experts typical).
Router network: Selects top-K experts.
Each token routed independently.
Parameters: Total = dense + routing overhead
12 experts * 300M params = 3.6B params
But activate only 300M per token (efficient).

### Sparse Activation

Only K experts active per token (K=2 typical).
Compute increases slowly with model size.
GShard: 600B sparse model on 16 TPUs.
Switch transformers: Using single expert (K=1).
Much more efficient than dense models.

### Load Balancing

Challenge: Some experts used more.
Loss term: Penalize imbalanced routing.
Auxiliary loss: 0.01 * (balance_loss)
Expert utilization target: Equal distribution.
Important for training stability.

## Inference Optimization

### KV-Cache

Store computed K, V from previous steps.
Attention(Q_new, K_cache, V_cache).
Avoids recomputing past positions.
Reduce compute from O(N^2) per token to O(N).
7B model: 30GB cache for batch=32, length=2048.

### Continuous Batching

Traditional: Wait for all sequences to finish.
Continuous: Remove finished, add new in-flight.
GPU utilization: 90%+ vs 60%.
Throughput increases but latency varies.
Critical for serving systems.

### Speculative Decoding

Small model generates candidate tokens.
Large model verifies (parallel).
Accept multiple tokens per forward pass.
2-3x speedup for large models.
Same outputs as standard decoding.

## Model Merging

### Simple Averaging

Merge weights: W = αW1 + (1-α)W2
Two fine-tuned models on alpha/beta.
Surprisingly effective (task ensemble).
Creates model good at both tasks.

### Task Vector Approach

Task vector = fine-tuned_weights - base_weights
Merge: base + α*task1 + β*task2
More interpretable than direct merge.
Better capacity allocation.
Can scale task contributions.

## Retrieval Augmented Generation

### Motivation

LLM memorizes training data.
Not effective for retrieval of facts.
RAG: Retrieve relevant docs, then generate.
Combines retriever + reader model.

### Architecture

1. Query embedding: Dense encoder
2. Retriever: Top-K passages from index
3. Reader: Generate answer conditioned on passages
Index: Dense passage retrieval (DPR)
Reader: BART or T5

### Results

SQuAD with RAG: 92% F1 vs 85% without
Natural questions: 60% vs 45% without
Reduces hallucination significantly.
Enables fact-checking (cite sources).

## Continual Learning

### Catastrophic Forgetting

Model trained on task A then task B.
Performance on A drops to 10%.
Weights optimized away from task A.
Major challenge for sequential learning.

### Elastic Weight Consolidation

Compute importance of each parameter.
Fisher information matrix: which params matter.
Penalize changing important parameters.
Loss = task_loss + λ * Σ F_i * (θ_i - θ_old)^2
Achieves 80%+ on both tasks.

### Parameter Isolation

Different tasks use different parameters.
Adapter modules (low-rank).
Sparse masks: Select parameters per task.
Complete isolation: No interference.
But requires more storage.

## Domain Adaptation

### Distribution Shift Problem

Train: General text (Wikipedia, CommonCrawl)
Test: Medical documents (MedBench)
Performance drops significantly.
Vocabulary mismatch, style differences.

### Domain-Adaptive Pre-training

Continued pre-training on domain data.
DAPT: Additional 10K steps on medical texts.
Task-adaptive pre-training: Fine-tune task data.
TAPT: 100 steps on task training set.
Combined: 5-10% improvement on downstream tasks.

## Multilingual Models

### Challenges

100+ languages, different scripts.
Vocabulary: mBERT has 110K tokens!
Resource imbalance: English 1000x more data.
Zero-shot cross-lingual transfer.

### Cross-lingual Transfer

Train on English, test on Hindi.
mBERT trained on 104 languages jointly.
33 language pairs show >80% transfer.
Magic: Shared representation space.
Enabled by shared tokenizer (WordPiece).

## Code Generation

### Why Hard

Syntax is strict (wrong bracket = error).
Long-range dependencies (matching braces).
Variable naming conventions.
Semantic correctness harder to define.

### CodeBERT/GraphCodeBERT

CodeBERT: Pre-train on code-documentation pairs.
GraphCodeBERT: Use data flow graph.
Results: Code-to-code search, clone detection.
Foundation for code understanding.

## Evaluation Benchmarks

### Standard Benchmarks

GLUE: 9 language understanding tasks
SQuAD: Reading comprehension (100K examples)
BLEU: Machine translation (automatic metric)
ROUGE: Summarization (lexical overlap)
Human evaluation: Always gold standard

### Emergent Abilities

Small models: Can't do in-context learning
Medium models: Few-shot breaks through
Large models: Chain-of-thought emerges
Scaling laws predict when they appear
Example: 62B model = 1000x better at math

## Appendix D: Hyperparameter Ranges

Learning rate: 1e-5 to 1e-3
Batch size: 8 to 512
Warmup: 0.1 to 0.3 of training
Weight decay: 0.0 to 0.1
Dropout: 0.0 to 0.3
Attention heads: 8 to 32
Hidden size: 256 to 4096

## Appendix E: Architecture Reference

BERT: 12 layers, 768 hidden, 12 heads
RoBERTa: Same but better pre-training
T5: Encoder-decoder, 12 layers each
GPT-2: 1.5B, 12 layers, 1024 hidden
GPT-3: 175B, 96 layers, 12288 hidden
MarkGPT-7B: 32 layers, 4096 hidden, 32 heads

