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

