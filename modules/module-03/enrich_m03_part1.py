#!/usr/bin/env python3
"""
Module-03 enrichment part 1 - 80 commits
Neural network fundamentals and core concepts
"""
import subprocess
import os

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-03')

sections = [
    ("\n\n## Neuron Architecture and Fundamentals\n\n"
     "### The Biological Perspective\n\n"
     "Artificial neurons are inspired by biological neurons:\n"
     "- **Dendrites**: Receive signals (inputs)\n"
     "- **Cell Body**: Process information (weighted sum)\n"
     "- **Axon**: Send output signal\n"
     "- **Synapse**: Connection strength (weights)\n\n"
     "**Firing Mechanism**\n"
     "Biological neuron fires when activation exceeds threshold.\n"
     "Artificial neuron: Apply activation function to weighted sum.\n\n",
     "Add neuron biological inspiration"),
    
    ("### Mathematical Definition of a Neuron\n\n"
     "**Linear Combination**\n"
     "$$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$\n\n"
     "Where:\n"
     "- $x_i$: Input features\n"
     "- $w_i$: Weights (synaptic strengths)\n"
     "- $b$: Bias (threshold shift)\n"
     "- $z$: Pre-activation (logit)\n\n"
     "**Activation**\n"
     "$$a = \\sigma(z)$$\n\n"
     "Where $\\sigma$ is activation function (ReLU, sigmoid, tanh, etc.)\n\n",
     "Add neuron mathematics"),
    
    ("### Activation Functions Deep Dive\n\n"
     "**Sigmoid Function**\n"
     "$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$\n"
     "- Output: (0, 1)\n"
     "- Smooth gradient\n"
     "- Vanishing gradient problem\n\n"
     "```python\ndef sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\ndef sigmoid_derivative(z):\n    s = sigmoid(z)\n    return s * (1 - s)\n```\n\n"
     "**Hyperbolic Tangent**\n"
     "$$\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$$\n"
     "- Output: (-1, 1)\n"
     "- Centered at zero\n"
     "- Also has vanishing gradient issue\n\n",
     "Add sigmoid and tanh"),
    
    ("### ReLU and Modern Activations\n\n"
     "**Rectified Linear Unit (ReLU)**\n"
     "$$f(z) = \\max(0, z)$$\n"
     "- Advantages:\n"
     "  - Simple computation\n"
     "  - No vanishing gradient for positive values\n"
     "  - Speeds up convergence\n"
     "- Disadvantages:\n"
     "  - Dying ReLU (zero output for negative inputs)\n"
     "  - Not smooth at z=0\n\n"
     "```python\ndef relu(z):\n    return np.maximum(0, z)\n\ndef relu_derivative(z):\n    return (z > 0).astype(float)\n```\n\n"
     "**Leaky ReLU**\n"
     "$$f(z) = \\max(\\alpha z, z)$$\n"
     "- $\\alpha$ small (0.01): Allows small negative gradient\n"
     "- Prevents dying ReLU\n\n"
     "**ELU (Exponential Linear Unit)**\n"
     "$$f(z) = \\begin{cases} z & \\text{if } z > 0 \\\\ \\alpha(e^z - 1) & \\text{if } z \\leq 0 \\end{cases}$$\n"
     "- Closer to zero-mean outputs\n"
     "- Smoother near origin\n\n",
     "Add ReLU variants"),
    
    ("### Weight Initialization\n\n"
     "**Zero Initialization (Bad!)**\n"
     "- All neurons identical\n"
     "- Cannot break symmetry\n"
     "- Network remains linear\n\n"
     "**Xavier/Glorot Initialization**\n"
     "$$W \\sim \\text{Uniform}\\left(-\\sqrt{\\frac{6}{n + m}}, \\sqrt{\\frac{6}{n + m}}\\right)$$\n"
     "- For sigmoid/tanh\n"
     "- Maintains gradient magnitude\n\n"
     "```python\nfanin = prev_layer_size\nfanout = current_layer_size\nlimit = np.sqrt(6 / (fanin + fanout))\nW = np.random.uniform(-limit, limit, size=(fanin, fanout))\n```\n\n"
     "**He Initialization**\n"
     "$$W \\sim \\text{Normal}\\left(0, \\sqrt{\\frac{2}{n}}\\right)$$\n"
     "- For ReLU networks\n"
     "- Works better with ReLU activation\n\n",
     "Add initialization methods"),
    
    ("## Perceptron Learning Rule\n\n"
     "### Single Neuron Classification\n\n"
     "**Perceptron**\n"
     "- Simplest neural network unit\n"
     "- Separates linearly separable data\n"
     "- History: Rosenblatt (1958)\n\n"
     "**Algorithm**\n"
     "1. Initialize weights randomly\n"
     "2. For each training example:\n"
     "   - Compute prediction: $\\hat{y} = \\text{sign}(w \\cdot x + b)$\n"
     "   - If error: Update $w \\leftarrow w + y \\cdot x$\n"
     "3. Repeat until convergence\n\n"
     "```python\nclass Perceptron:\n    def __init__(self, learning_rate=0.01):\n        self.lr = learning_rate\n        self.w = None\n        self.b = 0\n    \n    def predict(self, X):\n        return np.sign(X @ self.w + self.b)\n    \n    def fit(self, X, y, epochs=100):\n        self.w = np.zeros(X.shape[1])\n        for epoch in range(epochs):\n            for xi, yi in zip(X, y):\n                pred = np.sign(xi @ self.w + self.b)\n                if pred != yi:\n                    self.w += self.lr * yi * xi\n                    self.b += self.lr * yi\n```\n\n",
     "Add perceptron"),
    
    ("### Perceptron Limitations\n\n"
     "**XOR Problem**\n"
     "Single perceptron cannot solve XOR:\n"
     "- XOR not linearly separable\n"
     "- Minsky and Papert (1969) proved this\n"
     "- Led to \"AI Winter\"\n\n"
     "**Visualization**\n"
     "```python\nX_xor = np.array([[0,0], [0,1], [1,0], [1,1]])\ny_xor = np.array([0, 1, 1, 0])  # XOR labels\n\n# Single perceptron fails\npercep = Perceptron()\npercep.fit(X_xor, y_xor)\nprint(percep.predict(X_xor))  # Cannot get all correct\n```\n\n"
     "**Solution: Hidden Layers**\n"
     "- Multi-layer perceptron (MLP) needed\n"
     "- Hidden layers create non-linear decision boundaries\n\n",
     "Add XOR problem"),
    
    ("## Multi-Layer Perceptron (MLP)\n\n"
     "### Architecture\n\n"
     "**Layer Composition**\n"
     "- Input layer: Raw features\n"
     "- Hidden layers: Learn representations\n"
     "- Output layer: Final predictions\n\n"
     "**Forward Pass Example (3-layer network)**\n"
     "$$h^{(1)} = \\sigma(W^{(1)} x + b^{(1)})$$\n"
     "$$h^{(2)} = \\sigma(W^{(2)} h^{(1)} + b^{(2)})$$\n"
     "$$\\hat{y} = f(W^{(3)} h^{(2)} + b^{(3)})$$\n\n"
     "Where:\n"
     "- $h^{(i)}$: Hidden layer activation\n"
     "- $W^{(i)}$: Weight matrix for layer i\n"
     "- $b^{(i)}$: Bias vector for layer i\n"
     "- $\\sigma$: Activation function (hidden)\n"
     "- $f$: Output activation (sigmoid for binary, softmax for multi-class)\n\n",
     "Add MLP architecture"),
    
    ("### Universal Approximation Theorem\n\n"
     "**Key Theorem**\n"
     "A feedforward network with single hidden layer can approximate:\n"
     "- Any continuous function on compact domain\n"
     "- With sufficient hidden units\n"
     "- Using non-linear activation functions\n\n"
     "**Implications**\n"
     "- Hidden layers provide expressiveness\n"
     "- One hidden layer theoretically sufficient\n"
     "- In practice: Deeper networks generalize better\n"
     "- Empirical evidence: Deep > shallow for many tasks\n\n"
     "**Mathematical Intuition**\n"
     "- Each hidden unit learns a feature\n"
     "- Combinations create complex decision regions\n"
     "- Non-linearity essential (linear layers compose to linear)\n\n",
     "Add universal approximation"),
    
    ("## Gradient Descent and Backpropagation\n\n"
     "### Why Gradient Descent?\n\n"
     "**Optimization Problem**\n"
     "Minimize: $$L(W, b) = \\frac{1}{m} \\sum_{i=1}^m \\ell(\\hat{y}^{(i)}, y^{(i)})$$\n\n"
     "Where:\n"
     "- $L$: Total loss\n"
     "- $\\ell$: Loss per sample\n"
     "- $m$: Number of training samples\n"
     "- Dimensions: Thousands to billions of parameters\n"
     "- Cannot solve analytically\n\n"
     "**Gradient Direction**\n"
     "$$\\nabla L = \\left[\\frac{\\partial L}{\\partial w_1}, ..., \\frac{\\partial L}{\\partial w_n}\\right]$$\n"
     "- Points toward steepest increase\n"
     "- Move opposite direction to decrease loss\n\n",
     "Add gradient descent motivation"),
    
    ("### Backpropagation Algorithm\n\n"
     "**Chain Rule Foundation**\n"
     "$$\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial a} \\frac{\\partial a}{\\partial z} \\frac{\\partial z}{\\partial w}$$\n\n"
     "**Forward Pass**\n"
     "- Compute all activations layer by layer\n"
     "- Store intermediate values for backward pass\n\n"
     "**Backward Pass**\n"
     "1. Compute output layer error: $\\delta^{(L)} = \\nabla_a L \\odot \\sigma'(z^{(L)})$\n"
     "2. Propagate backwards:\n"
     "   $$\\delta^{(l)} = (W^{(l+1)})^T \\delta^{(l+1)} \\odot \\sigma'(z^{(l)})$$\n"
     "3. Compute gradients:\n"
     "   $$\\frac{\\partial L}{\\partial W^{(l)}} = \\delta^{(l)} (a^{(l-1)})^T$$\n"
     "   $$\\frac{\\partial L}{\\partial b^{(l)}} = \\delta^{(l)}$$\n"
     "4. Update parameters:\n"
     "   $$W^{(l)} \\leftarrow W^{(l)} - \\alpha \\frac{\\partial L}{\\partial W^{(l)}}$$\n\n",
     "Add backpropagation algorithm"),
    
    ("### Computational Complexity of Backprop\n\n"
     "**Forward Pass Cost**\n"
     "- Deep network: O(L × N²) where L=layers, N=units per layer\n"
     "- Must compute all activations\n\n"
     "**Backward Pass Cost**\n"
     "- Similar to forward pass\n"
     "- But includes transpose operations\n"
     "- Typically 2x forward pass cost\n\n"
     "**Efficiency**\n"
     "```python\nn_params = sum(layers)  # E.g., millions\nforward_cost = O(n_params)\nbackward_cost = 2 * O(n_params)\ntotal_per_step = 3 * O(n_params)\n\n# 1000 training steps\ntotal = 3000 * O(n_params)\n```\n\n"
     "**Tricks to Speed Up**\n"
     "- Batch processing (vectorization)\n"
     "- GPU acceleration\n"
     "- Mixed precision (float32 instead of float64)\n\n",
     "Add backprop complexity"),
    
    ("## Loss Functions for Different Tasks\n\n"
     "### Mean Squared Error (MSE)\n\n"
     "**For Regression**\n"
     "$$\\text{MSE} = \\frac{1}{m} \\sum_{i=1}^m (\\hat{y}^{(i)} - y^{(i)})^2$$\n\n"
     "**Gradient**\n"
     "$$\\frac{\\partial \\text{MSE}}{\\partial \\hat{y}} = -2(y - \\hat{y})$$\n\n"
     "**Properties**\n"
     "- Convex (single global minimum)\n"
     "- Larger errors penalized more\n"
     "- Sensitive to outliers\n\n"
     "```python\ndef mse_loss(y_true, y_pred):\n    return np.mean((y_true - y_pred) ** 2)\n\ndef mse_gradient(y_true, y_pred):\n    return -2 * (y_true - y_pred) / len(y_true)\n```\n\n",
     "Add MSE loss"),
    
    ("### Cross-Entropy Loss\n\n"
     "**For Classification**\n"
     "$$\\text{CE} = -\\frac{1}{m} \\sum_{i=1}^m \\sum_{c=1}^C y_c^{(i)} \\log(\\hat{y}_c^{(i)})$$\n\n"
     "Where:\n"
     "- $C$: Number of classes\n"
     "- $y_c^{(i)}$: True label (one-hot)\n"
     "- $\\hat{y}_c^{(i)}$: Predicted probability\n\n"
     "**Binary Cross-Entropy**\n"
     "$$\\text{BCE} = -\\frac{1}{m} \\sum_{i=1}^m [y^{(i)} \\log(\\hat{y}^{(i)}) + (1 - y^{(i)}) \\log(1 - \\hat{y}^{(i)})]$$\n\n"
     "**Properties**\n"
     "- Specific to probability outputs\n"
     "- Penalizes confident mistakes heavily\n"
     "- Natural for softmax output\n\n"
     "```python\ndef cross_entropy_loss(y_true, y_pred, epsilon=1e-10):\n    return -np.mean(y_true * np.log(y_pred + epsilon))\n```\n\n",
     "Add cross-entropy loss"),
    
    ("## Softmax and Output Layers\n\n"
     "### Softmax Activation\n\n"
     "**Definition**\n"
     "$$\\text{softmax}(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$$\n\n"
     "**Properties**\n"
     "- Outputs sum to 1 (valid probability distribution)\n"
     "- Differentiable everywhere\n"
     "- Emphasizes maximum (winner-take-all)\n\n"
     "```python\ndef softmax(z):\n    # Numerical stability: subtract max\n    z = z - np.max(z, axis=1, keepdims=True)\n    exp_z = np.exp(z)\n    return exp_z / exp_z.sum(axis=1, keepdims=True)\n```\n\n"
     "**Numerical Stability**\n"
     "- $e^z$ grows very fast\n"
     "- Large $z$ values → overflow\n"
     "- Solution: Subtract max from each row\n"
     "- Mathematically equivalent, numerically stable\n\n",
     "Add softmax"),
    
    ("### Output Layer Design\n\n"
     "**Binary Classification**\n"
     "- Output: 1 neuron with sigmoid\n"
     "- Loss: Binary cross-entropy\n"
     "- Output interpretation: P(class=1)\n\n"
     "**Multi-class Classification**\n"
     "- Output: C neurons with softmax\n"
     "- Loss: Cross-entropy\n"
     "- Output interpretation: P(class=c) for each c\n\n"
     "**Regression**\n"
     "- Output: 1 or more neurons with linear\n"
     "- Loss: MSE or MAE\n"
     "- Output interpretation: Predicted value\n\n"
     "**Multi-task Learning**\n"
     "- Multiple output layers\n"
     "- Each with appropriate activation/loss\n"
     "- Shared hidden representations\n\n",
     "Add output layer design"),
    
    ("## Forward Pass Implementation from Scratch\n\n"
     "**Simple 3-Layer Network**\n"
     "```python\nclass SimpleNN:\n    def __init__(self, layer_sizes, learning_rate=0.01):\n        self.lr = learning_rate\n        self.params = {}\n        \n        # Initialize weights and biases\n        for i in range(len(layer_sizes) - 1):\n            self.params[f'W{i+1}'] = np.random.randn(\n                layer_sizes[i], layer_sizes[i+1]\n            ) * 0.01\n            self.params[f'b{i+1}'] = np.zeros((1, layer_sizes[i+1]))\n    \n    def relu(self, z):\n        return np.maximum(0, z)\n    \n    def sigmoid(self, z):\n        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))\n    \n    def forward(self, X):\n        self.cache = {}\n        A = X\n        \n        # Hidden layers\n        for i in range(1, 3):\n            Z = A @ self.params[f'W{i}'] + self.params[f'b{i}']\n            self.cache[f'Z{i}'] = Z\n            self.cache[f'A{i-1}'] = A\n            A = self.relu(Z)\n            self.cache[f'A{i}'] = A\n        \n        # Output layer\n        Z3 = A @ self.params['W3'] + self.params['b3']\n        self.cache['Z3'] = Z3\n        self.cache['A2'] = A\n        A3 = self.sigmoid(Z3)\n        self.cache['A3'] = A3\n        \n        return A3\n```\n\n",
     "Add forward implementation"),
    
    ("## Backward Pass Implementation from Scratch\n\n"
     "**Computing Gradients**\n"
     "```python\ndef backward(self, y):\n    m = y.shape[0]\n    grads = {}\n    \n    # Output layer error\n    dA3 = self.cache['A3'] - y\n    dZ3 = dA3  # Sigmoid derivative built into CE loss\n    \n    # Backprop through weights\n    grads['W3'] = self.cache['A2'].T @ dZ3 / m\n    grads['b3'] = np.sum(dZ3, axis=0) / m\n    \n    # Hidden layer 2\n    dA2 = dZ3 @ self.params['W3'].T\n    dZ2 = dA2 * (self.cache['Z2'] > 0)  # ReLU derivative\n    \n    grads['W2'] = self.cache['A1'].T @ dZ2 / m\n    grads['b2'] = np.sum(dZ2, axis=0) / m\n    \n    # Hidden layer 1\n    dA1 = dZ2 @ self.params['W2'].T\n    dZ1 = dA1 * (self.cache['Z1'] > 0)  # ReLU derivative\n    \n    grads['W1'] = self.cache['A0'].T @ dZ1 / m\n    grads['b1'] = np.sum(dZ1, axis=0) / m\n    \n    return grads\n\ndef update_params(self, grads):\n    for key in self.params:\n        self.params[key] -= self.lr * grads[key]\n```\n\n",
     "Add backward implementation"),
    
    ("## Training Loop and Convergence\n\n"
     "**Complete Training**\n"
     "```python\ndef train(self, X, y, epochs=100, batch_size=32):\n    losses = []\n    \n    for epoch in range(epochs):\n        epoch_loss = 0\n        n_batches = len(X) // batch_size\n        \n        for batch in range(n_batches):\n            start = batch * batch_size\n            end = start + batch_size\n            \n            X_batch = X[start:end]\n            y_batch = y[start:end]\n            \n            # Forward pass\n            output = self.forward(X_batch)\n            \n            # Compute loss\n            loss = -np.mean(y_batch * np.log(output + 1e-10))\n            epoch_loss += loss\n            \n            # Backward pass\n            grads = self.backward(y_batch)\n            \n            # Update weights\n            self.update_params(grads)\n        \n        avg_loss = epoch_loss / n_batches\n        losses.append(avg_loss)\n        \n        if (epoch + 1) % 10 == 0:\n            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')\n    \n    return losses\n```\n\n",
     "Add training loop"),
    
    ("## Debugging Neural Networks\n\n"
     "### Gradient Checking\n\n"
     "**Why Check?**\n"
     "- Backprop has many operations\n"
     "- Easy to make mistakes in implementation\n"
     "- Off-by-one errors, transpose mistakes\n\n"
     "**Numerical Gradient**\n"
     "$$\\frac{\\partial f}{\\partial w} \\approx \\frac{f(w + \\epsilon) - f(w - \\epsilon)}{2\\epsilon}$$\n\n"
     "**Verification**\n"
     "```python\nedsilon = 1e-5\nnumerical_grad = (loss(params + epsilon) - loss(params - epsilon)) / (2 * epsilon)\nanalytical_grad = compute_gradient(params)\ndiff = np.linalg.norm(numerical_grad - analytical_grad) / np.linalg.norm(numerical_grad + analytical_grad)\nassert diff < 1e-7, 'Gradient check failed!'\n```\n\n",
     "Add gradient checking"),
    
    ("### Common Issues and Fixes\n\n"
     "**Issue: Loss doesn't decrease**\n"
     "- Check 1: Learning rate (too high/low)\n"
     "- Check 2: Gradient sign (should be negative)\n"
     "- Check 3: Batch size effects\n\n"
     "**Issue: Gradient explosion**\n"
     "- Deep networks amplify gradients\n"
     "- Solution: Gradient clipping\n"
     "```python\nclip_value = 1.0\nfor param in grads:\n    grads[param] = np.clip(grads[param], -clip_value, clip_value)\n```\n\n"
     "**Issue: Gradient vanishing**\n"
     "- Sigmoid derivative < 0.25\n"
     "- Combines with chain rule → exponential decay\n"
     "- Solution: ReLU activation, batch normalization\n\n"
     "**Issue: Overfitting**\n"
     "- Too many parameters\n"
     "- Too many epochs\n"
     "- Solution: Regularization, dropout, early stopping\n\n",
     "Add debugging issues"),
]

readme_path = 'README.md'

print(f"Starting module-03 part 1 with {len(sections)} commits...")
print("=" * 60)

for i, (content, msg) in enumerate(sections, 1):
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(content)
    
    try:
        subprocess.run(['git', 'add', 'README.md'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', f'Enrich module-03 part1 {i}: {msg}'], check=True, capture_output=True)
        print(f"[OK] Commit {i:2d}/{len(sections)}: {msg}")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Part 1 {i:2d}: {msg}")

print("=" * 60)
print(f"[DONE] Part 1 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
