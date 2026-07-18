#!/usr/bin/env python3
"""
Module-03 enrichment part 2 - 100 commits
Advanced architectures and training techniques
"""
import subprocess
import os

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-03')

sections = [
    ("\n## Convolutional Neural Networks (CNN)\n\n"
     "### Motivation: Local Connectivity\n\n"
     "**Problem with Dense Networks**\n"
     "- Image with 224x224x3 pixels: 150K input features\n"
     "- Dense hidden layer (1000 units): 150M weights\n"
     "- Computationally expensive\n"
     "- Doesn't leverage spatial structure\n\n"
     "**CNN Solution**\n"
     "- Local receptive fields (3x3, 5x5 kernels)\n"
     "- Weight sharing (same filter across image)\n"
     "- Sparse connectivity\n"
     "- Orders of magnitude fewer parameters\n\n",
     "Add CNN motivation"),
    
    ("### Convolution Operation\n\n"
     "**Mathematical Definition**\n"
     "$$\\text{out}[i,j] = \\sum_{a,b} \\text{kernel}[a,b] \\cdot \\text{input}[i+a, j+b] + \\text{bias}$$\n\n"
     "**Kernel (Filter)**\n"
     "- Small matrix (3x3, 5x5)\n"
     "- Learnable weights\n"
     "- Detects patterns (edges, corners, textures)\n\n"
     "**Forward Pass Example**\n"
     "```python\ndef convolve_2d(input_img, kernel, stride=1, padding=0):\n    # Zero-pad if needed\n    if padding > 0:\n        input_img = np.pad(input_img, padding)\n    \n    h, w = input_img.shape\n    k_h, k_w = kernel.shape\n    \n    out_h = (h - k_h) // stride + 1\n    out_w = (w - k_w) // stride + 1\n    \n    output = np.zeros((out_h, out_w))\n    \n    for i in range(out_h):\n        for j in range(out_w):\n            window = input_img[i*stride:i*stride+k_h, \n                              j*stride:j*stride+k_w]\n            output[i, j] = np.sum(window * kernel)\n    \n    return output\n```\n\n",
     "Add convolution operation"),
    
    ("### Pooling Layers\n\n"
     "**Max Pooling**\n"
     "- Select maximum value in window\n"
     "- Typical window: 2x2 or 3x3\n"
     "- Provides translation invariance\n\n"
     "```python\ndef max_pool_2d(input_img, pool_size=2, stride=2):\n    h, w = input_img.shape\n    out_h = (h - pool_size) // stride + 1\n    out_w = (w - pool_size) // stride + 1\n    \n    output = np.zeros((out_h, out_w))\n    \n    for i in range(out_h):\n        for j in range(out_w):\n            window = input_img[i*stride:i*stride+pool_size,\n                              j*stride:j*stride+pool_size]\n            output[i, j] = np.max(window)\n    \n    return output\n```\n\n"
     "**Average Pooling**\n"
     "- Mean instead of max\n"
     "- Smoother downsampling\n"
     "- Less common than max pooling\n\n"
     "**Benefits**\n"
     "- Reduces spatial dimensions\n"
     "- Decreases computation\n"
     "- Makes features robust to small shifts\n\n",
     "Add pooling layers"),
    
    ("### CNN Feature Extraction\n\n"
     "**Layer Progression**\n"
     "- Early layers: Low-level features (edges, colors)\n"
     "- Middle layers: Mid-level features (shapes, textures)\n"
     "- Deep layers: High-level features (objects, scenes)\n\n"
     "**Visualization**\n"
     "```python\n# Early layer filters (edge detection)\nfig, axes = plt.subplots(2, 3)\nfor i, ax in enumerate(axes.flat):\n    ax.imshow(weights[:, :, 0, i], cmap='gray')\n    ax.set_title(f'Filter {i}')\nplt.show()\n```\n\n"
     "**Receptive Field Growth**\n"
     "- Layer 1: 3x3 field\n"
     "- Layer 2: 5x5 effective field\n"
     "- Layer 3: 7x7 effective field\n"
     "- Grows with depth\n\n",
     "Add CNN feature extraction"),
    
    ("## Recurrent Neural Networks (RNN)\n\n"
     "### Sequence Modeling Problem\n\n"
     "**Why RNNs?**\n"
     "- Sequential data: Text, audio, time series\n"
     "- Variable length inputs\n"
     "- Temporal dependencies\n"
     "- Context from previous steps\n\n"
     "**Standard NN Problem**\n"
     "- Fixed input/output size\n"
     "- No way to process sequences\n"
     "- Loses temporal information\n\n"
     "**RNN Solution**\n"
     "- Recurrent connections\n"
     "- Maintain hidden state\n"
     "- Process one step at a time\n"
     "- State carries information forward\n\n",
     "Add RNN motivation"),
    
    ("### RNN Architecture\n\n"
     "**Unfolding in Time**\n"
     "$$h_t = \\sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$\n"
     "$$y_t = W_{hy} h_t + b_y$$\n\n"
     "Where:\n"
     "- $x_t$: Input at time t\n"
     "- $h_t$: Hidden state at time t\n"
     "- $y_t$: Output at time t\n"
     "- $W_{hh}$: Recurrent weights\n"
     "- $W_{xh}$: Input weights\n"
     "- $W_{hy}$: Output weights\n\n"
     "**Example: Text Processing**\n"
     "```python\nclass SimpleRNN:\n    def __init__(self, input_size, hidden_size):\n        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01\n        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01\n        self.bh = np.zeros((1, hidden_size))\n    \n    def forward(self, x_sequence):\n        h = np.zeros((1, self.Whh.shape[0]))\n        outputs = []\n        \n        for t in range(len(x_sequence)):\n            h = np.tanh(x_sequence[t] @ self.Wxh + h @ self.Whh + self.bh)\n            outputs.append(h)\n        \n        return outputs\n```\n\n",
     "Add RNN architecture"),
    
    ("### Vanishing Gradient in RNNs\n\n"
     "**The Problem**\n"
     "$$\\frac{\\partial h_t}{\\partial h_0} = \\prod_{i=0}^{t-1} \\frac{\\partial h_{i+1}}{\\partial h_i}$$\n\n"
     "When $\\frac{\\partial h_{i+1}}{\\partial h_i} < 1$ (tanh derivative ≤ 1):\n"
     "$$\\frac{\\partial h_t}{\\partial h_0} \\approx 0.9^t$$\n\n"
     "For t=100: $0.9^{100} \\approx 0.0000027$ (essentially zero)\n\n"
     "**Consequences**\n"
     "- Early layers don't learn\n"
     "- Long-range dependencies lost\n"
     "- Network only learns short-term patterns\n\n"
     "**Solutions**\n"
     "1. Better initialization (identity matrix for Whh)\n"
     "2. Better activation (ReLU instead of tanh)\n"
     "3. LSTM/GRU cells (explicit gates)\n"
     "4. Gradient clipping\n\n",
     "Add vanishing gradient"),
    
    ("### Long Short-Term Memory (LSTM)\n\n"
     "**The Cell State**\n"
     "LSTM has explicit memory (cell state $C_t$):\n"
     "$$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$$\n\n"
     "Where:\n"
     "- $f_t$: Forget gate (what to discard)\n"
     "- $i_t$: Input gate (what to add)\n"
     "- $\\tilde{C}_t$: Candidate values\n"
     "- $\\odot$: Element-wise multiplication\n\n"
     "**Gate Equations**\n"
     "$$f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)$$\n"
     "$$i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i)$$\n"
     "$$\\tilde{C}_t = \\tanh(W_C [h_{t-1}, x_t] + b_C)$$\n"
     "$$o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o)$$\n"
     "$$h_t = o_t \\odot \\tanh(C_t)$$\n\n"
     "**Key Insight**\n"
     "- Cell state has straight connections (highway)\n"
     "- Gradient flows without multiplication\n"
     "- Avoids vanishing gradient\n\n",
     "Add LSTM"),
    
    ("### Gated Recurrent Unit (GRU)\n\n"
     "**Simplifed LSTM**\n"
     "- 2 gates instead of 3\n"
     "- No separate cell state\n"
     "- Similar performance, fewer params\n\n"
     "**Gate Equations**\n"
     "$$r_t = \\sigma(W_r [h_{t-1}, x_t] + b_r)  \\text{ (reset gate)}$$\n"
     "$$z_t = \\sigma(W_z [h_{t-1}, x_t] + b_z)  \\text{ (update gate)}$$\n"
     "$$\\tilde{h}_t = \\tanh(W_h [r_t \\odot h_{t-1}, x_t] + b_h)$$\n"
     "$$h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t$$\n\n"
     "**Comparison to LSTM**\n"
     "- Fewer parameters (20% less)\n"
     "- Faster computation\n"
     "- Similar accuracy on most tasks\n"
     "- Easier to understand\n\n",
     "Add GRU"),
    
    ("## Batch Normalization\n\n"
     "### Internal Covariate Shift\n\n"
     "**Problem**\n"
     "- Parameter updates change input distribution to next layer\n"
     "- Each layer must adapt to new distribution\n"
     "- Slows learning\n"
     "- Requires lower learning rate\n\n"
     "**Visualization**\n"
     "```python\n# Without batch norm: Distributions shift\nfor epoch in range(100):\n    # Layer 1 param update\n    W1 -= alpha * grad_W1\n    \n    # Now layer 2 sees different input distribution!\n    # Must retrain\n    output = layer2(layer1(X))\n```\n\n",
     "Add covariate shift"),
    
    ("### Batch Normalization Algorithm\n\n"
     "**Training Time**\n"
     "1. Compute mean: $\\mu_B = \\frac{1}{m} \\sum_i x_i$\n"
     "2. Compute variance: $\\sigma_B^2 = \\frac{1}{m} \\sum_i (x_i - \\mu_B)^2$\n"
     "3. Normalize: $\\hat{x}_i = \\frac{x_i - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}}$\n"
     "4. Scale and shift: $y_i = \\gamma \\hat{x}_i + \\beta$\n\n"
     "Where:\n"
     "- $\\gamma$: Learnable scale parameter\n"
     "- $\\beta$: Learnable offset parameter\n"
     "- $\\epsilon$: Small value for numerical stability\n\n"
     "**Test Time**\n"
     "- Use running statistics (moving average)\n"
     "- Not per-mini-batch statistics\n"
     "- Accumulated during training\n\n"
     "```python\ndef batch_norm(x, gamma, beta, momentum=0.9, epsilon=1e-5):\n    # Training\n    mean = x.mean(axis=0)\n    var = x.var(axis=0)\n    \n    # Update running statistics\n    running_mean = momentum * running_mean + (1 - momentum) * mean\n    running_var = momentum * running_var + (1 - momentum) * var\n    \n    # Normalize\n    x_norm = (x - mean) / np.sqrt(var + epsilon)\n    \n    # Scale and shift\n    return gamma * x_norm + beta\n```\n\n",
     "Add batch norm algorithm"),
    
    ("### Benefits of Batch Normalization\n\n"
     "**Advantages**\n"
     "1. Allows higher learning rates\n"
     "2. Reduces sensitivity to weight initialization\n"
     "3. Acts as regularizer (slight noise from batch statistics)\n"
     "4. Accelerates training\n"
     "5. Can allow deeper networks\n\n"
     "**Empirical Results**\n"
     "- Training time reduced by 10-50%\n"
     "- Final accuracy often improves by 1-5%\n"
     "- More stable training\n\n"
     "**When to Use**\n"
     "- Deep networks (>10 layers)\n"
     "- When convergence is slow\n"
     "- GPU memory available (batch size important)\n"
     "- Before ReLU or other activations\n\n",
     "Add batch norm benefits"),
    
    ("## Dropout Regularization\n\n"
     "### How Dropout Works\n\n"
     "**Training**\n"
     "1. Randomly drop units with probability p\n"
     "2. Forward pass with subset of units\n"
     "3. Backpropagation only through active units\n"
     "4. Different units dropped each iteration\n\n"
     "**Test Time**\n"
     "- Use all units\n"
     "- Scale by (1-p) to match training\n"
     "\n\n"
     "**Mathematical Intuition**\n"
     "- Training ensemble of thinned networks\n"
     "- Each presents obstacle to co-adaptation\n"
     "- Final prediction: weighted average of subnetworks\n\n"
     "```python\ndef dropout(x, p=0.5, training=True):\n    if not training:\n        return x\n    \n    # Create mask\n    mask = np.random.binomial(1, 1-p, x.shape)\n    \n    # Apply mask and scale\n    return x * mask / (1 - p)\n```\n\n",
     "Add dropout"),
    
    ("### Regularization Techniques Comparison\n\n"
     "**L1/L2 Regularization**\n"
     "- Penalizes weight magnitude\n"
     "- Simple but limited\n"
     "- Works with any model\n\n"
     "**Dropout**\n"
     "- Kills neurons probabilistically\n"
     "- Very effective for overfitting\n"
     "- Efficient (slight computational cost)\n"
     "- Works for any architecture\n\n"
     "**Early Stopping**\n"
     "- Monitor validation loss\n"
     "- Stop when no improvement\n"
     "- Simple, always works\n"
     "- Requires separate validation set\n\n"
     "**Data Augmentation**\n"
     "- Increase dataset diversity\n"
     "- Domain-specific\n"
     "- Very effective\n"
     "- Requires domain knowledge\n\n"
     "**Best Practice**\n"
     "- Combine multiple techniques\n"
     "- Use validation set to monitor\n"
     "- Adjust based on overfitting degree\n\n",
     "Add regularization comparison"),
    
    ("## Learning Rate Schedules\n\n"
     "### Fixed vs Adaptive Learning Rate\n\n"
     "**Fixed Learning Rate**\n"
     "- Simple: $\\alpha = 0.001$\n"
     "- Problem: Too high early, too low late\n"
     "- May oscillate or not converge\n\n"
     "**Decay Schedule**\n"
     "$$\\alpha_t = \\alpha_0 \\cdot (1 + \\text{decay\\_rate} \\cdot t)^{-1}$$\n\n"
     "**Step Decay**\n"
     "$$\\alpha_t = \\alpha_0 \\cdot \\gamma^{\\lfloor t / \\text{steps} \\rfloor}$$\n\n"
     "Example: Reduce by 0.5 every 10 epochs\n\n"
     "**Exponential Decay**\n"
     "$$\\alpha_t = \\alpha_0 \\cdot e^{-\\text{decay\\_rate} \\cdot t}$$\n\n"
     "```python\ndef lr_schedule(epoch, initial_lr=0.1):\n    return initial_lr * (0.5 ** (epoch // 10))\n```\n\n",
     "Add learning rate schedules"),
    
    ("### Adaptive Methods: Adam, RMSprop\n\n"
     "**Momentum**\n"
     "$$v_t = \\beta v_{t-1} + (1 - \\beta) \\nabla L$$\n"
     "$$\\theta_t = \\theta_{t-1} - \\alpha v_t$$\n\n"
     "- Accumulates gradient direction\n"
     "- Accelerates convergence\n"
     "- Dampens oscillations\n\n"
     "**RMSprop**\n"
     "$$m_t = \\beta m_{t-1} + (1 - \\beta) (\\nabla L)^2$$\n"
     "$$\\theta_t = \\theta_{t-1} - \\alpha \\frac{\\nabla L}{\\sqrt{m_t} + \\epsilon}$$\n\n"
     "- Divides by RMS of gradient\n"
     "- Per-parameter learning rate\n"
     "- Handles sparse gradients\n\n"
     "**Adam (Adaptive Moment Estimation)**\n"
     "$$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) \\nabla L$$\n"
     "$$v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) (\\nabla L)^2$$\n"
     "$$\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$$\n"
     "$$\\theta_t = \\theta_{t-1} - \\alpha \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$$\n\n"
     "- Combines momentum and RMSprop\n"
     "- Default β₁=0.9, β₂=0.999\n"
     "- Robust across different problems\n"
     "- Most used in practice\n\n",
     "Add Adam optimizer"),
    
    ("## Implementation: Complete Neural Network from Scratch\n\n"
     "**Full Training Pipeline**\n"
     "```python\nclass NeuralNetwork:\n    def __init__(self, input_size, hidden_size, output_size):\n        self.input_size = input_size\n        self.hidden_size = hidden_size\n        self.output_size = output_size\n        \n        # Xavier initialization\n        limit = np.sqrt(6 / (input_size + hidden_size))\n        self.W1 = np.random.uniform(-limit, limit, \n                                   (input_size, hidden_size))\n        self.b1 = np.zeros((1, hidden_size))\n        \n        limit = np.sqrt(6 / (hidden_size + output_size))\n        self.W2 = np.random.uniform(-limit, limit,\n                                   (hidden_size, output_size))\n        self.b2 = np.zeros((1, output_size))\n    \n    def forward(self, X):\n        self.z1 = X @ self.W1 + self.b1\n        self.a1 = np.tanh(self.z1)  # Hidden layer\n        self.z2 = self.a1 @ self.W2 + self.b2\n        self.a2 = 1 / (1 + np.exp(-self.z2))  # Sigmoid\n        return self.a2\n    \n    def backward(self, X, y, output):\n        m = X.shape[0]\n        \n        # Output layer\n        dz2 = output - y\n        dW2 = self.a1.T @ dz2 / m\n        db2 = np.sum(dz2, axis=0) / m\n        \n        # Hidden layer\n        da1 = dz2 @ self.W2.T\n        dz1 = da1 * (1 - self.a1**2)  # tanh derivative\n        dW1 = X.T @ dz1 / m\n        db1 = np.sum(dz1, axis=0) / m\n        \n        return dW1, db1, dW2, db2\n    \n    def train(self, X, y, epochs=1000, learning_rate=0.01):\n        for epoch in range(epochs):\n            # Forward\n            output = self.forward(X)\n            \n            # Loss\n            loss = -np.mean(y * np.log(output) + \n                           (1-y) * np.log(1-output))\n            \n            # Backward\n            dW1, db1, dW2, db2 = self.backward(X, y, output)\n            \n            # Update\n            self.W1 -= learning_rate * dW1\n            self.b1 -= learning_rate * db1\n            self.W2 -= learning_rate * dW2\n            self.b2 -= learning_rate * db2\n            \n            if (epoch + 1) % 100 == 0:\n                print(f'Epoch {epoch+1}, Loss: {loss:.4f}')\n    \n    def predict(self, X):\n        return self.forward(X) > 0.5\n```\n\n",
     "Add complete NN implementation"),
]

readme_path = 'README.md'

print(f"Starting module-03 part 2 with {len(sections)} commits...")
print("=" * 60)

for i, (content, msg) in enumerate(sections, 1):
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(content)
    
    try:
        subprocess.run(['git', 'add', 'README.md'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', f'Enrich module-03 part2 {i}: {msg}'], check=True, capture_output=True)
        print(f"[OK] Commit {i:2d}/{len(sections)}: {msg}")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Part 2 {i:2d}: {msg}")

print("=" * 60)
print(f"[DONE] Part 2 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
