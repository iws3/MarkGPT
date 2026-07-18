#!/usr/bin/env python3
"""
Module-04 enrichment part 1 - 25 commits
RNN fundamentals and vanishing gradients
"""
import subprocess
import os

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-04')

sections = [
    ("## Recurrent Neural Networks - Fundamentals\n\n"
     "### What is a Recurrent Network?\n\n"
     "Standard feedforward: x → h → output\n"
     "Recurrent: x_t → h_t → output_t\n"
     "h_t depends on x_t AND h_{t-1}\n"
     "Hidden state carries information from past.\n\n"
     "Processing sequences: One element at a time\n"
     "Weights shared across time steps (parameter efficiency).\n",
     "Add RNN fundamentals"),
    
    ("### Hidden State and Time Unrolling\n\n"
     "h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)\n"
     "o_t = W_o @ h_t + b_o\n\n"
     "Time unrolling: Unfold RNN for T time steps\n"
     "Creates deep feedforward network (depth = sequence length)\n"
     "Backprop through time (BPTT): Chain rule across time.\n\n",
     "Add hidden state"),
    
    ("## Sequence to Output\n\n"
     "### Many-to-One (e.g., sentiment)\n\n"
     "Input: Sequence of 100 words\n"
     "Process: Apply RNN at each step\n"
     "Output: Only use final h_T for classification\n"
     "Loss computes only on last output.\n"
     "Gradient flows backward through all steps.\n\n",
     "Add many-to-one"),
    
    ("### One-to-Many (e.g., image captioning)\n\n"
     "Input: Single image\n"
     "Process: Encode to h_0\n"
     "Output: Generate caption word by word\n"
     "h_0 from CNN → fed to RNN\n"
     "Each RNN step outputs word token.\n\n",
     "Add one-to-many"),
    
    ("### Many-to-Many (e.g., NER, machine translation)\n\n"
     "Input: Sequence of N tokens\n"
     "Process: RNN processes all\n"
     "Output: Sequence of N predictions\n"
     "Each time step has input and output.\n"
     "Examples: Sequence labeling, translation.\n\n",
     "Add many-to-many"),
    
    ("## The Vanishing Gradient Problem\n\n"
     "### The Issue\n\n"
     "BPTT: Chain rule multiplies gradients\n"
     "∂L/∂h_0 = (∂L/∂h_T) * (∂h_T/∂h_{T-1}) * ... * (∂h_1/∂h_0)\n"
     "Each ∂h_t/∂h_{t-1} < 1 typically\n"
     "Product of T < 1 terms → exponentially small\n"
     "Gradient for h_0 becomes nearly 0.\n\n",
     "Add vanishing gradients"),
    
    ("### Why This Matters\n\n"
     "Early inputs get negligible gradients.\n"
     "Model forgets distant past (effective window ~5-20 steps).\n"
     "Long-range dependencies can't be learned.\n"
     "Example: Pronoun in position 1, reference at position 50.\n"
     "RNN unlikely to learn this dependency.\n\n",
     "Add vanishing impact"),
    
    ("### Exploding Gradients (Opposite Problem)\n\n"
     "If ∂h_t/∂h_{t-1} > 1:\n"
     "Product of T > 1 terms → exponentially large\n"
     "Gradients overflow to NaN/Inf\n"
     "Training becomes unstable.\n"
     "Less common than vanishing but worse when it happens.\n\n",
     "Add exploding gradients"),
    
    ("## Solutions: Gradient Clipping\n\n"
     "### The Fix\n\n"
     "Gradient norm clipping:\n"
     "if ||∇|| > threshold:\n"
     "  ∇ = ∇ * (threshold / ||∇||)\n"
     "Rescales large gradients.\n"
     "Prevents explosion.\n"
     "Threshold: 1.0 or 5.0 typical.\n\n",
     "Add gradient clipping"),
    
    ("### Implementation\n\n"
     "Compute gradients as usual.\n"
     "Compute L2 norm: sqrt(sum of g^2).\n"
     "If norm > max_norm: rescale.\n"
     "Apply update.\n"
     "Handles both explosion and (partially) vanishing.\n\n",
     "Add clipping impl"),
    
    ("## Solutions: Better Activation Functions\n\n"
     "### ReLU in RNNs\n\n"
     "tanh: Saturates, derivative → 0\n"
     "ReLU: Linear on positive side, derivative = 1\n"
     "Helps gradients flow better.\n"
     "But can have dying ReLU problem.\n"
     "ELU/GELU: Smooth, no saturation.\n\n",
     "Add ReLU RNN"),
    
    ("## Simple RNN Implementation\n\n"
     "### Core Loop\n\n"
     "```python\n"
     "class SimpleRNN:\n"
     "  def forward(self, X):  # X: (T, batch, input_dim)\n"
     "    h = zeros((batch, hidden_dim))\n"
     "    outputs = []\n"
     "    for t in range(T):\n"
     "      h = tanh(X[t] @ Wx + h @ Wh + bh)\n"
     "      out = h @ Wo + bo\n"
     "      outputs.append(out)\n"
     "    return stack(outputs)\n"
     "```\n\n",
     "Add SimpleRNN impl"),
    
    ("### Backpropagation Through Time\n\n"
     "```python\n"
     "def backward(self, grad_output):  # (T, batch, out_dim)\n"
     "  dWx, dWh, dWo = 0, 0, 0\n"
     "  dh_next = 0\n"
     "  for t in reversed(range(T)):\n"
     "    dh = (grad_output[t] @ Wo.T + dh_next)\n"
     "    dWo += h[t].T @ grad_output[t]\n"
     "    dh = dh * (1 - h[t]**2)  # tanh derivative\n"
     "    dWx += X[t].T @ dh\n"
     "    dWh += h[t-1].T @ dh\n"
     "    dh_next = dh @ Wh.T\n"
     "```\n\n",
     "Add BPTT impl"),
    
    ("## Truncated Backpropagation\n\n"
     "### Motivation\n\n"
     "Full BPTT through entire sequence → slow\n"
     "Backprop only through last k steps\n"
     "Practical compromise: Efficient + reasonably good\n"
     "k values: 20-50 steps typical\n"
     "Still captures local temporal dependencies.\n\n",
     "Add truncated BPTT"),
    
    ("## Weight Initialization\n\n"
     "### Why It Matters\n\n"
     "Poor init: Gradients vanish/explode from start\n"
     "Good init: Preserve signal variance across layers\n"
     "Key: Keep ||h_t|| roughly constant\n"
     "Var(h_t) ≈ Var(h_{t-1})\n\n",
     "Add init motivation"),
    
    ("### Orthogonal Initialization\n\n"
     "Initialize Wh as orthogonal matrix\n"
     "Properties: Preserves vector norm\n"
     "Eigenvalues = 1 (no growth/decay)\n"
     "Prevents gradient explosion/vanishing initially.\n"
     "Recommended for RNNs.\n\n",
     "Add orthogonal init"),
    
    ("## Bidirectional RNNs\n\n"
     "### Motivation\n\n"
     "Forward RNN: Process left to right\n"
     "Backward RNN: Process right to left\n"
     "Concatenate outputs: [h_fwd; h_bwd]\n"
     "Access context from both directions.\n"
     "Improves performance on tagging tasks.\n\n",
     "Add bidirectional"),
    
    ("### Architecture\n\n"
     "Input sequence: [w1, w2, w3, w4]\n"
     "Forward pass: → → → →\n"
     "Backward pass: ← ← ← ←\n"
     "Output at t: [fwd_h_t; bwd_h_t]\n"
     "Dimension: 2 * hidden_dim\n\n",
     "Add bidir arch"),
    
    ("## Peephole Connections\n\n"
     "### With RNN\n\n"
     "Standard: h_t = f(W @ [x_t; h_{t-1}])\n"
     "No dependency on cell state (in basic RNN).\n"
     "Gradient flow during forward pass constrained.\n\n",
     "Add peephole intro"),
    
    ("## Sequence Padding and Masking\n\n"
     "### Variable Length Sequences\n\n"
     "Real sequences: Different lengths\n"
     "Batch processing: Need same length\n"
     "Solution: Pad short sequences\n"
     "Padding token: 0 (special index)\n"
     "Sequence lengths: Store actual lengths\n\n",
     "Add padding"),
    
    ("### Masking\n\n"
     "During forward: Process padded positions\n"
     "During loss: Ignore padded positions\n"
     "Loss = sum(loss[i] * mask[i]) / sum(mask)\n"
     "Prevents gradients from padding tokens.\n"
     "Attention:  Mask with -inf (softmax → 0).\n\n",
     "Add masking"),
    
    ("## Common RNN Patterns\n\n"
     "### Encoder-Decoder (No Attention)\n\n"
     "Encoder: Process input → final h_T\n"
     "h_T: Summary of entire input\n"
     "Decoder: Initialize with h_T, generate output\n"
     "Limitation: All info in single vector\n"
     "Better approach: Use attention (module-05).\n\n",
     "Add encoder-decoder"),
    
    ("### Autoregressive Generation\n\n"
     "At test time: Generate one token at a time\n"
     "Use own output as next input\n"
     "Temperature: Control randomness\n"
     "Sampling vs beam search tradeoffs\n"
     "Exposure bias: Train vs test mismatch.\n\n",
     "Add autoregressive"),
    
    ("## Practical Considerations\n\n"
     "### Sequence Length\n\n"
     "Very long sequences: Truncated BPTT\n"
     "Typical: 50-512 tokens\n"
     "Maximum: GPU memory constraint\n"
     "Tradeoff: Longer = more context, slower training\n\n",
     "Add seq length"),
    
    ("### Batch Size\n\n"
     "Standard: 32-128\n"
     "Affects gradient estimate quality\n"
     "Memory per sequence * batch_size\n"
     "Typical GPU: batch_size=64 for seq_len=512\n"
     "Larger batch = noisier gradients\n\n",
     "Add batch size"),
]

readme_path = 'README.md'

print(f"Starting module-04 part 1 with {len(sections)} commits...")
print("=" * 60)

for i, (content, msg) in enumerate(sections, 1):
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(content)
    
    try:
        subprocess.run(['git', 'add', 'README.md'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', f'Enrich module-04 part1 {i}: {msg}'], check=True, capture_output=True)
        print(f"[OK] Commit {i:2d}/{len(sections)}: {msg}")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Part 1 {i:2d}: {msg}")

print("=" * 60)
print(f"[DONE] Part 1 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
