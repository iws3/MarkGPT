#!/usr/bin/env python3
"""
Module-04 enrichment part 2 - 22 commits
LSTM and GRU variants
"""
import subprocess
import os

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-04')

sections = [
    ("## Long Short-Term Memory (LSTM)\n\n"
     "### The Cell State Innovation\n\n"
     "Key insight: Separate cell state from hidden state\n"
     "c_t: Cell state (internal memory)\n"
     "h_t: Hidden state (external output)\n"
     "Cell state acts like \"conveyor belt\"\n"
     "Gradient can flow without vanishing.\n\n",
     "Add LSTM intro"),
    
    ("### Gates: Forget, Input, Output\n\n"
     "Three gating mechanisms:\n"
     "1. Forget gate: f_t = sigmoid(W_f @ [h_{t-1}; x_t] + b_f)\n"
     "   Controls what to discard from c_{t-1}\n"
     "2. Input gate: i_t = sigmoid(W_i @ [h_{t-1}; x_t] + b_i)\n"
     "   Controls what new info to add\n"
     "3. Output gate: o_t = sigmoid(W_o @ [h_{t-1}; x_t] + b_o)\n"
     "   Controls what to expose from c_t\n\n",
     "Add gates"),
    
    ("### Cell State Update\n\n"
     "Candidate cell state:\n"
     "c̃_t = tanh(W_c @ [h_{t-1}; x_t] + b_c)\n\n"
     "Cell state update:\n"
     "c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t\n"
     "⊙ denotes element-wise multiplication\n\n"
     "Hidden state:\n"
     "h_t = o_t ⊙ tanh(c_t)\n\n",
     "Add cell update"),
    
    ("### Gradient Flow Through Cell State\n\n"
     "∂c_t / ∂c_{t-1} = f_t (Hadamard product)\n"
     "f_t values in (0, 1) but not multiplication of many terms\n"
     "Much better gradient flow than standard RNN\n"
     "Allows gradients to propagate 100+ steps\n"
     "Solves vanishing gradient problem!\n\n",
     "Add LSTM gradients"),
    
    ("### LSTM Advantages\n\n"
     "Long-range dependencies: Can learn 100-200 step dependencies\n"
     "Forget gate: Can selectively discard info\n"
     "Input gate: Can control what to remember\n"
     "Output gate: Can control what to reveal\n"
     "Trade-off: 4x parameters vs standard RNN\n\n",
     "Add LSTM advantages"),
    
    ("## GRU: A Simpler Alternative\n\n"
     "### Motivation\n\n"
     "LSTM: 4 gates, complex, many parameters\n"
     "Can we simplify?\n"
     "GRU: 2 gates, simpler, 3x fewer parameters\n"
     "Similar performance on most tasks\n\n",
     "Add GRU intro"),
    
    ("### GRU Gates\n\n"
     "Reset gate: r_t = sigmoid(W_r @ [h_{t-1}; x_t] + b_r)\n"
     "Controls how much of h_{t-1} to use\n\n"
     "Update gate: z_t = sigmoid(W_z @ [h_{t-1}; x_t] + b_z)\n"
     "Controls how much of new info vs old\n\n"
     "Candidate state:\n"
     "h̃_t = tanh(W @ [r_t ⊙ h_{t-1}; x_t] + b)\n\n"
     "Final state:\n"
     "h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t\n\n",
     "Add GRU gates"),
    
    ("### LSTM vs GRU\n\n"
     "LSTM: Better with complex patterns\n"
     "GRU: Faster, fewer parameters\n"
     "Empirically: Often similar performance\n"
     "Use GRU: When compute is limited\n"
     "Use LSTM: When data is abundant\n"
     "Modern trend: Transformers replace both\n\n",
     "Add LSTM vs GRU"),
    
    ("## Stacked RNNs\n\n"
    "### Multiple Layers\n\n"
    "1D: Single RNN layer\n"
    "2D: Stack 2 RNN layers\n"
    "Output of layer 1 → input of layer 2\n\n"
    "Deep encoders: 2-4 layers beneficial\n"
    "Each layer computes higher-level features\n"
    "Example: Word embeddings → syntax → semantics\n\n"
    "Parameters: L * layers\n"
    "Training time: ~L * slower\n\n",
    "Add stacked RNNs"),
    
    ("### Residual Connections in RNNs\n\n"
     "Very deep RNNs: Training becomes hard\n"
     "Add skip connections: x_{l+2} = f(x_{l+1}) + x_l\n"
     "Enables training 4+ layer RNNs\n"
     "Helps gradient flow\n"
     "Used in cutting-edge models.\n\n",
     "Add residual RNN"),
    
    ("## Attention in RNNs\n\n"
     "### Problem: Bottleneck\n\n"
     "Encoder outputs h_T (single vector)\n"
     "Must contain all input information\n"
     "Problematic for long sequences (100+ tokens)\n"
     "Solution: Use all h_1, h_2, ..., h_T\n\n",
     "Add attention intro"),
    
    ("### Attention Mechanism\n\n"
     "Query: Decoder state s_t\n"
     "Keys: Encoder states h_1, ..., h_T\n"
     "Values: Encoder states h_1, ..., h_T\n\n"
     "Score: e_t,j = v^T @ tanh(W_s @ [s_t; h_j])\n"
     "Weights: α_t,j = softmax(e_t,j)\n"
     "Context: c_t = Σ α_t,j @ h_j\n\n"
     "Output: decoder processes [s_t; c_t]\n\n",
     "Add attention mech"),
    
    ("### Multiplicative Attention\n\n"
     "Simpler form (used in transformers):\n"
     "Score: e_t,j = (s_t @ h_j) / sqrt(d)\n"
     "No learned parameters in scoring\n"
     "Just dot product + softmax\n"
     "Scale by 1/sqrt(d) for stability\n"
     "Very efficient!\n\n",
     "Add multiplicative"),
    
    ("## Bidirectional LSTM\n\n"
     "### Design\n\n"
     "Forward LSTM: Process left to right\n"
     "Backward LSTM: Process right to left\n"
     "Outputs: [fwd_h_t; bwd_h_t]\n"
     "Cannot be used for generation (needs input sequence end)\n"
     "Great for tagging, classification\n\n",
     "Add bidirectional"),
    
    ("## PyTorch/TensorFlow LSTM Usage\n\n"
     "### PyTorch\n\n"
     "```python\n"
     "lstm = nn.LSTM(input_size, hidden_size, num_layers,\n"
     "               batch_first=True, bidirectional=True)\n"
     "outputs, (h_n, c_n) = lstm(x)  # x: (batch, T, input_size)\n"
     "# outputs: (batch, T, 2*hidden if bidirectional)\n"
     "# h_n: (num_layers*2, batch, hidden) if bidirectional\n"
     "# c_n: (num_layers*2, batch, hidden)\n"
     "```\n\n",
     "Add PyTorch LSTM"),
    
    ("### TensorFlow\n\n"
     "```python\n"
     "lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)\n"
     "outputs = lstm(x)  # x: (batch, T, input_size)\n"
     "# outputs: (batch, T, hidden_size)\n"
     "# For last output: outputs[:, -1, :]\n"
     "# Bidirectional: Bidirectional(LSTM(...))\n"
     "```\n\n",
     "Add TensorFlow LSTM"),
    
    ("## Encoder-Decoder with Attention\n\n"
     "### Architecture\n\n"
     "Encoder: Bi-LSTM reads input sequence\n"
     "Outputs: h_1, h_2, ..., h_T\n"
     "Decoder: LSTM generates output\n"
     "Each step: Attends to encoder outputs\n"
     "Completely parallelizable (replaced by Transformers)\n\n",
     "Add enc-dec attention"),
    
    ("### Context Vector\n\n"
     "Each decoder step:\n"
     "1. Query from decoder state\n"
     "2. Compute attention weights over all encoder outputs\n"
     "3. Weighted sum of encoder outputs\n"
     "4. Concatenate with decoder input for next step\n"
     "Very powerful (translation baseline ~30 BLEU).\n\n",
     "Add context vector"),
    
    ("## Common Practices\n\n"
     "### Dropout\n\n"
     "Apply to:\n"
     "- Input x_t\n"
     "- Between LSTM layers\n"
     "- NOT between time steps (breaks temporal coherence)\n"
     "Typical rate: 0.3-0.5\n"
     "Prevents overfitting on small datasets\n\n",
     "Add LSTM dropout"),
    
    ("### Learning Rate\n\n"
     "RNNs very sensitive to learning rate\n"
     "Start with 1e-3\n"
     "If diverges: Lower to 1e-4\n"
     "Use gradient clipping (max_norm=5.0)\n"
     "Warmup beneficial: Linear increase first 5% steps\n\n",
     "Add LSTM LR"),
    
    ("## Practical Tips\n\n"
     "- Start with 2 layers, expand if needed\n"
     "- Hidden size: 128-512 typical\n"
     "- Sequence length: 32-256 for NLP\n"
     "- Longer = more memory, smaller batches\n"
     "- Check gradient flow: norms should be O(0.1-1.0)\n"
     "- Monitor validation loss during training\n"
     "- Save checkpoint with best validation\n\n",
     "Add practical tips"),
]

readme_path = 'README.md'

print(f"Starting module-04 part 2 with {len(sections)} commits...")
print("=" * 60)

for i, (content, msg) in enumerate(sections, 1):
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(content)
    
    try:
        subprocess.run(['git', 'add', 'README.md'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', f'Enrich module-04 part2 {i}: {msg}'], check=True, capture_output=True)
        print(f"[OK] Commit {i:2d}/{len(sections)}: {msg}")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Part 2 {i:2d}: {msg}")

print("=" * 60)
print(f"[DONE] Part 2 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
