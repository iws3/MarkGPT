"""
Day 20 Exercise: Recurrent Neural Networks (RNNs)
Module 04: Sequence Modeling
================================================

Task E20.1: Build a character-level RNN trained on the Book of John.
- Implement a vanilla RNN cell: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
- Use Backpropagation Through Time (BPTT).
- Implement Gradient Clipping to handle exploding gradients.
- Generate text continuations.
"""

import numpy as np
import os
import json
import time

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

class CharRNN:
    def __init__(self, vocab_size, hidden_size, sequence_length):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # Model parameters
        self.W_xh = np.random.randn(hidden_size, vocab_size) * 0.01 # input to hidden
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01 # hidden to hidden
        self.W_hy = np.random.randn(vocab_size, hidden_size) * 0.01 # hidden to output
        self.b_h = np.zeros((hidden_size, 1)) # hidden bias
        self.b_y = np.zeros((vocab_size, 1)) # output bias

    def forward(self, inputs, h_prev):
        """
        inputs: list of integers (token IDs)
        h_prev: (hidden_size, 1)
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0
        
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1)) # one-hot encoding
            xs[t][inputs[t]] = 1
            
            # RNN Cell: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            hs[t] = np.tanh(np.dot(self.W_xh, xs[t]) + np.dot(self.W_hh, hs[t-1]) + self.b_h)
            
            # Output: y_t = W_hy * h_t + b_y
            ys[t] = np.dot(self.W_hy, hs[t]) + self.b_y
            
            # Probabilities: p_t = softmax(y_t)
            ps[t] = softmax(ys[t].T).T
            
        return xs, hs, ps

    def backward(self, xs, hs, ps, targets):
        """
        BPTT implementation.
        """
        dWxh, dWhh, dWhy = np.zeros_like(self.W_xh), np.zeros_like(self.W_hh), np.zeros_like(self.W_hy)
        dbh, dby = np.zeros_like(self.b_h), np.zeros_like(self.b_y)
        dhnext = np.zeros_like(hs[0])
        
        loss = 0
        for t in reversed(range(len(targets))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # backprop through softmax/cross-entropy
            
            loss += -np.log(ps[t][targets[t], 0])
            
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            
            dh = np.dot(self.W_hy.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh
            
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.W_hh.T, dhraw)
            
        # Gradient Clipping
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
            
        return dWxh, dWhh, dWhy, dbh, dby, loss

    def sample(self, h, seed_ix, n):
        """
        Sample a sequence of n characters from the model.
        h: initial hidden state
        seed_ix: first character to start sampling from
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.W_xh, x) + np.dot(self.W_hh, h) + self.b_h)
            y = np.dot(self.W_hy, h) + self.b_y
            p = softmax(y.T).T
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

def load_data(data_dir):
    meta_path = os.path.join(data_dir, 'meta.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    train_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)
    val_data = np.fromfile(os.path.join(data_dir, 'val.bin'), dtype=np.uint16)
    
    return train_data, val_data, meta

def train():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 'data', 'processed', 'john_char')
    if not os.path.exists(data_dir):
        # Try absolute path from workspace root if relative fails
        workspace_root = r'c:\Users\the eye informatique\Desktop\ML\AI\MarkGPT'
        data_dir = os.path.join(workspace_root, 'data', 'processed', 'john_char')
        if not os.path.exists(data_dir):
            print(f"Error: Data directory {data_dir} not found. Run scripts/preprocess_char.py first.")
            return

    train_data, val_data, meta = load_data(data_dir)
    vocab_size = meta['vocab_size']
    itos = {int(k): v for k, v in meta['itos'].items()}
    
    # Hyperparameters
    hidden_size = 100
    seq_length = 25
    learning_rate = 1e-1
    max_iters = 5000
    
    model = CharRNN(vocab_size, hidden_size, seq_length)
    
    # AdaGrad memory
    mWxh, mWhh, mWhy = np.zeros_like(model.W_xh), np.zeros_like(model.W_hh), np.zeros_like(model.W_hy)
    mbh, mby = np.zeros_like(model.b_h), np.zeros_like(model.b_y)
    
    p = 0
    h_prev = np.zeros((hidden_size, 1))
    smooth_loss = -np.log(1.0/vocab_size) * seq_length
    
    print(f"Starting training on {len(train_data)} characters...")
    print(f"Vocab size: {vocab_size}, Hidden size: {hidden_size}, Seq length: {seq_length}")
    
    for i in range(max_iters):
        # Prepare inputs and targets
        if p + seq_length + 1 >= len(train_data):
            p = 0
            h_prev = np.zeros((hidden_size, 1))
            
        inputs = [int(x) for x in train_data[p : p + seq_length]]
        targets = [int(x) for x in train_data[p + 1 : p + seq_length + 1]]
        
        # Forward pass
        xs, hs, ps = model.forward(inputs, h_prev)
        
        # Backward pass
        dWxh, dWhh, dWhy, dbh, dby, loss = model.backward(xs, hs, ps, targets)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        
        # Update parameters with AdaGrad
        for param, dparam, mem in zip([model.W_xh, model.W_hh, model.W_hy, model.b_h, model.b_y],
                                    [dWxh, dWhh, dWhy, dbh, dby],
                                    [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
            
        h_prev = hs[seq_length - 1]
        p += seq_length
        
        if i % 500 == 0:
            print(f"Iter {i}, Smooth Loss: {smooth_loss:.4f}", flush=True)
            sample_ix = model.sample(h_prev, inputs[0], 100)
            txt = ''.join(itos[ix] for ix in sample_ix)
            print(f"----\n{txt}\n----", flush=True)

            
    print("Training finished.")
    
    # Final sample
    print("\nFinal sampling:")
    sample_ix = model.sample(np.zeros((hidden_size, 1)), [int(x) for x in train_data[:1]][0], 500)
    print(''.join(itos[ix] for ix in sample_ix))

if __name__ == "__main__":
    train()
