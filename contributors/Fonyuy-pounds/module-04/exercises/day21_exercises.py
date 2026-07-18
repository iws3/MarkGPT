"""
Day 21 Exercise: Long Short-Term Memory (LSTM) Networks
Module 04: Sequence Modeling
=====================================================

Task E21.1: Build a character-level LSTM trained on the Book of John.
- Implement an LSTM cell with forget, input, and output gates.
- Maintain both hidden state (h) and cell state (c).
- Compare performance/loss with the vanilla RNN from Day 20.
- Generate text continuations to see improved long-term dependency handling.
"""

import numpy as np
import os
import json
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y * y

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

class CharLSTM:
    def __init__(self, vocab_size, hidden_size, sequence_length):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # Combined weights for all 4 gates (forget, input, candidate, output)
        # We concatenate h_prev and x_t, so input size is hidden_size + vocab_size
        input_dim = hidden_size + vocab_size
        
        # Initialize weights
        self.W = np.random.randn(4 * hidden_size, input_dim) * 0.01
        self.b = np.zeros((4 * hidden_size, 1))
        
        # Output layer weights
        self.W_hy = np.random.randn(vocab_size, hidden_size) * 0.01
        self.b_y = np.zeros((vocab_size, 1))

    def forward(self, inputs, h_prev, c_prev):
        """
        inputs: list of integers
        h_prev: (hidden_size, 1)
        c_prev: (hidden_size, 1)
        """
        xs, hs, cs, is_, fs, os, gs, ys, ps = {}, {}, {}, {}, {}, {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(c_prev)
        
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            
            # Concatenate h_{t-1} and x_t
            concat = np.vstack((hs[t-1], xs[t]))
            
            # Compute all gates at once
            z = np.dot(self.W, concat) + self.b
            
            # Split z into 4 parts for the 4 gates
            f = sigmoid(z[0:self.hidden_size]) # forget gate
            i = sigmoid(z[self.hidden_size:2*self.hidden_size]) # input gate
            g = tanh(z[2*self.hidden_size:3*self.hidden_size]) # candidate cell state
            o = sigmoid(z[3*self.hidden_size:4*self.hidden_size]) # output gate
            
            fs[t], is_[t], gs[t], os[t] = f, i, g, o
            
            # Cell state update
            cs[t] = f * cs[t-1] + i * g
            
            # Hidden state update
            hs[t] = o * tanh(cs[t])
            
            # Output
            ys[t] = np.dot(self.W_hy, hs[t]) + self.b_y
            ps[t] = softmax(ys[t].T).T
            
        return xs, hs, cs, is_, fs, os, gs, ps

    def backward(self, xs, hs, cs, is_, fs, os, gs, ps, targets):
        dW, db = np.zeros_like(self.W), np.zeros_like(self.b)
        dWhy, dby = np.zeros_like(self.W_hy), np.zeros_like(self.b_y)
        
        dhnext = np.zeros_like(hs[0])
        dcnext = np.zeros_like(cs[0])
        
        loss = 0
        for t in reversed(range(len(targets))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            loss += -np.log(ps[t][targets[t], 0])
            
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            
            dh = np.dot(self.W_hy.T, dy) + dhnext
            
            # Output gate gradient
            do = dh * tanh(cs[t])
            do_raw = dsigmoid(os[t]) * do
            
            # Cell state gradient
            dc = dh * os[t] * dtanh(tanh(cs[t])) + dcnext
            
            # Candidate gradient
            dg = dc * is_[t]
            dg_raw = dtanh(gs[t]) * dg
            
            # Input gate gradient
            di = dc * gs[t]
            di_raw = dsigmoid(is_[t]) * di
            
            # Forget gate gradient
            df = dc * cs[t-1]
            df_raw = dsigmoid(fs[t]) * df
            
            # Combined gate gradients
            dz = np.vstack((df_raw, di_raw, dg_raw, do_raw))
            
            # Weight gradients
            concat = np.vstack((hs[t-1], xs[t]))
            dW += np.dot(dz, concat.T)
            db += dz
            
            # Next hidden and cell state gradients
            dhnext = np.dot(self.W[:self.hidden_size, :self.hidden_size].T, df_raw) + \
                     np.dot(self.W[self.hidden_size:2*self.hidden_size, :self.hidden_size].T, di_raw) + \
                     np.dot(self.W[2*self.hidden_size:3*self.hidden_size, :self.hidden_size].T, dg_raw) + \
                     np.dot(self.W[3*self.hidden_size:4*self.hidden_size, :self.hidden_size].T, do_raw)
            
            dcnext = fs[t] * dc
            
        # Gradient Clipping
        for dparam in [dW, db, dWhy, dby]:
            np.clip(dparam, -5, 5, out=dparam)
            
        return dW, db, dWhy, dby, loss

    def sample(self, h, c, seed_ix, n):
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            concat = np.vstack((h, x))
            z = np.dot(self.W, concat) + self.b
            f = sigmoid(z[0:self.hidden_size])
            i = sigmoid(z[self.hidden_size:2*self.hidden_size])
            g = tanh(z[2*self.hidden_size:3*self.hidden_size])
            o = sigmoid(z[3*self.hidden_size:4*self.hidden_size])
            c = f * c + i * g
            h = o * tanh(c)
            y = np.dot(self.W_hy, h) + self.b_y
            p = softmax(y.T).T
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

class CharGRU:
    def __init__(self, vocab_size, hidden_size, sequence_length):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # Combined weights for reset and update gates
        input_dim = hidden_size + vocab_size
        self.Wz = np.random.randn(hidden_size, input_dim) * 0.01
        self.Wr = np.random.randn(hidden_size, input_dim) * 0.01
        self.Wh = np.random.randn(hidden_size, input_dim) * 0.01
        
        self.bz = np.zeros((hidden_size, 1))
        self.br = np.zeros((hidden_size, 1))
        self.bh = np.zeros((hidden_size, 1))
        
        # Output layer
        self.W_hy = np.random.randn(vocab_size, hidden_size) * 0.01
        self.b_y = np.zeros((vocab_size, 1))

    def forward(self, inputs, h_prev):
        xs, hs, rs, zs, h_hats, ys, ps = {}, {}, {}, {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            
            concat = np.vstack((hs[t-1], xs[t]))
            
            # Gates
            zs[t] = sigmoid(np.dot(self.Wz, concat) + self.bz)
            rs[t] = sigmoid(np.dot(self.Wr, concat) + self.br)
            
            # Candidate hidden state
            concat_reset = np.vstack((rs[t] * hs[t-1], xs[t]))
            h_hats[t] = tanh(np.dot(self.Wh, concat_reset) + self.bh)
            
            # Final hidden state
            hs[t] = (1 - zs[t]) * hs[t-1] + zs[t] * h_hats[t]
            
            ys[t] = np.dot(self.W_hy, hs[t]) + self.b_y
            ps[t] = softmax(ys[t].T).T
            
        return xs, hs, rs, zs, h_hats, ps

    def sample(self, h, seed_ix, n):
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            concat = np.vstack((h, x))
            z = sigmoid(np.dot(self.Wz, concat) + self.bz)
            r = sigmoid(np.dot(self.Wr, concat) + self.br)
            concat_reset = np.vstack((r * h, x))
            h_hat = tanh(np.dot(self.Wh, concat_reset) + self.bh)
            h = (1 - z) * h + z * h_hat
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
    # Attempt to find the data directory
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 'data', 'processed', 'john_char'),
        r'c:\Users\the eye informatique\Desktop\ML\AI\MarkGPT\data\processed\john_char'
    ]
    
    data_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            data_dir = path
            break
            
    if not data_dir:
        print("Error: Data directory not found. Please ensure John dataset is preprocessed.")
        return

    train_data, val_data, meta = load_data(data_dir)
    vocab_size = meta['vocab_size']
    itos = {int(k): v for k, v in meta['itos'].items()}
    
    # Hyperparameters
    hidden_size = 128
    seq_length = 32
    learning_rate = 1e-1
    max_iters = 5000
    
    model = CharLSTM(vocab_size, hidden_size, seq_length)
    
    # AdaGrad memory
    mW, mb = np.zeros_like(model.W), np.zeros_like(model.b)
    mWhy, mby = np.zeros_like(model.W_hy), np.zeros_like(model.b_y)
    
    p = 0
    h_prev = np.zeros((hidden_size, 1))
    c_prev = np.zeros((hidden_size, 1))
    smooth_loss = -np.log(1.0/vocab_size) * seq_length
    
    print(f"Starting LSTM training on {len(train_data)} characters...")
    print(f"Vocab: {vocab_size}, Hidden: {hidden_size}, Seq: {seq_length}")
    
    start_time = time.time()
    for i in range(max_iters):
        if p + seq_length + 1 >= len(train_data):
            p = 0
            h_prev = np.zeros((hidden_size, 1))
            c_prev = np.zeros((hidden_size, 1))
            
        inputs = [int(x) for x in train_data[p : p + seq_length]]
        targets = [int(x) for x in train_data[p + 1 : p + seq_length + 1]]
        
        # Forward
        xs, hs, cs, is_, fs, os, gs, ps = model.forward(inputs, h_prev, c_prev)
        
        # Backward
        dW, db, dWhy, dby, loss = model.backward(xs, hs, cs, is_, fs, os, gs, ps, targets)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        
        # Update
        for param, dparam, mem in zip([model.W, model.b, model.W_hy, model.b_y],
                                    [dW, db, dWhy, dby],
                                    [mW, mb, mWhy, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
            
        h_prev = hs[seq_length - 1]
        c_prev = cs[seq_length - 1]
        p += seq_length
        
        if i % 500 == 0:
            elapsed = time.time() - start_time
            print(f"Iter {i}, Loss: {smooth_loss:.4f}, Time: {elapsed:.2f}s")
            sample_ix = model.sample(h_prev, c_prev, inputs[0], 100)
            txt = ''.join(itos[ix] for ix in sample_ix)
            print(f"----\n{txt}\n----")

    print(f"Training finished in {time.time() - start_time:.2f}s.")
    
    # Final sampling
    print("\nFinal Sampling (LSTM):")
    sample_ix = model.sample(np.zeros((hidden_size, 1)), np.zeros((hidden_size, 1)), int(train_data[0]), 500)
    print(''.join(itos[ix] for ix in sample_ix))

if __name__ == "__main__":
    train()
