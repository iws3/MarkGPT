"""
Day 19 Exercise: The Problem of Memory
Module 04: Sequence Modeling
========================================

Task: Implement a "manual" sequence processor.
We will try to use a standard MLP to 'remember' a sequence by updating
a hidden state at each step. We will then observe how quickly the 
information from the beginning of the sequence is lost (Forgotten).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ManualSequenceProcessor:
    def __init__(self, input_dim, hidden_dim):
        # We simulate a "step" function: h_t = sigmoid(W_xh * x_t + W_hh * h_{t-1} + b)
        # Note: This is essentially one step of a simple RNN cell, 
        # but we are building it to see the limitations.
        
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self.W_xh = np.random.randn(input_dim, hidden_dim) * 0.1
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b = np.zeros((1, hidden_dim))
        
    def step(self, x, h_prev):
        # x: (1, input_dim)
        # h_prev: (1, hidden_dim)
        h_next = np.tanh(np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b)
        return h_next

def test_memory():
    input_dim = 10
    hidden_dim = 20
    sequence_length = 50
    
    processor = ManualSequenceProcessor(input_dim, hidden_dim)
    
    # Let's see if the hidden state can 'remember' the first input
    # We'll use a specific 'signal' as the first input and random noise for the rest
    signal = np.ones((1, input_dim))
    h = np.zeros((1, hidden_dim))
    
    # Step 0: Feed the signal
    h = processor.step(signal, h)
    initial_h = h.copy()
    
    correlations = [1.0] # Correlation of current h with the initial 'memory' h
    
    print(f"Sequence Length: {sequence_length}")
    print("-" * 30)
    
    for t in range(1, sequence_length):
        # Feed random noise
        noise = np.random.randn(1, input_dim) * 0.1
        h = processor.step(noise, h)
        
        # Calculate cosine similarity between current h and initial_h
        # This tells us how much of the original 'signal' remains in the state
        similarity = np.dot(initial_h, h.T) / (np.linalg.norm(initial_h) * np.linalg.norm(h))
        correlations.append(similarity[0,0])
        
        if t % 10 == 0:
            print(f"Step {t}: Similarity to initial state = {correlations[-1]:.4f}")

    # Plotting the 'forgetting' curve
    plt.figure(figsize=(10, 6))
    plt.plot(correlations, marker='o', color='red')
    plt.title("Memory Decay in a Simple Sequential MLP Step")
    plt.xlabel("Time Step (Random Noise added)")
    plt.ylabel("Cosine Similarity to Initial Signal State")
    plt.grid(True)
    
    # Save the plot
    import os
    save_path = os.path.join(os.path.dirname(__file__), "day19_memory_decay.png")
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")
    print("\nObservation: Notice how the similarity drops as more noise steps are processed.")
    print("This demonstrates why simple recurrent steps without gating (like LSTMs) or")
    print("global attention struggle with long-term memory.")

if __name__ == "__main__":
    test_memory()
