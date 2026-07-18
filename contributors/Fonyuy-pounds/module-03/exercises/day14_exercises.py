import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def relu(z):
    """ReLU activation function"""
    return np.maximum(0, z)

class Layer:
    """A single layer in a Multi-Layer Perceptron"""
    def __init__(self, n_inputs, n_neurons, activation_fn=sigmoid):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.bias = np.zeros((1, n_neurons))
        self.activation_fn = activation_fn
        self.inputs = None
        self.z = None
        self.a = None
        
    def forward(self, inputs):
        self.inputs = inputs
        # z = X . W + b
        self.z = np.dot(inputs, self.weights) + self.bias
        self.a = self.activation_fn(self.z)
        return self.a

class MLP:
    """A Multi-Layer Perceptron"""
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

def visualize_decision_boundary(model, X, y, title="Decision Boundary"):
    """Visualizes the decision boundary of the MLP on a 2D dataset."""
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Flatten grid and pass through model
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid)
    Z = (Z > 0.5).astype(int) # Threshold at 0.5
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='bwr', edgecolors='k', s=100)
    plt.title(title)
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.grid(True)
    plt.savefig('day14_xor_decision_boundary.png')
    print(f"Plot saved as day14_xor_decision_boundary.png")

if __name__ == "__main__":
    # 1. Define XOR Dataset
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    print("--- Solving XOR with a 2-Layer MLP ---")
    
    # 2. Initialize MLP
    mlp = MLP()
    
    # Hidden Layer: 2 inputs, 2 neurons
    hidden_layer = Layer(2, 2, activation_fn=sigmoid)
    # Output Layer: 2 inputs (from hidden), 1 neuron
    output_layer = Layer(2, 1, activation_fn=sigmoid)
    
    mlp.add(hidden_layer)
    mlp.add(output_layer)
    
    # 3. Hand-coded weights to solve XOR
    # Since we learn backprop in Day 15, we manually set weights here to demonstrate
    # the representational power of the hidden layer.
    
    # Hidden neuron 1: OR gate
    # Hidden neuron 2: AND gate
    hidden_layer.weights = np.array([[20, 20],   # Weights for input 1
                                     [20, 20]])  # Weights for input 2
    hidden_layer.bias = np.array([[-10, -30]])   # Bias [OR, AND]
    
    # Output neuron: (OR) AND (NOT AND) -> XOR
    output_layer.weights = np.array([[20],      # +ve weight from OR
                                     [-40]])    # -ve weight from AND
    output_layer.bias = np.array([[-10]])
    
    # 4. Forward Pass
    predictions = mlp.forward(X_xor)
    print("XOR Inputs:\n", X_xor)
    print("Predictions (Probabilities):\n", predictions)
    print("Predictions (Binary):\n", (predictions > 0.5).astype(int))
    
    # 5. Visualize Decision Boundary
    visualize_decision_boundary(mlp, X_xor, y_xor, title="XOR Problem Solved by 2-Layer MLP")
    
    print("\n--- Exploring Representational Power of Depth ---")
    print("Adding a third layer allows the network to learn combinations of the boundaries")
    print("learned by the previous layers, enabling it to carve out highly complex regions.")
