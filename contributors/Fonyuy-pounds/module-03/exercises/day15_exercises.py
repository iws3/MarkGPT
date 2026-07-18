import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of the sigmoid function"""
    s = sigmoid(z)
    return s * (1 - s)

class MLP:
    """A Multi-Layer Perceptron with Backpropagation"""
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
    def forward(self, X):
        """Perform the forward pass and cache values for backprop"""
        self.X = X
        
        # Layer 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        
        # Layer 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        
        return self.A2
        
    def backward(self, Y, learning_rate=0.1):
        """Perform the backward pass and update weights using gradient descent"""
        m = Y.shape[0] # batch size
        
        # 1. Output Layer Error
        # dL/dZ2 = (A2 - Y) * sigmoid_derivative(Z2)
        dZ2 = (self.A2 - Y) * sigmoid_derivative(self.Z2) / m
        
        # Gradients for Layer 2
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # 2. Hidden Layer Error
        # dA1 = np.dot(dZ2, W2.T)
        # dZ1 = dA1 * sigmoid_derivative(Z1)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.Z1)
        
        # Gradients for Layer 1
        dW1 = np.dot(self.X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # 3. Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

def train_xor():
    # XOR Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    
    # Initialize MLP: 2 inputs, 4 hidden neurons, 1 output neuron
    mlp = MLP(input_size=2, hidden_size=4, output_size=1)
    
    epochs = 10000
    learning_rate = 1.0
    losses = []
    
    print("--- Training MLP on XOR with Backpropagation ---")
    
    for epoch in range(epochs):
        # Forward pass
        predictions = mlp.forward(X)
        
        # Compute loss (Mean Squared Error)
        loss = np.mean(0.5 * (predictions - Y) ** 2)
        losses.append(loss)
        
        # Backward pass
        mlp.backward(Y, learning_rate)
        
        if (epoch + 1) % 2000 == 0:
            print(f"Epoch {epoch + 1:5d} / {epochs} | Loss: {loss:.4f}")
            
    print("\n--- Training Complete ---")
    print("Final Predictions:")
    final_preds = mlp.forward(X)
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Target: {Y[i][0]} -> Predicted: {final_preds[i][0]:.4f}")
        
    # Plot loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(losses, color='purple', linewidth=2)
    plt.title("Training Loss Curve (XOR Problem)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.savefig('day15_loss_curve.png')
    print("\nLoss curve saved as day15_loss_curve.png")

if __name__ == "__main__":
    train_xor()
