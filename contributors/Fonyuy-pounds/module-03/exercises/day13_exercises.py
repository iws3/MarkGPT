import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Sigmoid activation function: 1 / (1 + exp(-z))"""
    return 1 / (1 + np.exp(-z))

def tanh(z):
    """Hyperbolic tangent activation function: (exp(z) - exp(-z)) / (exp(z) + exp(-z))"""
    return np.tanh(z)

def relu(z):
    """Rectified Linear Unit: max(0, z)"""
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    """Leaky Rectified Linear Unit: max(alpha*z, z)"""
    return np.where(z > 0, z, alpha * z)

class Neuron:
    """
    A simple artificial neuron (perceptron-like) that implements:
    output = activation(sum(weights * inputs) + bias)
    """
    def __init__(self, n_inputs, activation_fn=relu):
        # Initialize weights randomly (small values) and bias to zero
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.activation_fn = activation_fn
        
    def forward(self, inputs):
        """Perform the forward pass"""
        # z = w . x + b
        self.z = np.dot(self.weights, inputs) + self.bias
        # a = f(z)
        self.a = self.activation_fn(self.z)
        return self.a

def visualize_activations():
    """Visualize different activation functions"""
    z = np.linspace(-5, 5, 100)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(z, sigmoid(z), label='Sigmoid', color='blue')
    plt.title('Sigmoid Activation')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(z, tanh(z), label='Tanh', color='orange')
    plt.title('Tanh Activation')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(z, relu(z), label='ReLU', color='green')
    plt.title('ReLU Activation')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(z, leaky_relu(z), label='Leaky ReLU', color='red')
    plt.title('Leaky ReLU Activation')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('activation_functions.png')
    print("Activation functions plot saved as 'activation_functions.png'")

if __name__ == "__main__":
    # 1. Test Single Neuron
    print("--- Testing Single Neuron ---")
    my_neuron = Neuron(n_inputs=3, activation_fn=sigmoid)
    my_neuron.weights = np.array([0.5, -0.2, 0.1])
    my_neuron.bias = 0.1
    
    test_input = np.array([1.0, 2.0, 3.0])
    output = my_neuron.forward(test_input)
    
    print(f"Weights: {my_neuron.weights}")
    print(f"Bias: {my_neuron.bias}")
    print(f"Input: {test_input}")
    print(f"Output (Sigmoid): {output:.4f}")
    
    # 2. Test Different Activations
    print("\n--- Testing Activation Functions ---")
    z_test = 2.0
    print(f"Input z = {z_test}")
    print(f"Sigmoid({z_test}) = {sigmoid(z_test):.4f}")
    print(f"Tanh({z_test}) = {tanh(z_test):.4f}")
    print(f"ReLU({z_test}) = {relu(z_test):.4f}")
    
    z_test_neg = -2.0
    print(f"\nInput z = {z_test_neg}")
    print(f"Sigmoid({z_test_neg}) = {sigmoid(z_test_neg):.4f}")
    print(f"Tanh({z_test_neg}) = {tanh(z_test_neg):.4f}")
    print(f"ReLU({z_test_neg}) = {relu(z_test_neg):.4f}")
    print(f"Leaky ReLU({z_test_neg}) = {leaky_relu(z_test_neg):.4f}")

    # 3. Visualization (Optional)
    # visualize_activations()
