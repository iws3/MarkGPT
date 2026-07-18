import torch
import torch.nn as nn
import torch.optim as optim

def main():
    print("=== Day 12: PyTorch Basics ===")
    
    # 1. Create tensors
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    
    print("Tensor x:\n", x)
    print("Tensor y:\n", y)
    
    # 2. Perform operations
    z = torch.matmul(x, y)
    print("Result of x @ y:\n", z)
    
    # 3. Gradient computation with autograd
    loss = z.sum()
    loss.backward()
    
    print("Gradient of x:\n", x.grad)
    
    # 4. Train a single neuron
    print("\n--- Training a Single Neuron ---")
    # Simple linear regression task: y = 2x + 1
    X_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]])
    
    # Define a single neuron (Linear layer with 1 input and 1 output)
    model = nn.Linear(1, 1)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    print(f"Initial weights: {model.weight.item():.4f}, bias: {model.bias.item():.4f}")
    
    epochs = 1000
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    print(f"Final weights: {model.weight.item():.4f}, bias: {model.bias.item():.4f}")
    
    # Test the trained neuron
    test_val = torch.tensor([[5.0]])
    prediction = model(test_val).item()
    print(f"Prediction for x=5.0: {prediction:.4f} (Expected: 11.0)")

if __name__ == "__main__":
    main()
