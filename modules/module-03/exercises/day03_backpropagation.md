# Module 3 Exercise: Backpropagation from Scratch

## Objective
Implement backpropagation by hand to understand gradient flow.

## Background
Most ML frameworks (PyTorch, TensorFlow) compute gradients automatically (**autograd**). But understanding the manual process teaches you:
1. Where gradients come from
2. Why some architectures have vanishing/exploding gradients
3. How to debug training failures

## Part 1: Gradient of Simple Functions

```python
import numpy as np

# TODO: Compute gradient of f(x) = x^3 at x=2
# By hand: df/dx = 3x^2, so at x=2: df/dx = 3*4 = 12

x = 2.0
gradient = None  # Replace with your computation

# Verify with numerical gradient
def numerical_gradient(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

f = lambda x: x**3
numerical_grad = numerical_gradient(f, x)
assert np.isclose(gradient, numerical_grad), f"Got {gradient}, expected {numerical_grad}"
```

## Part 2: Chain Rule for Composite Functions

```python
# f(x) = (x^2 + 3)^2
# What is df/dx at x=2?

# By hand using chain rule:
# Let u = x^2 + 3
# f = u^2
# df/dx = df/du * du/dx = 2u * 2x = 2(x^2 + 3) * 2x

x = 2.0

# TODO: Compute du/dx (derivative of inner function)
du_dx = None

# TODO: Compute df/du (derivative of outer function) at u
u = x**2 + 3
df_du = None

# TODO: Apply chain rule: df/dx = df/du * du/dx
gradient = None

# Verify
numerical_grad = numerical_gradient(lambda x: (x**2 + 3)**2, x)
assert np.isclose(gradient, numerical_grad)
```

## Part 3: Backpropagation Through a Network

Simple network: x → [Linear + ReLU] → [Linear] → loss

```python
import numpy as np

# Network: y = W2 @ ReLU(W1 @ x + b1) + b2
# where ReLU(z) = max(0, z)

# Forward pass
x = np.array([1.0, 2.0, 3.0])           # Input (3,)
W1 = np.random.randn(4, 3)              # Hidden weights (4, 3)
b1 = np.zeros(4)
W2 = np.random.randn(1, 4)              # Output weights (1, 4)
b2 = np.zeros(1)

y_true = np.array([5.0])                # Target

# Forward
z1 = W1 @ x + b1                        # (4,)
a1 = np.maximum(0, z1)                  # ReLU
z2 = W2 @ a1 + b2                       # (1,)

# Loss: MSE = (z2 - y_true)^2
loss = (z2 - y_true) ** 2

# BACKWARD: Compute gradients manually

# TODO: dL/dz2 (gradient of loss w.r.t. output)
dloss_dz2 = None  # Hint: d(MSE)/dz2 = 2*(z2 - y_true)

# TODO: dL/dW2 = (dL/dz2) * (dz2/dW2) = dL/dz2 @ a1.T
dloss_dW2 = None

# TODO: dL/db2 = (dL/dz2)
dloss_db2 = None

# TODO: dL/da1 = (dL/dz2) @ W2
dloss_da1 = None

# TODO: dL/dz1 = (dL/da1) * (da1/dz1), where da1/dz1 = 1 if z1>0 else 0 (ReLU derivative)
dloss_dz1 = None  # Mask gradient by (z1 > 0)

# TODO: dL/dW1 = (dL/dz1) @ x.T
dloss_dW1 = None

# TODO: dL/db1 = (dL/dz1)
dloss_db1 = None

# Verify with numerical gradients (for W1, first element only)
eps = 1e-5
W1_plus = W1.copy()
W1_plus[0, 0] += eps
z1_plus = W1_plus @ x + b1
a1_plus = np.maximum(0, z1_plus)
z2_plus = W2 @ a1_plus + b2
loss_plus = (z2_plus - y_true) ** 2

W1_minus = W1.copy()
W1_minus[0, 0] -= eps
z1_minus = W1_minus @ x + b1
a1_minus = np.maximum(0, z1_minus)
z2_minus = W2 @ a1_minus + b2
loss_minus = (z2_minus - y_true) ** 2

numerical_dW1[0, 0] = (loss_plus - loss_minus) / (2 * eps)
print(f"Analytical dW1[0,0]: {dloss_dW1[0, 0]:.6f}")
print(f"Numerical dW1[0,0]: {numerical_dW1[0, 0]:.6f}")
assert np.isclose(dloss_dW1[0, 0], numerical_dW1[0, 0], rtol=1e-3)
```

## Key Insights

1. **Gradients flow backward**: From loss → hidden layers → weights
2. **Chain rule**: Multiply local gradients along the path
3. **ReLU gradient**: Zero for negative inputs (sparse gradient)
4. **Numerical verification**: Always check analytical gradients against numerical!

## Debugging Tip: Gradient Checking

```python
def check_gradients(model, x, y_true, eps=1e-5):
    """Verify all gradients are correct"""
    
    # Analytical: use backprop
    loss, gradients = model.backward(x, y_true)
    
    # Numerical: perturb each parameter
    for name, param in model.params.items():
        for i in range(param.size):
            # Perturb
            param.flat[i] += eps
            loss_plus = model.forward(x, y_true)
            
            param.flat[i] -= 2*eps
            loss_minus = model.forward(x, y_true)
            
            param.flat[i] += eps
            
            # Compare
            numerical_grad = (loss_plus - loss_minus) / (2*eps)
            analytical_grad = gradients[name].flat[i]
            
            # Should match
            rel_error = abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + abs(analytical_grad) + 1e-8)
            if rel_error > 1e-3:
                print(f"MISMATCH in {name}[{i}]: analytical={analytical_grad:.6f}, numerical={numerical_grad:.6f}")
```

## References

- Goodfellow et al. (2016). *Deep Learning*, Chapter 6: Backpropagation and Other Differentiation Algorithms.
- Karpathy, A. "Yes, you should understand backprop": http://karpathy.github.io/2016/05/31/ema/
