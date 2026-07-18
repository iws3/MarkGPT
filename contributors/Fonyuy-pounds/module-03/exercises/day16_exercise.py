"""
Day 16 Exercise: Loss Functions & Optimization

Module: Module 03 - Neural Networks from Scratch
Day: 16/60

Topics:
- MSE Loss Function
- Cross-Entropy Loss Function
- SGD Optimizer
- SGD with Momentum
- Adam Optimizer Implementation
- Loss landscape visualization

Instructions:
1. Implement each loss function from scratch
2. Implement each optimizer from scratch
3. Compare optimizers on same problem
4. Plot loss curves
5. Document findings in day16_journal.md
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os

# ============================================================================
# PART 1: LOSS FUNCTIONS
# ============================================================================

def mse_loss(y_pred, y_true):
    """
    Mean Squared Error Loss
    
    Formula: MSE = (1/n) * Σ(y_pred - y_true)^2
    
    Args:
        y_pred: Predicted values (numpy array)
        y_true: True values (numpy array)
    
    Returns:
        Scalar loss value
    """
    return np.mean((y_pred - y_true) ** 2)


def cross_entropy_loss(y_pred, y_true):
    """
    Cross-Entropy Loss (softmax cross-entropy)
    
    Formula: CE = -Σ(y_true * log(y_pred))
    
    Args:
        y_pred: Predicted probabilities (numpy array, shape: [batch_size, num_classes])
        y_true: True labels one-hot encoded (numpy array, shape: [batch_size, num_classes])
    
    Returns:
        Scalar loss value
    """
    epsilon = 1e-15  # Prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def binary_cross_entropy_loss(y_pred, y_true):
    """
    Binary Cross-Entropy Loss
    
    Formula: BCE = -(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
    
    Args:
        y_pred: Predicted probabilities (numpy array, values between 0 and 1)
        y_true: True binary labels (numpy array, 0 or 1)
    
    Returns:
        Scalar loss value
    """
    epsilon = 1e-15  # Prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ============================================================================
# PART 2: SIMPLE TEST FUNCTION
# ============================================================================

def f(x):
    """
    Test function: f(x) = x^4 - 4x^2 + x
    Used for optimization visualization
    """
    return x**4 - 4*x**2 + x


def df(x):
    """
    Derivative of f(x)
    f'(x) = 4x^3 - 8x + 1
    """
    return 4*x**3 - 8*x + 1


# ============================================================================
# PART 3: OPTIMIZERS
# ============================================================================

class SGD:
    """
    Stochastic Gradient Descent Optimizer
    
    Update rule: w = w - lr * gradient
    """
    
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.history = []
    
    def step(self, x, gradient):
        """
        Single optimization step
        
        Args:
            x: Current parameter value
            gradient: Gradient at current point
        
        Returns:
            Updated parameter value
        """
        x = x - self.lr * gradient
        self.history.append(x)
        return x


class SGD_Momentum:
    """
    SGD with Momentum Optimizer
    
    Uses velocity accumulation to speed up convergence
    
    Update rules:
        v = beta * v + gradient
        w = w - lr * v
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = 0
        self.history = []
    
    def step(self, x, gradient):
        """
        Single optimization step with momentum
        
        Args:
            x: Current parameter value
            gradient: Gradient at current point
        
        Returns:
            Updated parameter value
        """
        self.velocity = self.momentum * self.velocity + gradient
        x = x - self.lr * self.velocity
        self.history.append(x)
        return x


class Adam:
    """
    Adaptive Moment Estimation (Adam) Optimizer
    
    Combines momentum with adaptive learning rates
    
    Update rules:
        m = beta1 * m + (1 - beta1) * gradient         # First moment (mean)
        v = beta2 * v + (1 - beta2) * gradient^2       # Second moment (variance)
        m_hat = m / (1 - beta1^t)                       # Bias correction
        v_hat = v / (1 - beta2^t)
        w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0  # First moment
        self.v = 0  # Second moment
        self.t = 0  # Timestep
        self.history = []
    
    def step(self, x, gradient):
        """
        Single optimization step with Adam
        
        Args:
            x: Current parameter value
            gradient: Gradient at current point
        
        Returns:
            Updated parameter value
        """
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        x = x - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.history.append(x)
        return x


# ============================================================================
# PART 4: OPTIMIZATION COMPARISON
# ============================================================================

def optimize_with_method(start_x, optimizer, num_iterations=100):
    """
    Run optimization algorithm for multiple iterations
    
    Args:
        start_x: Starting point
        optimizer: Optimizer object (SGD, SGD_Momentum, or Adam)
        num_iterations: Number of iterations to run
    
    Returns:
        List of x values visited, List of f(x) values visited
    """
    x = start_x
    x_history = [x]
    f_history = [f(x)]
    
    for _ in range(num_iterations):
        gradient = df(x)
        x = optimizer.step(x, gradient)
        
        x_history.append(x)
        f_history.append(f(x))
        
        # Prevent overflow
        if abs(x) > 1e5:
            break
    
    return x_history, f_history


def main():
    """Main exercise driver"""
    
    print("=" * 70)
    print("DAY 16 EXERCISE: Loss Functions & Optimization")
    print("=" * 70)
    
    # ====================================================================
    # EXERCISE 1: Test Loss Functions
    # ====================================================================
    print("\n[EXERCISE 1] Testing Loss Functions")
    print("-" * 70)
    
    # Create synthetic data
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    
    # Test MSE Loss
    try:
        mse = mse_loss(y_pred, y_true)
        print(f"\nMSE Loss: {mse:.4f}")
    except:
        print("\n❌ MSE Loss not implemented yet")
    
    # Test Binary Cross-Entropy
    try:
        bce = binary_cross_entropy_loss(y_pred, y_true)
        print(f"Binary Cross-Entropy: {bce:.4f}")
    except:
        print("❌ Binary Cross-Entropy not implemented yet")
    
    # ====================================================================
    # EXERCISE 2: Compare Optimizers
    # ====================================================================
    print("\n[EXERCISE 2] Comparing Optimizers")
    print("-" * 70)
    
    start_x = 0.0
    num_iterations = 100
    
    # Initialize optimizers
    sgd = SGD(learning_rate=0.01)
    sgd_momentum = SGD_Momentum(learning_rate=0.01, momentum=0.9)
    adam = Adam(learning_rate=0.05)
    
    print(f"Starting point: x = {start_x}")
    print(f"Target: minimize f(x) = x^4 - 4x^2 + x")
    print(f"Number of iterations: {num_iterations}\n")
    
    # Run optimizations
    try:
        sgd_x, sgd_f = optimize_with_method(start_x, sgd, num_iterations)
        print(f"SGD: Final x = {sgd_x[-1]:.4f}, f(x) = {sgd_f[-1]:.4f}, steps = {len(sgd_x)}")
    except:
        print("❌ SGD not implemented yet")
        sgd_x, sgd_f = None, None
    
    try:
        sgd_m_x, sgd_m_f = optimize_with_method(start_x, sgd_momentum, num_iterations)
        print(f"SGD+Momentum: Final x = {sgd_m_x[-1]:.4f}, f(x) = {sgd_m_f[-1]:.4f}, steps = {len(sgd_m_x)}")
    except:
        print("❌ SGD+Momentum not implemented yet")
        sgd_m_x, sgd_m_f = None, None
    
    try:
        adam_x, adam_f = optimize_with_method(start_x, adam, num_iterations)
        print(f"Adam: Final x = {adam_x[-1]:.4f}, f(x) = {adam_f[-1]:.4f}, steps = {len(adam_x)}")
    except:
        print("❌ Adam not implemented yet")
        adam_x, adam_f = None, None
    
    # ====================================================================
    # EXERCISE 3: Visualization
    # ====================================================================
    print("\n[EXERCISE 3] Plotting Results")
    print("-" * 70)
    
    if sgd_f is not None and sgd_m_f is not None and adam_f is not None:
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Loss Curves
        plt.subplot(1, 2, 1)
        plt.plot(sgd_f, marker='o', label='SGD', alpha=0.7)
        plt.plot(sgd_m_f, marker='s', label='SGD+Momentum', alpha=0.7)
        plt.plot(adam_f, marker='^', label='Adam', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('f(x)')
        plt.title('Optimizer Comparison: Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Trajectory on function landscape
        plt.subplot(1, 2, 2)
        x_vals = np.linspace(-3, 3, 400)
        y_vals = f(x_vals)
        plt.plot(x_vals, y_vals, 'k-', linewidth=2, label='f(x)')
        
        plt.plot(sgd_x, sgd_f, marker='o', label='SGD', alpha=0.7)
        plt.plot(sgd_m_x, sgd_m_f, marker='s', label='SGD+Momentum', alpha=0.7)
        plt.plot(adam_x, adam_f, marker='^', label='Adam', alpha=0.7)
        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Optimization Trajectories')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(os.getcwd(), 'day16_optimizer_comparison.png')
        plt.savefig(output_path, dpi=150)
        print(f"✓ Plot saved to {output_path}")
    else:
        print("⚠ Cannot plot: Not all optimizers implemented")
    
    # ====================================================================
    # EXERCISE 4: Reflection Questions
    # ====================================================================
    print("\n[EXERCISE 4] Reflection Questions")
    print("-" * 70)
    print("Answer these questions in your day16_journal.md:")
    print("1. Why does Adam converge faster than SGD?")
    print("2. What is the advantage of momentum in optimization?")
    print("3. When would you use MSE loss vs Cross-Entropy loss?")
    print("4. How would you adapt these optimizers for a multi-dimensional problem?")
    
    print("\n" + "=" * 70)
    print("Day 16 Exercise Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
