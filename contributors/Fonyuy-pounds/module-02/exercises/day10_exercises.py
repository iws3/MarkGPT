import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def f(x):
    return x**4 - 4*x**2 + x

def df(x):
    return 4*x**3 - 8*x + 1

def gradient_descent(start_x, lr, num_iterations=100):
    x_history = [start_x]
    x = start_x
    for _ in range(num_iterations):
        grad = df(x)
        x = x - lr * grad
        x_history.append(x)
        # Prevent overflow for diverging learning rates
        if abs(x) > 1e5:
            break
    return np.array(x_history)

def main():
    x_vals = np.linspace(-3, 3, 400)
    y_vals = f(x_vals)
    
    start_x = 0.0  # Let's start from x=0
    learning_rates = [0.01, 0.1, 0.25]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(x) = x^4 - 4x^2 + x', color='black', linewidth=2)
    
    colors = ['blue', 'green', 'red']
    
    for lr, color in zip(learning_rates, colors):
        history = gradient_descent(start_x, lr)
        y_history = f(history)
        
        plt.plot(history, y_history, marker='o', markersize=4, linestyle='-', alpha=0.7, 
                 label=f'LR = {lr}', color=color)
        
        print(f"LR={lr}: Converged to x={history[-1]:.4f} in {len(history)} steps. f(x)={y_history[-1]:.4f}")
        if abs(history[-1]) > 1e4 or np.isnan(history[-1]):
            print(f"  -> Diverged!")
        elif len(history) == 101: # max iterations + start
             if abs(history[-1] - history[-2]) < 1e-5:
                 print(f"  -> Converged successfully.")
             else:
                 print(f"  -> Converged slowly.")
                 
    plt.title('Gradient Descent Trajectory: f(x) = x^4 - 4x^2 + x')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(os.path.dirname(__file__), 'gradient_descent_trajectory.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
