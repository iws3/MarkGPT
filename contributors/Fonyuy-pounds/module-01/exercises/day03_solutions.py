import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# Create plots directory if it doesn't exist
plots_dir = "c:/Users/the eye informatique/Desktop/ML/AI/MarkGPT/contributors/Fonyuy-pounds/module-01/exercises/plots"
os.makedirs(plots_dir, exist_ok=True)

print("--- E03.1 Matrix Multiplication ---")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 10]])  # Using the user's version with 10
result = np.dot(A, B)
print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")
print(f"Result (A * B):\n{result}")

print("\n--- E03.2 Function Plotting ---")
x = np.linspace(-10, 10, 100) # Reduced number of points

def save_plot(filename, x, y, title, label, color):
    print(f"Starting plot: {title}")
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=label, color=color)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    path = os.path.join(plots_dir, filename)
    print(f"Saving to {path}")
    plt.savefig(path)
    plt.close()
    print(f"Finished {filename}")

# 1. f(x) = x^2
save_plot("squared_function.png", x, x**2, "Squared Function", "f(x) = x²", "blue")

# 2. f(x) = sin(x)
save_plot("sine_function.png", x, np.sin(x), "Sine Function", "f(x) = sin(x)", "green")

# 3. f(x) = 1/(1+e^-x) (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

save_plot("sigmoid_function.png", x, sigmoid(x), "Sigmoid Function", "f(x) = Sigmoid", "red")

print("\n--- E03.3 Probability Puzzle ---")
p_the = 0.3
p_dog = 0.1
p_barked = 0.05
p_sentence = p_the * p_dog * p_barked
print(f"P('the dog barked') = {p_the} * {p_dog} * {p_barked} = {p_sentence:.5f}")
print("\n--- ALL DONE ---")
