import time
import numpy as np

def manual_matrix_multiply(A, B):
    """
    Multiply matrix A (m x n) and matrix B (n x p) without using numpy.dot or @.
    Return a new matrix C (m x p) as a nested list.
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0 for _ in range(p)] for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def numpy_matrix_multiply(A, B):
    """
    Multiply matrix A and matrix B using numpy.
    """
    return A @ B

def softmax(x):
    """
    Implement the softmax function mathematically:
    softmax(x_i) = exp(x_i) / sum(exp(x_j))
    """
    # Using np.exp and subtracting max for numerical stability so exp doesn't blow up
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def main():
    print("Welcome to Day 7: NumPy & Vectorization")
    
    # Generate random matrices for benchmarking
    size = 100
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    A_list = A.tolist()
    B_list = B.tolist()
    
    # 1. Benchmark nested loop multiplication
    start = time.time()
    C_manual = manual_matrix_multiply(A_list, B_list)
    manual_time = time.time() - start
    print(f"Manual multiplication time: {manual_time:.5f} seconds")
    
    # 2. Benchmark numpy multiplication
    start = time.time()
    C_numpy = numpy_matrix_multiply(A, B)
    numpy_time = time.time() - start
    print(f"NumPy multiplication time: {numpy_time:.5f} seconds")
    
    # 3. Test softmax
    logits = np.array([2.0, 1.0, 0.1])
    print("Logits:", logits)
    print("Softmax output:", softmax(logits))
    print("Sum of softmax:", np.sum(softmax(logits)))

if __name__ == "__main__":
    main()
