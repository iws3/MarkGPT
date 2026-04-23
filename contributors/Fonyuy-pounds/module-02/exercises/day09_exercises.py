import numpy as np
import matplotlib.pyplot as plt

def pca_from_scratch(X, n_components=2):
    """
    Perform PCA from scratch on data X.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of components to keep
    
    Returns:
        X_projected: Data projected onto the top components
        components: Top principal components (eigenvectors)
    """
    # 1. Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # 2. Compute the Covariance Matrix
    # Using rowvar=False because rows are samples
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # 3. Compute Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort eigenvalues and keep the top n_components
    # eigh returns sorted in ascending order, so we reverse it
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    top_components = eigenvectors[:, :n_components]
    
    # 5. Project the data
    X_projected = np.dot(X_centered, top_components)
    
    return X_projected, top_components

def main():
    print("Welcome to Day 9: Linear Algebra Deep Dive")
    
    # Generate some toy data
    np.random.seed(42)
    X = np.random.rand(100, 5) # 100 samples, 5 features
    
    # Run PCA
    X_projected, components = pca_from_scratch(X, n_components=2)
    
    print(f"Original shape: {X.shape}")
    print(f"Projected shape: {X_projected.shape}")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.7, c='blue', edgecolors='k')
    plt.title("PCA Projection from Scratch (2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.savefig('pca_projection.png')
    print("Saved pca_projection.png")

if __name__ == "__main__":
    main()
