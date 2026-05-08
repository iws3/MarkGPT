"""
Day 17 Exercise: Regularization & Generalization

Module: Module 03 - Neural Networks from Scratch
Day: 17/60

Topics:
- Overfitting vs. Underfitting
- L1 & L2 Regularization
- Dropout Mechanism
- Batch Normalization
- Early Stopping
- Bias-Variance Tradeoff

Instructions:
1. Implement regularization techniques from scratch
2. Compare regularized vs. unregularized models
3. Visualize train/validation loss divergence
4. Measure the effect of dropout and batch norm
5. Document findings in day17_journal.md
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================================
# PART 1: GENERATE SYNTHETIC DATA (Overfitting-Prone)
# ============================================================================

def generate_data(n_samples=100, n_features=50, noise=0.5, random_seed=42):
    """
    Generate synthetic data prone to overfitting.
    
    We create high-dimensional data with limited samples, classic overfitting scenario.
    
    Args:
        n_samples: Number of data points
        n_features: Dimensionality
        noise: Gaussian noise level
        random_seed: For reproducibility
    
    Returns:
        X_train, y_train, X_val, y_val
    """
    np.random.seed(random_seed)
    
    # True underlying relationship: only first 5 features matter
    true_weights = np.zeros(n_features)
    true_weights[:5] = [2.0, -1.5, 0.8, 1.2, -0.9]
    
    # Generate data
    X = np.random.randn(n_samples, n_features)
    y = X @ true_weights + np.random.randn(n_samples) * noise
    
    # Split into train/val
    n_train = int(0.7 * n_samples)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    return X_train, y_train, X_val, y_val


# ============================================================================
# PART 2: REGULARIZATION IMPLEMENTATIONS
# ============================================================================

class LinearRegressionWithRegularization:
    """
    Linear regression with L1, L2, or Elastic Net regularization.
    
    Loss = MSE + λ * (α * L1 + (1-α) * L2)
    where α ∈ [0, 1]: α=1 → L1 (Lasso), α=0 → L2 (Ridge), 0<α<1 → Elastic Net
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=100, lambda_reg=0.0, alpha=0.5):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.alpha = alpha  # 0=L2, 1=L1, between=Elastic Net
        self.weights = None
        self.bias = None
        self.train_loss_history = []
        self.val_loss_history = []
    
    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fit the model using gradient descent with regularization.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (for monitoring)
        """
        n_samples, n_features = X_train.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        for iteration in range(self.n_iterations):
            # Forward pass
            y_pred_train = X_train @ self.weights + self.bias
            y_pred_val = X_val @ self.weights + self.bias
            
            # Loss computation
            train_mse = np.mean((y_pred_train - y_train) ** 2)
            val_mse = np.mean((y_pred_val - y_val) ** 2)
            
            # Regularization penalty
            l1_penalty = np.sum(np.abs(self.weights))
            l2_penalty = np.sum(self.weights ** 2)
            reg_penalty = self.lambda_reg * (self.alpha * l1_penalty + (1 - self.alpha) * l2_penalty)
            
            train_loss = train_mse + reg_penalty
            val_loss = val_mse + self.lambda_reg * (self.alpha * l1_penalty + (1 - self.alpha) * l2_penalty)
            
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            
            # Gradient computation
            error = y_pred_train - y_train
            dw = (2 / n_samples) * X_train.T @ error
            db = (2 / n_samples) * np.sum(error)
            
            # Add regularization gradient
            if self.alpha == 1:  # L1
                dw += self.lambda_reg * np.sign(self.weights)
            else:  # L2 or Elastic Net
                dw += self.lambda_reg * (1 - self.alpha) * 2 * self.weights
                if self.alpha > 0:
                    dw += self.lambda_reg * self.alpha * np.sign(self.weights)
            
            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


# ============================================================================
# PART 3: DROPOUT IMPLEMENTATION
# ============================================================================

class Dropout:
    """
    Dropout layer: randomly disables neurons during training.
    
    During training: keep each neuron with probability p, scale by 1/p
    During inference: use all neurons (no scaling needed due to training scaling)
    """
    
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob
    
    def apply(self, x, training=True):
        """
        Apply dropout.
        
        Args:
            x: Input tensor
            training: If True, apply dropout; if False, return as-is
        
        Returns:
            x_dropout: Dropped-out tensor
        """
        if not training:
            return x
        
        # Create binary mask
        mask = np.random.binomial(1, self.keep_prob, x.shape)
        # Scale by 1/keep_prob to maintain expected value
        return x * mask / self.keep_prob


# ============================================================================
# PART 4: BATCH NORMALIZATION IMPLEMENTATION
# ============================================================================

class BatchNormalization:
    """
    Batch normalization: normalize layer inputs per minibatch.
    
    Formulas:
    - x_normalized = (x - mean) / sqrt(var + epsilon)
    - x_scaled = gamma * x_normalized + beta
    
    During training: use minibatch statistics
    During inference: use running averages
    """
    
    def __init__(self, n_features, momentum=0.9, epsilon=1e-5):
        self.n_features = n_features
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        
        # Running statistics for inference
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
    
    def forward(self, x, training=True):
        """
        Forward pass with optional batch norm.
        
        Args:
            x: Input minibatch (n_samples, n_features)
            training: If True, use minibatch stats; else use running stats
        
        Returns:
            x_normalized: Normalized output
        """
        if training:
            # Compute minibatch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            # Use running statistics
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        # Normalize
        x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
        
        # Scale and shift
        x_scaled = self.gamma * x_normalized + self.beta
        
        return x_scaled


# ============================================================================
# PART 5: EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """
    Stop training when validation loss plateaus (doesn't improve).
    """
    
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.wait_count = 0
        self.should_stop = False
    
    def check(self, val_loss):
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
        
        Returns:
            Boolean: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.should_stop = True
        
        return self.should_stop


# ============================================================================
# PART 6: MAIN EXERCISE DRIVER
# ============================================================================

def main():
    """Main exercise driver"""
    
    print("=" * 70)
    print("DAY 17 EXERCISE: Regularization & Generalization")
    print("=" * 70)
    
    # Generate data
    print("\n[SETUP] Generating overfitting-prone dataset...")
    X_train, y_train, X_val, y_val = generate_data(n_samples=100, n_features=50)
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # ====================================================================
    # EXERCISE 1: Observe Overfitting
    # ====================================================================
    print("\n[EXERCISE 1] Observing Overfitting")
    print("-" * 70)
    
    model_noref = LinearRegressionWithRegularization(learning_rate=0.01, n_iterations=200, lambda_reg=0.0)
    model_noref.fit(X_train, y_train, X_val, y_val)
    
    y_pred_train = model_noref.predict(X_train)
    y_pred_val = model_noref.predict(X_val)
    
    train_mse = model_noref.mse(y_train, y_pred_train)
    val_mse = model_noref.mse(y_val, y_pred_val)
    
    print(f"Unregularized Model:")
    print(f"  Training MSE: {train_mse:.4f}")
    print(f"  Validation MSE: {val_mse:.4f}")
    print(f"  Overfitting Ratio: {val_mse / train_mse:.2f}x worse on validation! ⚠️")
    
    # ====================================================================
    # EXERCISE 2: L1 and L2 Regularization
    # ====================================================================
    print("\n[EXERCISE 2] L1 and L2 Regularization")
    print("-" * 70)
    
    results = {}
    
    # No regularization (baseline)
    model = LinearRegressionWithRegularization(lambda_reg=0.0)
    model.fit(X_train, y_train, X_val, y_val)
    results['No Regularization'] = {
        'train': model.mse(y_train, model.predict(X_train)),
        'val': model.mse(y_val, model.predict(X_val)),
        'sparsity': np.sum(np.abs(model.weights) < 0.01) / len(model.weights)
    }
    
    # L2 (Ridge)
    model = LinearRegressionWithRegularization(lambda_reg=0.01, alpha=0.0)
    model.fit(X_train, y_train, X_val, y_val)
    results['L2 (λ=0.01)'] = {
        'train': model.mse(y_train, model.predict(X_train)),
        'val': model.mse(y_val, model.predict(X_val)),
        'sparsity': np.sum(np.abs(model.weights) < 0.01) / len(model.weights)
    }
    
    # L1 (Lasso)
    model = LinearRegressionWithRegularization(lambda_reg=0.01, alpha=1.0)
    model.fit(X_train, y_train, X_val, y_val)
    results['L1 (λ=0.01)'] = {
        'train': model.mse(y_train, model.predict(X_train)),
        'val': model.mse(y_val, model.predict(X_val)),
        'sparsity': np.sum(np.abs(model.weights) < 0.01) / len(model.weights)
    }
    
    # Elastic Net
    model = LinearRegressionWithRegularization(lambda_reg=0.01, alpha=0.5)
    model.fit(X_train, y_train, X_val, y_val)
    results['Elastic Net (λ=0.01)'] = {
        'train': model.mse(y_train, model.predict(X_train)),
        'val': model.mse(y_val, model.predict(X_val)),
        'sparsity': np.sum(np.abs(model.weights) < 0.01) / len(model.weights)
    }
    
    print("\nRegularization Comparison:")
    print(f"{'Method':<25} {'Train MSE':<12} {'Val MSE':<12} {'Sparsity':<10}")
    print("-" * 60)
    for method, metrics in results.items():
        print(f"{method:<25} {metrics['train']:<12.4f} {metrics['val']:<12.4f} {metrics['sparsity']:<10.1%}")
    
    # ====================================================================
    # EXERCISE 3: Dropout
    # ====================================================================
    print("\n[EXERCISE 3] Dropout Regularization")
    print("-" * 70)
    
    dropout = Dropout(keep_prob=0.5)
    
    # Create synthetic data for demonstration
    X_sample = np.random.randn(32, 100)  # Minibatch of 32
    
    print(f"Original data shape: {X_sample.shape}")
    print(f"Original mean activation: {np.mean(X_sample):.4f}")
    
    # Apply dropout at training time
    X_dropout = dropout.apply(X_sample, training=True)
    print(f"After dropout (training): mean = {np.mean(X_dropout):.4f}")
    print(f"Dropout fraction: {np.sum(X_dropout == 0) / X_dropout.size:.1%}")
    
    # At test time, no dropout
    X_test = dropout.apply(X_sample, training=False)
    print(f"After dropout (testing): mean = {np.mean(X_test):.4f} (unchanged)")
    
    print("\n✓ Dropout properly maintains expected activation magnitude")
    
    # ====================================================================
    # EXERCISE 4: Batch Normalization
    # ====================================================================
    print("\n[EXERCISE 4] Batch Normalization Impact")
    print("-" * 70)
    
    bn = BatchNormalization(n_features=50)
    
    print(f"Input minibatch mean: {np.mean(X_train):.4f}, std: {np.std(X_train):.4f}")
    
    X_normalized = bn.forward(X_train, training=True)
    print(f"After BN mean: {np.mean(X_normalized):.4f}, std: {np.std(X_normalized):.4f}")
    print(f"→ Normalized to ~0 mean and ~1 std")
    
    # ====================================================================
    # VISUALIZATION
    # ====================================================================
    print("\n[EXERCISE 5] Plotting Results")
    print("-" * 70)
    
    # Re-train models for visualization
    model_base = LinearRegressionWithRegularization(lambda_reg=0.0, n_iterations=200)
    model_base.fit(X_train, y_train, X_val, y_val)
    
    model_l2 = LinearRegressionWithRegularization(lambda_reg=0.01, alpha=0.0, n_iterations=200)
    model_l2.fit(X_train, y_train, X_val, y_val)
    
    # Create plots
    plt.figure(figsize=(14, 5))
    
    # Plot 1: Train vs Validation Loss (No Regularization)
    plt.subplot(1, 2, 1)
    plt.plot(model_base.train_loss_history, label='Training Loss', linewidth=2)
    plt.plot(model_base.val_loss_history, label='Validation Loss', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Overfitting: No Regularization\n(Validation diverges from training)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Train vs Validation Loss (With L2 Regularization)
    plt.subplot(1, 2, 2)
    plt.plot(model_l2.train_loss_history, label='Training Loss', linewidth=2)
    plt.plot(model_l2.val_loss_history, label='Validation Loss', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('With L2 Regularization (λ=0.01)\n(Train and validation stay close)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(os.getcwd(), 'day17_regularization_comparison.png')
    plt.savefig(output_path, dpi=150)
    print(f"✓ Plot saved to {output_path}")
    
    # ====================================================================
    # REFLECTION QUESTIONS
    # ====================================================================
    print("\n[EXERCISE 6] Reflection Questions")
    print("-" * 70)
    print("Answer these questions in your day17_journal.md:")
    print("1. What's the difference between overfitting and underfitting?")
    print("2. How does L1 differ from L2 regularization in their effects?")
    print("3. Why is dropout effective even though it seems counterintuitive?")
    print("4. When would you use batch normalization vs other regularization?")
    print("5. How would you choose regularization strength λ in practice?")
    
    print("\n" + "=" * 70)
    print("Day 17 Exercise Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
