#!/usr/bin/env python3
"""
Script to add educational content to module-1.1 READMEs and commit each paragraph separately.
"""
import os
import subprocess
import re

# Define content for each lesson
CONTENT = {
    "01-linear_regression": [
        {
            "section": "Mathematics and Cost Function Derivation",
            "content": "The mathematical foundation of linear regression is built on minimizing the cost function, typically defined as Mean Squared Error (MSE). For a dataset with $m$ samples, the cost function is: $$J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2$$ where $h_\\theta(x) = \\theta_0 + \\theta_1 x_1 + ... + \\theta_n x_n$ represents the hypothesis function. The goal is to find the parameter vector $\\theta$ that minimizes this cost function. The normal equation provides a closed-form solution: $$\\theta = (X^T X)^{-1} X^T y$$ where $X$ is the design matrix augmented with a column of ones, and $y$ is the target vector. This approach is computationally efficient for small to medium-sized datasets but becomes impractical when $n$ (number of features) is very large due to the computational complexity of matrix inversion being $O(n^3)$."
        },
        {
            "section": "Gradient Descent Optimization",
            "content": "Gradient descent is an iterative optimization algorithm that updates parameters by moving in the direction of steepest descent. The update rule is: $$\\theta_j := \\theta_j - \\alpha \\frac{\\partial J(\\theta)}{\\partial \\theta_j}$$ where $\\alpha$ is the learning rate that controls the step size. For linear regression, the partial derivative of the cost function with respect to parameter $\\theta_j$ is: $$\\frac{\\partial J(\\theta)}{\\partial \\theta_j} = \\frac{1}{m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$. There are three variants: Batch Gradient Descent (uses entire dataset), Stochastic Gradient Descent (uses one sample), and Mini-batch Gradient Descent (uses subset of samples). The choice of learning rate $\\alpha$ is critical—too small leads to slow convergence, while too large may cause divergence. Feature scaling (normalization or standardization) dramatically improves convergence speed of gradient descent."
        },
        {
            "section": "Implementation Considerations",
            "content": "When implementing linear regression in practice, several considerations enhance model performance and reliability. Feature engineering plays a crucial role: creating polynomial features, interaction terms, or domain-specific transformations can capture non-linear relationships while maintaining linear regression's interpretability. Handling missing data requires careful attention—options include deletion, mean imputation, or more sophisticated methods like K-NN imputation. Outliers can disproportionately affect linear regression due to the quadratic penalty in MSE; robust regression techniques using loss functions like Huber loss are alternatives when outliers are present. Multicollinearity (high correlation between features) causes numerical instability in the normal equation and inflated coefficients; solutions include removing correlated features, using regularization techniques, or applying Principal Component Analysis (PCA)."
        },
        {
            "section": "Evaluation Metrics and Model Assessment",
            "content": "Beyond the cost function, multiple metrics evaluate linear regression model performance for different purposes. Mean Absolute Error (MAE) = $\\frac{1}{m}\\sum_{i=1}^{m}|y^{(i)} - \\hat{y}^{(i)}|$ is more robust to outliers than MSE. The coefficient of determination, $R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}$, measures the proportion of variance explained by the model (where $SS_{res} = \\sum(y^{(i)} - \\hat{y}^{(i)})^2$ and $SS_{tot} = \\sum(y^{(i)} - \\bar{y})^2$). Root Mean Squared Error (RMSE) is the square root of MSE, making it interpretable in the original units. Cross-validation techniques (k-fold, leave-one-out) provide unbiased estimates of model generalization error and help detect overfitting. Residual analysis—examining $\\epsilon^{(i)} = y^{(i)} - \\hat{y}^{(i)}$—verifies model assumptions: residuals should be normally distributed, have zero mean, constant variance, and show no pattern with predictions."
        }
    ],
    "02-logistic_regression": [
        {
            "section": "Sigmoid Function and Probability Modeling",
            "content": "Logistic regression extends linear regression to classification problems by applying a sigmoid (logistic) function to transform continuous outputs into probabilities bounded between 0 and 1. The sigmoid function is defined as: $$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$ where $z = \\theta_0 + \\theta_1 x_1 + ... + \\theta_n x_n$. This transformation ensures that predictions are valid probabilities. The hypothesis becomes: $$h_\\theta(x) = \\sigma(\\theta^T x) = P(y=1|x;\\theta)$$, which represents the probability that an instance belongs to the positive class. The decision boundary occurs at $P = 0.5$, corresponding to $z = 0$. In binary classification, instances with $h_\\theta(x) \\geq 0.5$ are classified as class 1, while those with $h_\\theta(x) < 0.5$ are classified as class 0. The beauty of logistic regression lies in its probabilistic interpretation: the output directly gives confidence in predictions."
        },
        {
            "section": "Cost Function and Maximum Likelihood Estimation",
            "content": "Unlike linear regression which uses squared error, logistic regression employs the log loss (cross-entropy) cost function derived from maximum likelihood estimation. The cost function for a single training example is: $$J(\\theta) = -[y \\log(h_\\theta(x)) + (1-y) \\log(1-h_\\theta(x))]$$ where $y \\in \\{0, 1\\}$ is the binary label. For the entire training set: $$J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} [y^{(i)} \\log(h_\\theta(x^{(i)})) + (1-y^{(i)}) \\log(1-h_\\theta(x^{(i)}))]$$. This cost function has desirable properties: when the model's prediction is correct and confident, the cost is near zero; when wrong or uncertain, the cost is large. The cost function is convex, guaranteeing that gradient descent will find the global minimum. There is no closed-form solution for logistic regression, so gradient descent (or other optimization algorithms like Newton's method) must be used iteratively."
        },
        {
            "section": "Multiclass Classification Techniques",
            "content": "While binary logistic regression handles two-class problems, multiclass classification with $k > 2$ classes requires extensions. The One-vs-Rest (OvR) approach trains $k$ separate binary classifiers, each distinguishing one class from all others. For a new instance, predictions from all classifiers are obtained, and the class with the highest probability is selected: $$\\hat{y} = \\arg\\max_i h_\\theta^{(i)}(x)$$. Alternatively, the One-vs-One (OvO) method trains $\\binom{k}{2}$ binary classifiers, one for each pair of classes, using more classifiers but potentially better separation. The Softmax regression (multinomial logistic regression) generalizes logistic regression directly to multiclass problems, modeling the probability distribution across all classes using the softmax function: $$P(y=j|x;\\theta) = \\frac{e^{\\theta_j^T x}}{\\sum_{l=1}^{k} e^{\\theta_l^T x}}$$. Softmax is preferred in neural networks and provides a principled probabilistic framework."
        },
        {
            "section": "Regularization and Overfitting Prevention",
            "content": "Logistic regression, like linear regression, can overfit training data, especially with high-dimensional features or limited training samples. L2 regularization (Ridge) adds a penalty term: $$J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} [y^{(i)} \\log(h_\\theta(x^{(i)})) + (1-y^{(i)}) \\log(1-h_\\theta(x^{(i)}))] + \\frac{\\lambda}{2m} \\sum_{j=1}^{n} \\theta_j^2$$. L1 regularization (Lasso) encourages sparsity (zero coefficients): $$J(\\theta) = ... + \\frac{\\lambda}{m} \\sum_{j=1}^{n} |\\theta_j|$$. The regularization parameter $\\lambda$ controls the trade-off between fitting the data and keeping coefficients small. Cross-validation helps select optimal $\\lambda$. Elastic Net combines L1 and L2 penalties. Regularization improves generalization, prevents coefficient blow-up from multicollinearity, and produces interpretable models by reducing feature importance to zero when appropriate."
        }
    ],
}

def add_content_and_commit(lesson_dir, paragraphs):
    """Add content paragraphs one at a time with separate commits."""
    readme_path = os.path.join(lesson_dir, "README.md")
    
    if not os.path.exists(readme_path):
        print(f"README not found: {readme_path}")
        return
    
    for idx, para in enumerate(paragraphs, 1):
        # Read current file
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find insertion point (before the final ---)
        match = re.search(r'(---\s*$)', content, re.MULTILINE)
        if not match:
            print(f"Could not find insertion point in {readme_path}")
            return
        
        insertion_point = match.start()
        
        # Insert new section
        new_section = f"\n## {para['section']}\n\n{para['content']}\n"
        updated_content = content[:insertion_point] + new_section + content[insertion_point:]
        
        # Write updated content
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        # Commit
        lesson_name = os.path.basename(lesson_dir)
        commit_msg = f"Add: {para['section']} to {lesson_name} ({idx}/{len(paragraphs)})"
        
        try:
            subprocess.run(['git', 'add', readme_path], capture_output=True, text=True, check=True)
            subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True, text=True, check=True)
            print(f"✓ {commit_msg}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to commit: {e}")

def main():
    """Process all lessons."""
    os.chdir('c:\\Users\\THE EYE INFORMATIQUE\\OneDrive\\Desktop\\All\\MarkGPT-LLM-Curriculum\\MarkGPT-LLM-Curriculum')
    
    base_dir = 'modules/module-1.1/1_supervised_learning'
    
    for lesson_key, paragraphs in CONTENT.items():
        lesson_path = os.path.join(base_dir, lesson_key)
        print(f"\nProcessing {lesson_key}...")
        add_content_and_commit(lesson_path, paragraphs)

if __name__ == '__main__':
    main()
