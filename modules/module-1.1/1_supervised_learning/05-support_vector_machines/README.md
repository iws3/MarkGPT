# Support Vector Machines (SVM)

## Fundamentals

Support Vector Machines (SVM) are powerful supervised learning algorithms that find the optimal hyperplane maximizing the margin between different classes in classification tasks. SVMs can handle both linear and non-linear classification through various kernel functions, and they are particularly effective in high-dimensional spaces. The algorithm focuses on support vectors—the critical data points near the decision boundary—making it efficient for sparse problems. SVMs have strong theoretical foundations rooted in statistical learning theory and have been successfully applied to image classification, text classification, bioinformatics, and financial prediction.

## Key Concepts

- **Margin Maximization**: Distance between hyperplane and nearest points
- **Support Vectors**: Critical points defining the decision boundary
- **Kernel Trick**: Non-linear mapping without explicit transformation
- **Soft Margin**: Allowing some misclassification (C parameter)

## Applications

- Image classification
- Text and document classification
- DNA sequence classification
- Financial data classification
- Bioinformatics

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### The Maximum Margin Principle

Support Vector Machines (SVMs) are based on the principle of finding the optimal separating hyperplane that maximizes the margin between classes. The margin is defined as the distance from the hyperplane to the nearest data points of either class. SVMs solve the optimization problem of finding the hyperplane W that maximizes this margin while correctly classifying all training examples. Mathematically, this is formulated as minimizing ||W||² subject to the constraint that y_i(W·x_i + b) ≥ 1 for all training examples. The maximum margin principle provides strong generalization guarantees; a larger margin means the hyperplane is more robust to small perturbations in the data. The data points closest to the hyperplane that define the margin are called support vectors, and only these points are needed to define the decision boundary, making SVMs memory-efficient for prediction.

### Handling Non-Linear Separability with Kernels

Most real-world classification problems are not linearly separable in the input space. SVMs handle this through the kernel trick, which implicitly projects data into a higher-dimensional feature space where linear separation becomes possible. Common kernels include the polynomial kernel K(x, x') = (x·x' + c)^d, the radial basis function (RBF) kernel K(x, x') = exp(-γ||x - x'||²), and the sigmoid kernel. The key insight of the kernel trick is that the computation can be done in the original input space using kernel functions without explicitly computing the high-dimensional projection. The RBF kernel is particularly popular because it can handle complex non-linear patterns effectively without over-parameterization. The kernel parameter γ controls the reach of each training example; small γ leads to smooth boundaries while large γ creates more intricate patterns following individual data points.

### Soft Margin Formulation and Regularization

Strict maximum margin formulation assumes perfect linear (or kernel-induced) separability, which is unrealistic when classes overlap or noise is present. The soft margin formulation introduces slack variables ξ_i that allow some misclassification, leading to the optimization problem: minimize ||W||² + C·Σξ_i subject to y_i(W·x_i + b) ≥ 1 - ξ_i. The regularization parameter C controls the trade-off between maximizing the margin and minimizing classification error. Large C values penalize misclassifications heavily, leading to complex decision boundaries that fit training data closely, while small C values allow more training errors for a larger margin and simpler decision boundaries. Cross-validation is used to select an appropriate C value. The soft margin formulation also applies when using kernels, making SVMs practical for real-world noisy datasets.

### Multi-class Classification and Practical Considerations

SVMs are primarily designed for binary classification, but several strategies extend them to multi-class problems. One-vs-Rest creates |K| binary classifiers, each trained to separate one class from all others. One-vs-One creates |K|(|K|-1)/2 binary classifiers for each pair of classes. The predictions from multiple binary classifiers are combined through voting or probability calibration. In practice, implementing SVMs involves several considerations: feature scaling is critical since SVMs are sensitive to the magnitude of features, kernel selection requires domain knowledge or cross-validation, and training time complexity is O(n²) or O(n³) depending on the implementation, making SVMs less suitable for very large datasets. However, SVMs provide strong theoretical guarantees on generalization and remain competitive for medium-sized datasets, particularly in high-dimensional spaces where the margin principle provides substantial benefits.

### Kernel Selection and Parameter Tuning

Kernel choice dramatically affects SVM performance. Linear kernels suit linearly separable data; they're fast and interpretable (features multiply coefficients). RBF kernels suit non-linear data; they implicitly project to infinite-dimensional space, enabling complex boundaries. Polynomial kernels (degree 2-3) work for specific problems but are less common. Sigmoid kernels (similar to neural networks) are rarely used. In practice, try linear first; if performance is poor, try RBF. The γ (gamma) parameter in RBF controls influence range: small γ (smooth boundaries) considers distant points; large γ (wiggly boundaries) only near points matter. γ too large causes overfitting; too small causes underfitting. The C parameter (regularization strength) balances margin maximization and training error: large C fits training data tightly; small C emphasizes margin. Grid search over (C, γ) via cross-validation finds optimal values. Typically C in {0.1, 1, 10, 100} and γ in {0.001, 0.01, 0.1, 1} are tested.

### Feature Scaling and Numerical Stability

SVMs are significantly affected by feature scaling; features with large ranges dominate distance calculations. StandardScaler centers and scales to zero mean, unit variance. Without scaling, features with range [0, 1000] overshadow features with range [0, 1]. After fitting, scaling parameters (mean, std) must be stored for identical transformation during prediction. This is non-obvious but critical; skip it and predictions are meaningless. Numerical stability matters: RBF kernel computation involves exponentials; extreme feature values cause numerical overflow/underflow. Scaled features are typically in [-3, 3], avoiding numerical issues. SVM implementations use algorithms like SMO (Sequential Minimal Optimization) that work iteratively; they iterate until convergence (dual objective changes less than `tol`). Very tight tolerances increase computation but ensure precision. For large datasets (n > 100k), linear SVMs with SGDClassifier are more practical than kernel SVMs.

### Support Vector Counts and Model Complexity

Support vectors are training points closest to the decision boundary; only these points define the boundary (other points don't matter). The number of support vectors indicates model complexity: fewer SV means simpler, more generalizable model; many SVs indicate complex boundary closely fitting training data. With good hyperparameters (balanced C, appropriate γ), support vectors should be ~1-30% of training samples. Much higher fractions indicate underfitting (C too small) or overfitting (C too large). Examining which samples become support vectors provides insights: outliers often become SVs; samples in overlapping regions between classes become SVs. This information guides data cleaning: removing outliers might reduce SV count, improving generalization. SVs are the only samples needed at prediction time; sparse solutions (few SVs) enable fast prediction. This is advantageous for deployed models; if 5% of training data becomes SVs, prediction is fast.

### Multi-class SVM Strategies

Standard SVM is binary; multi-class requires strategies. One-vs-Rest trains |K| binary classifiers; each separates one class from all others. Predictions combine via voting or probability estimates. One-vs-One trains |K|(|K|-1)/2 binary classifiers for each pair; again voting combines predictions. One-vs-Rest is more efficient (fewer classifiers) but often produces worse probability estimates. One-vs-One is more computation but sometimes generalizes better. In scikit-learn, `SVC(decision_function_shape='ovr')` uses one-vs-rest; default is one-vs-one. For imbalanced multi-class, class weights (`class_weight='balanced'`) help. Strategies differ: one-vs-rest trains each classifier to separate a class from everything. One-vs-one is symmetric (classifier for class A vs B is identical to B vs A). Empirically, both usually perform similarly; one-vs-rest is typical.