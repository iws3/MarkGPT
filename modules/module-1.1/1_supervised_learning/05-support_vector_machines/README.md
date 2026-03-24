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