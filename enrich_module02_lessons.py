#!/usr/bin/env python3
"""
Comprehensive lesson enrichment script for module-02
Adds detailed educational content and creates commits for each paragraph
"""

import os
import subprocess
import json
from datetime import datetime, timedelta

# Define all lessons in module-02
LESSONS = [
    {
        "name": "L07.1_python-for-ml",
        "title": "Python for Machine Learning",
        "content_sections": [
            {
                "section": "Understanding Python Environments",
                "paragraphs": [
                    "Python environments are isolated development spaces that allow you to manage project dependencies independently. When working on machine learning projects, you might need different versions of libraries for different projects. Virtual environments solve this problem by creating self-contained directories with their own Python installations and package libraries. This prevents version conflicts and ensures reproducibility across different machines and development stages.",
                    "Setting up a virtual environment is straightforward: use 'python -m venv env_name' to create a new environment, then activate it with 'source env_name/bin/activate' on Linux/Mac or 'env_name\\Scripts\\activate' on Windows. Once activated, you can install packages with pip, and they'll be isolated to that environment. This practice is essential in professional ML development because it ensures that your code will run consistently regardless of what other packages are installed globally on a system.",
                    "Managing dependencies with requirements.txt files allows you to document all packages your project needs. You can generate this with 'pip freeze > requirements.txt', which captures the exact versions of all installed packages. When sharing your project or deploying it, others can replicate your environment exactly by running 'pip install -r requirements.txt'. This is crucial for collaborative ML projects where reproducibility and consistency are paramount.",
                ]
            },
            {
                "section": "ML-Specific Python Libraries",
                "paragraphs": [
                    "NumPy is the foundation of numerical computing in Python and is essential for machine learning. It provides efficient multidimensional array operations, mathematical functions, and linear algebra capabilities that are far faster than native Python lists. Understanding NumPy arrays is fundamental because most ML libraries like scikit-learn, TensorFlow, and PyTorch are built on top of NumPy's architecture. NumPy arrays are homogeneous (all elements are the same type) and support vectorized operations, meaning you can perform operations on entire arrays without explicit loops.",
                    "Pandas builds on NumPy and provides data structures for data manipulation and analysis. The DataFrame object is essentially a tabular data structure with labeled rows and columns, similar to an Excel spreadsheet or SQL table. Pandas excels at data cleaning, transformation, and exploratory analysis. For ML projects, Pandas is typically used in the data preprocessing phase to handle missing values, normalize features, and prepare data in the format required by ML algorithms.",
                    "Scikit-learn is the primary library for implementing traditional machine learning algorithms in Python. It provides consistent interfaces for classification, regression, clustering, and dimensionality reduction. Scikit-learn's philosophy emphasizes simplicity and consistency: all estimators follow the same API with fit(), predict(), and transform() methods. This makes it excellent for learning, prototyping, and building production systems. It also includes utilities for model evaluation, cross-validation, and hyperparameter tuning.",
                    "TensorFlow and PyTorch are deep learning frameworks that enable you to build and train neural networks. TensorFlow, developed by Google, focuses on production scalability and deployment, with tools for mobile and edge devices. PyTorch, developed by Meta, emphasizes research flexibility with dynamic computation graphs. Both support GPU acceleration, which is essential for training large neural networks. The choice between them often depends on your use case: TensorFlow for production systems, PyTorch for research.",
                ]
            },
            {
                "section": "Python Programming Best Practices for ML",
                "paragraphs": [
                    "Code organization is critical in ML projects. Separate your code into distinct modules: data loading, preprocessing, feature engineering, model training, evaluation, and prediction. This modular approach makes your code more maintainable, testable, and reusable. Each module should have a single responsibility, following the Single Responsibility Principle from software engineering. This structure also facilitates collaboration, as team members can work on different components independently.",
                    "Version control with Git is non-negotiable in professional ML development. Track changes to your code, data preprocessing scripts, and model configurations. Use meaningful commit messages that describe what changed and why. Create branches for experiments while keeping the main branch stable. This allows you to explore new ideas without affecting production code and provides a complete history of how your models evolved.",
                    "Documentation and comments are often overlooked but are essential for ML projects. Document the purpose of each function, the expected input formats, and the model's assumptions. Explain non-obvious ML decisions such as why you chose a particular algorithm or how you handled imbalanced data. Good documentation helps teammates understand your work faster and helps you remember your own reasoning months later when reviewing old code.",
                    "Testing in ML contexts goes beyond unit testing. You need to validate data integrity, check for data leakage, ensure reproducibility through fixed random seeds, and monitor model performance over time. Create tests for edge cases and verify that your preprocessing pipeline works correctly. Consider implementing integration tests that validate the entire ML pipeline from raw data to predictions.",
                ]
            }
        ]
    },
    {
        "name": "L07.2_numpy-arrays",
        "title": "NumPy Arrays and Operations",
        "content_sections": [
            {
                "section": "Fundamentals of NumPy Arrays",
                "paragraphs": [
                    "NumPy arrays are the fundamental data structure for numerical computing in Python. Unlike Python lists which are heterogeneous and store references to objects, NumPy arrays are homogeneous and store primitive data types directly in memory. This design choice makes NumPy arrays significantly more memory-efficient and faster for numerical operations. A NumPy array of 1 million integers uses substantially less memory than a Python list of the same integers, and operations on the array are orders of magnitude faster due to optimized C implementations under the hood.",
                    "Arrays have several key attributes that define their characteristics. The shape attribute indicates the dimensions: a 1D array has shape (n,), a 2D array has shape (m, n), and higher-dimensional arrays follow the same convention. The dtype specifies the data type of elements (int32, float64, complex128, etc.). The size attribute gives the total number of elements. The ndim attribute returns the number of dimensions. Understanding these attributes helps you debug shape mismatches and ensure your data is in the correct format for algorithms.",
                    "Creating arrays can be done in multiple ways depending on your needs. The arange() function creates arrays with evenly spaced values, linspace() creates arrays with a specific number of points in a range, zeros() and ones() create arrays filled with zeros or ones, and random functions generate random arrays. For ML tasks, you often need to create arrays with specific properties: identity matrices for linear algebra operations, arrays of specific shapes for neural network inputs, or random initializations for model weights.",
                ]
            },
            {
                "section": "Array Indexing and Slicing",
                "paragraphs": [
                    "Indexing arrays allows you to access individual elements or groups of elements. For 1D arrays, indexing works like Python lists: arr[0] gets the first element, arr[-1] gets the last. For multidimensional arrays, you use comma-separated indices: arr[0, 1] accesses the element at row 0, column 1. Negative indexing counts from the end, useful for accessing the last elements without knowing the array size. This is particularly helpful in ML when working with batches of data where you might need the last batch without calculating its exact position.",
                    "Slicing extracts contiguous subarrays using the colon notation. arr[start:stop:step] extracts elements from start up to (but not including) stop with the given step. Omitting indices uses defaults: start defaults to 0, stop defaults to the array length, and step defaults to 1. Slicing is extremely useful in ML for extracting training and test sets, creating minibatches, or extracting specific features. Understanding slicing is crucial because it's used extensively in data preprocessing.",
                    "Boolean indexing provides powerful filtering capabilities. You create a boolean array using comparison operators (arr > 5) and use it to select elements: arr[arr > 5] returns only elements greater than 5. This is invaluable for data cleaning and feature engineering. For example, you can easily filter out outliers, select data for specific classes, or extract samples meeting certain criteria. Boolean indexing is more efficient than Python loops and reads more naturally than conditional statements.",
                    "Fancy indexing uses integer arrays to select elements. arr[[0, 2, 4]] selects elements at indices 0, 2, and 4. Combined with boolean indexing, this enables sophisticated data manipulation. In ML contexts, fancy indexing is used for randomly shuffling samples, selecting specific training examples, or reordering features. Mastering these indexing techniques significantly streamlines your data preprocessing code.",
                ]
            },
            {
                "section": "Broadcasting and Vectorized Operations",
                "paragraphs": [
                    "Broadcasting is a mechanism that allows NumPy to work with arrays of different shapes during arithmetic operations. When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions and works backward. Dimensions are compatible when they are equal or one of them is 1. If these conditions aren't met, broadcasting fails. This concept initially seems complex but is fundamental to writing efficient NumPy code.",
                    "Consider adding a 1D array of shape (3,) to a 2D array of shape (4, 3). NumPy treats the 1D array as having shape (1, 3), then stretches it to (4, 3) to match the 2D array's shape. The result is that the 1D array is effectively added to each row of the 2D array. Understanding this behavior prevents shape mismatch errors and allows you to write more elegant code. In ML, broadcasting is used constantly: normalizing features by subtracting means and dividing by standard deviations, applying bias terms in neural networks, and many other operations.",
                    "Vectorized operations are the heart of NumPy's efficiency. Instead of using Python loops to process elements, you apply operations directly to arrays. For example, arr * 2 multiplies every element by 2 without any explicit loop. These operations are implemented in C and are dramatically faster than equivalent Python loops. In ML, vectorization is essential for handling large datasets efficiently. A neural network with millions of parameters needs vectorized operations to train in reasonable time.",
                    "Broadcasting enables elegant solutions to complex problems. You can compute distances between multiple points with a single vectorized operation. You can apply different operations to different samples in a batch without loops. You can compute statistics across different axes without explicit iteration. Mastering broadcasting transforms your ability to write concise, efficient ML code.",
                ]
            },
            {
                "section": "Linear Algebra with NumPy",
                "paragraphs": [
                    "Matrix operations are fundamental to machine learning. The dot product of two vectors measures their similarity and is used throughout ML: in neural network computations, in similarity metrics, and in many optimization algorithms. NumPy's dot() function (or @ operator) efficiently computes matrix products. For two 1D arrays, it computes the dot product. For 2D arrays, it performs matrix multiplication. Understanding matrix operations is essential for understanding how ML algorithms work mathematically.",
                    "Matrix decomposition techniques reveal underlying structure in data. Singular Value Decomposition (SVD) decomposes a matrix into three matrices with special properties and is used for dimensionality reduction and denoising. Eigenvalue decomposition reveals the principal directions of variation in data and is the basis of Principal Component Analysis (PCA). NumPy provides functions for these decompositions, but understanding the mathematics helps you use them correctly and interpret results.",
                    "Systems of linear equations appear frequently in ML applications. From solving least squares problems in linear regression to optimizing neural networks, you need to solve Ax = b where A is a matrix, x is unknown, and b is known. NumPy's linalg.solve() efficiently solves these systems. You could also use the pseudoinverse (linalg.pinv()) to find the least-squares solution, which is robust when the system is underdetermined or overdetermined.",
                    "Matrix norms measure the magnitude of matrices and are used in regularization, convergence criteria, and model evaluation. The Frobenius norm is the generalization of the L2 norm to matrices. The spectral norm (largest singular value) bounds the growth of errors. Understanding norms helps you implement proper regularization in machine learning models and diagnose numerical stability issues.",
                ]
            }
        ]
    },
    {
        "name": "L08.1_pandas-dataframes",
        "title": "Pandas DataFrames and Data Manipulation",
        "content_sections": [
            {
                "section": "Introduction to DataFrames",
                "paragraphs": [
                    "A DataFrame is a 2D tabular data structure with labeled rows (index) and columns, similar to a spreadsheet or relational database table. Unlike NumPy arrays which are homogeneous, DataFrames can contain different data types in different columns: numerical columns, text columns, categorical data, datetime information, and more. This flexibility makes DataFrames ideal for real-world data which is rarely purely numerical. The index provides labels for rows (numeric or custom), and column names provide labels for columns, making your data self-documenting and easier to work with.",
                    "Creating DataFrames is straightforward. You can read from CSV files with pd.read_csv(), from Excel files with pd.read_excel(), from databases, or from Python dictionaries and lists. For example, pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}) creates a DataFrame with columns A and B. You can also convert NumPy arrays to DataFrames. In ML workflows, the ability to read from various file formats is crucial because data comes from many different sources.",
                    "DataFrame attributes provide essential information about your data. The shape attribute tells you the number of rows and columns. The dtypes attribute shows the data type of each column. The columns attribute lists all column names. The index attribute shows the row labels. The info() method provides a comprehensive summary including memory usage. The describe() method provides statistical summaries for numerical columns. These tools help you understand your dataset's structure and identify potential issues early.",
                ]
            },
            {
                "section": "Data Cleaning and Preparation",
                "paragraphs": [
                    "Missing data is one of the most common data quality issues. Missing values can occur for many reasons: measurement failures, data entry errors, or information that was not collected. Different strategies handle missing data: you might drop rows or columns with missing values if they're sparse, fill missing values with a meaningful constant (like 0 or the mean), use interpolation to estimate values based on neighboring data, or use advanced techniques like multiple imputation. The choice depends on the amount and pattern of missing data. In pandas, isnull() or isna() identifies missing values, and dropna() or fillna() handles them.",
                    "Duplicates can artificially inflate your dataset or cause bias. The duplicated() method identifies duplicate rows, and drop_duplicates() removes them. Sometimes duplicates are exact (identical across all columns), but often you need to consider only specific columns when checking for duplicates. For example, you might have multiple records for the same customer with slightly different information. Identifying and handling duplicates is crucial for data integrity, especially when the same sample would be counted multiple times in model training.",
                    "Data type conversions ensure your columns have appropriate types. pd.to_numeric() converts strings to numbers, pd.to_datetime() converts strings to datetime objects, and astype() converts between types. DateTime columns require special handling because they contain temporal information that standard numerical algorithms can't directly use. You need to extract features like day of week, month, or days since a reference date. Proper data typing enables both correct operations and memory efficiency.",
                    "Outliers are data points that deviate significantly from the norm. They might be genuine unusual observations or data entry errors. Identifying outliers uses statistical methods: values beyond mean ± 3*std, values beyond IQR (Interquartile Range) methods, or domain knowledge. Some algorithms are robust to outliers (decision trees), while others are sensitive (linear regression). You can remove, cap, or transform outliers depending on your analysis goals.",
                ]
            },
            {
                "section": "Data Transformation and Feature Engineering",
                "paragraphs": [
                    "Normalization and standardization transform features to have comparable scales. Normalization (Min-Max scaling) scales values to a [0, 1] or [-1, 1] range using (x - min) / (max - min). Standardization (Z-score normalization) subtracts the mean and divides by standard deviation: (x - mean) / std. When features have different scales—like age (0-100) versus income (0-1,000,000)—algorithms like K-means or neural networks perform better with normalized features. Distance-based algorithms are particularly sensitive because Euclidean distance is dominated by larger-scale features.",
                    "Categorical data requires special handling. One-hot encoding converts categorical variables into binary columns: a 'color' column with values red/green/blue becomes three binary columns. Ordinal encoding assigns integers to categories, useful when categories have inherent order (small < medium < large). Label encoding converts categories to integers, useful for tree-based algorithms but not for distance-based algorithms. The choice affects model performance, so understanding these transformations is crucial.",
                    "Feature creation and selection can dramatically improve model performance. You might create interaction features (age × income), polynomial features (age²), or domain-specific features (for sales data, creating features like 'is_weekend' or 'is_holiday'). Feature scaling (standardization) makes features comparable. Feature selection removes irrelevant or redundant features, improving interpretability and reducing overfitting. Pandas groupby() and apply() functions excel at creating aggregated features from raw data.",
                    "Time series data requires special handling beyond standard ML techniques. You often need to create lagged features (previous values), rolling statistics (moving averages), or seasonal decompositions. The shift() method creates lagged features, rolling() computes windowed statistics. These transformations capture temporal patterns that algorithms can learn from. Time series analysis is crucial in financial forecasting, demand prediction, and many other domains.",
                ]
            },
            {
                "section": "GroupBy Operations and Aggregations",
                "paragraphs": [
                    "The groupby() operation is one of Pandas' most powerful features. It splits data into groups based on one or more columns, applies a function to each group, and combines results. For example, df.groupby('department')['salary'].mean() calculates mean salary by department. This is useful for exploratory analysis and feature engineering. You can group by multiple columns to create hierarchical structures, then apply different aggregations to different columns.",
                    "Aggregation functions summarize data across groups. Common functions include sum(), mean(), median(), std(), min(), max(), and count(). You can apply a single function to all columns (df.groupby('category').sum()) or different functions to different columns using agg(). For example, agg({'sales': 'sum', 'profit': 'mean', 'quantity': 'count'}) applies sum to sales, mean to profit, and count to quantity in a single operation. Named aggregations make the output column names descriptive and the code clearer.",
                    "Transform operations apply functions to each group while preserving the original DataFrame shape and index. This is useful for creating features relative to group statistics. For example, (df['salary'] - df.groupby('department')['salary'].transform('mean')) creates a feature showing each employee's deviation from their department's mean salary. This pattern is invaluable for feature engineering in predictive models.",
                    "Window functions combine elements of grouping and rolling operations. You can apply a function to a window of data as you move through a dataset. The rolling() and expanding() methods are useful for time series: rolling(window=7) applies operations to 7-day windows, expanding() applies operations to all data up to the current point. These enable efficient feature engineering for sequential data.",
                ]
            }
        ]
    },
    {
        "name": "L08.2_visualization",
        "title": "Data Visualization for ML",
        "content_sections": [
            {
                "section": "Fundamentals of Data Visualization",
                "paragraphs": [
                    "Data visualization transforms raw numbers into visual representations that human brains can process quickly. Visual perception allows us to grasp patterns, outliers, and relationships that are invisible in tables of numbers. In ML, visualization serves multiple purposes: exploratory data analysis reveals patterns in your data, validation checks whether preprocessing worked correctly, model diagnosis shows whether your model is overfitting or underfitting, and communication presents findings to stakeholders. A well-chosen visualization can convey insights that would require paragraphs of text to explain.",
                    "Different visualizations serve different purposes. Histograms show the distribution of a single numerical variable, revealing whether data is normally distributed, skewed, bimodal, or has other patterns. Scatter plots show relationships between two numerical variables and are excellent for detecting correlations, clusters, or anomalies. Box plots show distribution summaries including median, quartiles, and outliers. Bar charts compare categories. Heatmaps show patterns in 2D data like correlation matrices. Choosing the right visualization for your data answers questions most effectively.",
                    "Color, position, and size encode different dimensions of data in visualizations. Position (x and y axes) is the most effective encoding. Size (bubble charts) and color (heatmaps) can encode additional dimensions. However, humans perceive color differently (color blindness affects about 8% of men), so using multiple encodings or accessible color palettes is important. Matplotlib uses RGB color space, but perceptually uniform colormaps like 'viridis' are better for scientific visualization.",
                    "Matplotlib is Python's foundational visualization library. Matplotlib's pyplot interface mimics MATLAB and is intuitive for creating plots. It provides fine-grained control over every aspect of a plot: colors, line styles, marker styles, font sizes, legends, and more. Seaborn builds on Matplotlib and provides high-level functions for statistical visualizations. Pandas provides convenient plotting methods for DataFrames. These tools form the visualization foundation for most ML work.",
                ]
            },
            {
                "section": "Exploratory Data Analysis Visualizations",
                "paragraphs": [
                    "Univariate analysis examines single variables in isolation. Histograms with various bin sizes reveal distribution shape. Kernel Density Estimation (KDE) plots provide smooth density estimates. Violin plots combine box plots with density information. These visualizations help identify the distribution type (normal, exponential, bimodal, etc.), detect skewness (asymmetry), and identify potential outliers. In ML preprocessing, understanding each variable's distribution guides decisions about transformations: you might log-transform highly skewed data or normalize normally distributed data.",
                    "Bivariate analysis examines relationships between pairs of variables. Scatter plots show relationships with continuous variables. Calculating and visualizing correlation coefficients quantifies linear relationships. Violin plots or swarm plots show distributions of a continuous variable across categorical groups. These visualizations reveal whether variables are independent, positively correlated, negatively correlated, or have non-linear relationships. Understanding relationships helps identify useful features and detect multicollinearity (highly correlated features that cause problems in some algorithms).",
                    "Multivariate visualization extends to multiple variables. Pair plots (scatter plot matrices) show relationships between all pairs of variables in your dataset, revealing patterns difficult to spot otherwise. Correlation heatmaps show all pairwise correlations in a single view. 3D scatter plots (though often harder to interpret) show three continuous variables simultaneously. Parallel coordinate plots show high-dimensional patterns. These comprehensive visualizations are invaluable during initial data exploration.",
                    "Time series visualization reveals temporal patterns. Line plots show how variables change over time. Seasonal decomposition plots separate time series into trend, seasonal, and residual components. Lag plots show autocorrelation patterns. These visualizations reveal whether data has trends (systematic increase or decrease), seasonality (repeating patterns), or cycles. Time series analysis requires special treatment in ML, so understanding temporal patterns is crucial.",
                ]
            },
            {
                "section": "Model-Related Visualizations",
                "paragraphs": [
                    "Feature importance visualizations show which features contribute most to model predictions. Bar plots ranking feature importances help you understand what patterns your model learned. In tree-based models, feature importance comes directly from the algorithm. In linear models, coefficient magnitudes show feature importance. In neural networks, you might visualize weight distributions or use gradient-based importance measures. Understanding feature importance builds confidence in your model and reveals whether it's learning sensible patterns.",
                    "Confusion matrices visualize classification model performance. Rows represent true labels, columns represent predicted labels. The diagonal shows correct predictions, off-diagonal elements show errors. Heatmap visualization makes patterns obvious: which classes are often confused with each other. From confusion matrices, you calculate metrics like precision, recall, and F1-score. For imbalanced datasets, confusion matrices reveal whether models are biased toward the majority class.",
                    "ROC (Receiver Operating Characteristic) curves and Precision-Recall curves evaluate binary classification models across different thresholds. The ROC curve plots true positive rate versus false positive rate, showing the trade-off. The area under the ROC curve (AUC-ROC) summarizes performance: 0.5 is random, 1.0 is perfect. Precision-Recall curves are more informative for imbalanced datasets. These curves help you choose appropriate thresholds for your use case.",
                    "Learning curves show how model performance improves with more training data. Plotting train and validation loss as training progresses reveals whether a model is suffering from bias (high train loss) or variance (gap between train and validation loss). These diagnostics guide decisions about collecting more data, simplifying the model, or increasing model complexity. Understanding the bias-variance trade-off is essential for effective model development.",
                ]
            }
        ]
    },
    {
        "name": "L09.1_linear-algebra",
        "title": "Linear Algebra Fundamentals for ML",
        "content_sections": [
            {
                "section": "Vectors and Matrices",
                "paragraphs": [
                    "Vectors are ordered collections of numbers, representing points in space. A 2D vector (3, 4) represents a point in 2D space. A 3D vector (1, 2, 3) represents a point in 3D space. Higher-dimensional vectors extend this concept. In ML, vectors represent data points: an image might be flattened into a vector where each component represents a pixel intensity, or a data sample with n features becomes an n-dimensional vector. Vector interpretation as points in space enables geometric intuition: distance between vectors measures similarity, angles between vectors indicate correlation.",
                    "Matrices are rectangular arrays of numbers, organizing vectors into a grid. A dataset with n samples and m features is represented as an n×m matrix. Each row is a sample (vector), each column is a feature. Matrix notation allows concise expression of operations that might otherwise require nested loops. Matrix multiplication combines transformations: if matrix A transforms vectors from space X to space Y, and matrix B transforms vectors from space Y to space Z, then B@A transforms from X to Z. This composition property is fundamental to understanding neural networks as chains of transformations.",
                    "Matrix transpose swaps rows and columns: (A^T)[i,j] = A[j,i]. Transpose is used in many ML contexts: when you need to switch between row-major and column-major perspectives, in optimization algorithms, and in computing Gram matrices. The product of a matrix and its transpose (A^T @ A) appears frequently in normal equations for linear regression and in computing covariance matrices.",
                    "Special matrices have unique properties. The identity matrix I acts like 1: A @ I = A. Diagonal matrices affect each component independently—scaling features by different amounts. Orthogonal matrices preserve lengths and angles: Q^T @ Q = I. These appear in QR decomposition and PCA. Understanding these special matrices accelerates your intuition about how transformations affect data.",
                ]
            },
            {
                "section": "Vector and Matrix Operations",
                "paragraphs": [
                    "The dot product measures similarity between vectors. For vectors u and v, the dot product is the sum of component-wise products: u·v = u1*v1 + u2*v2 + ... + un*vn. Geometrically, u·v = ||u|| ||v|| cos(θ), where θ is the angle between them. When vectors are normalized (unit length), the dot product equals the cosine similarity. This is why cosine similarity appears throughout ML: recommendation systems use it to find similar items, clustering algorithms minimize distance which relates to dot products, and neural networks compute dot products between inputs and weights.",
                    "Matrix multiplication combines dot products. Computing C = A @ B (where A is m×n and B is n×p results in C being m×p) involves taking dot products of rows of A with columns of B. Each element C[i,j] is the dot product of row i of A with column j of B. This operation appears everywhere: computing predictions in linear models (y = X @ w), composing neural network layers, transforming coordinates. Understanding matrix multiplication as a collection of dot products builds intuition for ML.",
                    "Norms quantify vector magnitude. The L2 norm (Euclidean norm) ||v||₂ = √(v1² + v2² + ... + vn²) is the most common. The L1 norm ||v||₁ = |v1| + |v2| + ... + |vn| is used in some algorithms. The max norm ||v||∞ = max(|v1|, |v2|, ..., |vn|) is used occasionally. In ML, norms are used to measure prediction errors (MSE uses L2 norm), to regularize models (L1 regularization encourages sparsity, L2 encourages small weights), and to normalize features to comparable scales.",
                    "Matrix norms extend vector norms. The Frobenius norm extends L2: ||A||F = √(sum of all squared elements). The spectral norm (operator norm) is the largest singular value, measuring how much the matrix stretches vectors. Understanding matrix norms helps you implement regularization correctly and diagnose numerical stability issues in optimization.",
                ]
            },
            {
                "section": "Solving Linear Systems",
                "paragraphs": [
                    "Linear systems appear constantly in ML. Linear regression solves Xw = y (actually the least-squares version when no exact solution exists). Linear constraints appear in optimization problems. Understanding how to solve these systems is fundamental. There are two main approaches: direct methods (like Gaussian elimination and its refinements) compute the solution in finite steps, while iterative methods (like gradient descent) gradually improve approximations.",
                    "Gaussian elimination is the foundation of direct methods. You transform the system into upper triangular form through row operations, then solve by back-substitution. This works perfectly for small systems, but numerical stability becomes an issue for large systems. The solution x = A^(-1) @ b uses the matrix inverse, but computing the inverse is computationally expensive and numerically unstable. NumPy's linalg.solve() uses optimized implementations (like LU decomposition) that are more stable and efficient than computing the inverse.",
                    "When exactly solving Ax = b is impossible (overdetermined systems: more equations than unknowns), we solve the least-squares problem: minimize ||Ax - b||². The solution is x = (A^T @ A)^(-1) @ A^T @ b. This is the foundation of linear regression. The pseudoinverse A⁺ = (A^T @ A)^(-1) @ A^T provides this least-squares solution directly. Computing the pseudoinverse via SVD is numerically stable and also reveals information about the system's conditioning.",
                    "The condition number measures how sensitive a system's solution is to perturbations in the input. A high condition number means small changes in A or b cause large changes in x, indicating numerical instability. Ill-conditioned systems (those with large condition numbers) are problematic: solving them gives unreliable results due to floating-point errors. In ML, ill-conditioning appears when features are highly correlated or when the feature matrix is nearly singular. Regularization addresses this by adding a small term to the diagonal of A^T @ A.",
                ]
            }
        ]
    },
    {
        "name": "L09.2_eigenvalues-pca",
        "title": "Eigenvalues and Principal Component Analysis",
        "content_sections": [
            {
                "section": "Eigenvalues and Eigenvectors",
                "paragraphs": [
                    "Eigenvectors are special vectors that a matrix only stretches (or shrinks) without changing direction. For a square matrix A, an eigenvector v and corresponding eigenvalue λ satisfy Av = λv. This equation says applying the transformation A to v is equivalent to simply scaling v by λ. Finding eigenvectors means finding directions where the matrix's action is pure scaling. For example, a rotation matrix has eigenvectors pointing along the axes of rotation (with eigenvalue ±1), while a scaling matrix's eigenvectors are all non-zero vectors (with eigenvalue equal to the scale factor).",
                    "Eigenvalues reveal how much stretching happens in each eigenvector direction. Positive eigenvalues mean stretching in the original direction, negative eigenvalues mean flipping and stretching, and eigenvalues between 0 and 1 mean shrinking. The magnitude of the eigenvalue indicates how much stretching occurs. In ML, the largest eigenvalues correspond to directions of maximum variance in data. Computing eigenvalues and eigenvectors is expensive for large matrices, but many problems don't require all of them—just the largest few, which can be computed efficiently.",
                    "The characteristic polynomial det(A - λI) = 0 has roots equal to the eigenvalues. This formula is the theoretical foundation for finding eigenvalues, but calculating it directly is inefficient. In practice, iterative algorithms (like the power method or QR algorithm) find eigenvalues more efficiently. These algorithms are essential because eigendecomposition is used throughout ML: covariance matrix eigendecomposition for PCA, Markov chain analysis, Google's PageRank algorithm, spectral clustering, and many other applications.",
                    "Symmetric matrices (A = A^T) have special properties: all eigenvalues are real, eigenvectors are orthogonal, and the matrix can be diagonalized. Covariance matrices are symmetric, which is why PCA relies on eigendecomposition. The eigenvector decomposition A = Q @ Λ @ Q^T expresses the matrix as a sum of rank-1 matrices (outer products of eigenvectors scaled by eigenvalues). This decomposition reveals the matrix's structure and is the foundation of dimensionality reduction techniques.",
                ]
            },
            {
                "section": "Singular Value Decomposition (SVD)",
                "paragraphs": [
                    "Singular Value Decomposition (SVD) extends eigendecomposition to rectangular matrices. Every matrix A (m×n) can be decomposed as A = U @ Σ @ V^T, where U is m×m orthogonal, Σ is m×n diagonal with singular values, and V^T is n×n orthogonal. The singular values (diagonal elements of Σ) are non-negative and typically ordered decreasing. SVD is more numerically stable and more generally applicable than eigendecomposition, making it the preferred method in practice.",
                    "The singular values σ₁ ≥ σ₂ ≥ ... measure how much the matrix stretches vectors in the corresponding directions. The ratio of the largest to smallest non-zero singular value (the condition number) indicates numerical stability. Small singular values mean the matrix nearly loses information in those directions—columns are nearly linearly dependent. In ML, small singular values indicate multicollinearity or overfitting, so regularization techniques often target them.",
                    "SVD reveals the rank of a matrix. The number of non-zero singular values equals the rank. A rank-deficient matrix (rank < min(m, n)) cannot be inverted and indicates redundancy in your data. In ML, rank deficiency means features contain redundant information or you have more features than independent samples. The null space (directions where A projects to zero, corresponding to zero singular values) reveals which combinations of features have no effect on the output.",
                    "SVD simplifies many problems. Solving least-squares problems via SVD is more stable than computing the pseudoinverse directly. Approximating a matrix with fewer rank requires keeping only the largest k singular values and corresponding vectors, useful for denoising and compression. Image compression, denoising, recommendation systems, and latent semantic analysis all use SVD or its principles.",
                ]
            },
            {
                "section": "Principal Component Analysis (PCA)",
                "paragraphs": [
                    "Principal Component Analysis (PCA) reduces data dimensionality while preserving as much variance as possible. It finds new axes (principal components) that are ordered by variance they capture. The first principal component is the direction of maximum variance in the data, the second is orthogonal to the first and captures the most remaining variance, and so on. After centering the data (subtracting the mean), PCA computes the covariance matrix, finds its eigenvalues and eigenvectors, and sorts them by eigenvalue magnitude. The eigenvectors become the principal components, and eigenvalues represent variance along each direction.",
                    "Projecting data onto principal components reduces dimensionality. If you keep only the first k principal components, you reduce from n to k dimensions while retaining the maximum possible variance with k dimensions. This reconstruction is not lossless—you lose any variance along the discarded components. However, if the discarded components contain mostly noise, dimensionality reduction actually improves model performance by reducing overfitting. The explained variance ratio (the fraction of total variance retained) guides choosing k.",
                    "PCA advantages are significant. It reduces data size, speeding up model training and reducing memory usage. It removes multicollinearity, enabling algorithms sensitive to correlated features. It enables visualization of high-dimensional data by projecting to 2D or 3D. It can reveal underlying structure in data through the principal components. However, PCA has limitations: it's an unsupervised method (ignores labels), it assumes linear relationships (though kernel PCA extends to non-linear), and the principal components are sometimes hard to interpret because each is a linear combination of all original features.",
                    "Practical PCA implementation requires careful scaling. PCA is sensitive to feature scales—if one feature has much larger values, it dominates the first principal component regardless of its importance. Always standardize features (subtract mean, divide by standard deviation) before computing components. The number of components to keep is typically chosen by examining the cumulative explained variance: keep enough to explain 90-95% of variance, or use the elbow method to find where adding more components gives diminishing returns.",
                ]
            }
        ]
    },
    {
        "name": "L10.1_calculus-derivatives",
        "title": "Calculus and Derivatives for ML",
        "content_sections": [
            {
                "section": "Functions and Rates of Change",
                "paragraphs": [
                    "A function maps inputs to outputs: y = f(x). The derivative measures how much the output changes when the input changes slightly. Formally, the derivative f'(x) is the limit of [f(x+h) - f(x)] / h as h approaches 0. Geometrically, it's the slope of the tangent line at x. For a simple example, if f(x) = x², then f'(x) = 2x: at x = 3, the slope is 6, meaning small increases in x cause the output to increase 6 times as much. Understanding derivatives as rates of change provides intuition for optimization.",
                    "Higher derivatives measure changes in rates of change. The second derivative f''(x) = d/dx[f'(x)] measures how fast the first derivative changes. Geometrically, it measures curvature: positive second derivative means the function curves upward (convex), negative means it curves downward (concave). In ML optimization, the second derivative appears in Newton's method, which uses curvature information for faster convergence than gradient descent. Hessians (matrices of second derivatives) reveal the local geometry around a point.",
                    "Partial derivatives extend to multiple variables. For a function f(x, y), the partial derivative ∂f/∂x measures how much the output changes when x changes slightly, holding y constant. The gradient ∇f = (∂f/∂x, ∂f/∂y) is a vector of all partial derivatives, pointing in the direction of steepest increase. In ML, the gradient drives optimization: gradient descent moves in the direction opposite to the gradient (steepest descent) to minimize the loss function.",
                    "The chain rule, one of calculus's most powerful tools, enables computing derivatives of composite functions. If y = f(g(x)), then dy/dx = (df/dg) × (dg/dx). In ML, the chain rule enables backpropagation in neural networks: computing gradients of the loss with respect to early-layer weights requires chaining derivatives from the output backward. Understanding the chain rule conceptually (each stage multiplies derivatives) prepares you to understand backpropagation.",
                ]
            },
            {
                "section": "Optimization and Critical Points",
                "paragraphs": [
                    "Critical points are where the derivative equals zero: f'(x) = 0. At critical points, the slope is zero, meaning the function temporarily stops increasing or decreasing. Critical points can be local minima (function value lower than nearby points), local maxima (higher than nearby points), or saddle points (neither). The second derivative reveals the type: f''(x) > 0 indicates a minimum, f''(x) < 0 indicates a maximum, f''(x) = 0 is inconclusive. In ML, we seek the global minimum of the loss function, but algorithms often find local minima instead.",
                    "Convex functions have at most one local minimum (the global minimum). The second derivative f''(x) ≥ 0 everywhere. Convex loss functions are desirable because any local minimum is global—optimization algorithms guarantee finding the best solution. Linear regression has a convex loss function, ensuring a unique global solution. Neural networks have non-convex loss functions where local minima can be suboptimal, making optimization harder. Convex optimization is a solved problem; non-convex optimization remains an open research area.",
                    "Gradient descent iteratively improves a solution by moving opposite the gradient. The update rule is x_{k+1} = x_k - α ∇f(x_k), where α is the learning rate controlling step size. Small step sizes guarantee the function value decreases but cause slow convergence. Large step sizes risk overshooting and diverging. Choosing the learning rate is crucial: too small and training takes forever, too large and the algorithm diverges. Adaptive methods (Adam, RMSprop) adjust the learning rate during training.",
                    "Second-order methods use curvature information for faster convergence. Newton's method uses the Hessian (matrix of second derivatives): x_{k+1} = x_k - H^(-1) ∇f(x_k), where H is the Hessian. When the Hessian is available and invertible, Newton's method converges much faster than gradient descent. However, computing and inverting the Hessian is expensive for large problems. Quasi-Newton methods (BFGS, L-BFGS) approximate the Hessian efficiently.",
                ]
            }
        ]
    },
    {
        "name": "L10.2_gradient-descent",
        "title": "Gradient Descent and Optimization",
        "content_sections": [
            {
                "section": "Gradient Descent Variants",
                "paragraphs": [
                    "Gradient descent comes in batch, stochastic (SGD), and mini-batch variants depending on how much data you use per update. Batch gradient descent computes gradients using all training data, then updates weights once. This is stable but slow for large datasets. Stochastic gradient descent (SGD) updates weights after each single sample, making progress quickly but with noisy updates. Mini-batch gradient descent (the standard approach) uses batches of 32-256 samples, balancing stability and speed. In practice, mini-batch creates an efficient update that's stable enough but allows quick progress through large datasets.",
                    "The choice of batch size affects convergence behavior. Larger batches give more accurate gradient estimates but are slower to process one batch. Smaller batches give noisier gradients but allow more weight updates and sometimes better generalization. The noise in mini-batch gradients acts as implicit regularization, discouraging the model from fitting noise—a phenomenon called the generalization gap. Modern practice uses relatively large batch sizes (power of 2, like 32-128) for stability and computational efficiency.",
                    "Learning rate scheduling adjusts the step size during training. A constant learning rate often works poorly: starting with a large learning rate enables fast initial progress, while a smaller rate later allows fine-tuning near the optimum. Schedules can be exponential (multiply by a constant factor), step-based (divide by 10 at specific epochs), or cosine (smoothly decrease). Some methods like warm restarts jump back to a larger learning rate periodically, helping escape local minima. The learning rate schedule significantly affects final model performance.",
                    "Momentum accelerates gradient descent by accumulating gradients over time. The update accumulates like v = β*v + ∇f(x), then x -= α*v. Momentum carries forward progress from previous steps, accelerating through directions of consistent gradient and damping oscillations. Classical momentum uses β ≈ 0.9. Nesterov momentum looks ahead before updating, correcting for the momentum trajectory. Momentum-based methods often converge 5-10x faster than vanilla gradient descent.",
                ]
            },
            {
                "section": "Adaptive Learning Rate Methods",
                "paragraphs": [
                    "Adagrad adapts learning rates per parameter based on historical gradients. Parameters with large gradients get smaller effective learning rates, while parameters with small gradients get larger rates. This works well when gradients vary drastically between parameters. However, Adagrad's accumulation causes learning rates to shrink monotonically, eventually becoming tiny and stopping learning. This makes Adagrad better suited for sparse data (where many parameters are zero) than for dense problems.",
                    "RMSprop addresses Adagrad's problem by using a weighted average of squared gradients instead of accumulating all past gradients. This keeps the learning rate from shrinking to zero. RMSprop's update: accumulate squared gradients with decay factor (typically 0.9), then divide the gradient by the square root of the accumulated squares. RMSprop works well in practice across many problems and is computationally efficient. It often outperforms vanilla momentum-based methods.",
                    "Adam (Adaptive Moment Estimation) combines momentum and RMSprop, maintaining both a moving average of gradients (first moment, like momentum) and squared gradients (second moment, like RMSprop). This provides both acceleration and adaptive learning rates. Adam often converges faster than RMSprop and requires less learning rate tuning. It's become the default optimizer in deep learning, recommended for trying first before tuning other optimizers. The bias correction in Adam (important in early training) is often overlooked but helps avoid erratic initial updates.",
                    "These adaptive methods have advantages and disadvantages. They generally train faster than vanilla gradient descent, requiring fewer epochs. However, they sometimes generalize worse than SGD with momentum (Adam and SGD sometimes find different minima with different generalization properties). Research continues on understanding why, and some practitioners use warm-up learning rates (starting small, then switching optimizers) or careful learning rate tuning to get the best of both worlds.",
                ]
            },
            {
                "section": "Convergence and Debugging",
                "paragraphs": [
                    "Convergence curves plot training progress, showing loss or accuracy versus epochs. Ideally, training loss decreases smoothly and validation loss follows, eventually plateauing near the global optimum. Characteristic patterns indicate problems: training loss stagnating suggests too-small learning rate or insufficient model capacity; training loss decreasing but validation loss increasing suggests overfitting. The gap between training and validation loss reveals the generalization gap. Monitoring these curves is the primary tool for diagnosing optimization progress.",
                    "Exploding and vanishing gradients plague deep neural networks. In networks with many layers, gradients are products of many matrices through backpropagation. If these products grow exponentially (exploding), updates become huge and diverge. If they shrink exponentially (vanishing), gradients become too small to drive learning—early layers barely change. Gradient clipping caps gradient magnitudes to prevent explosion. Batch normalization stabilizes gradient flow. Skip connections (residual networks) enable gradients to flow directly through early layers. Learning about these problems is crucial for training deep networks.",
                    "Hyperparameter tuning optimizes algorithm choices for your problem. Learning rate is paramount—too small and convergence is agonizingly slow, too large and optimization diverges. Batch size affects convergence stability and speed. The number of epochs (training iterations) balances convergence with computational cost. Weight decay (L2 regularization) controls overfitting. Trying multiple hyperparameter combinations systematically (grid search) or probabilistically (random search, Bayesian optimization) often finds better solutions. Automated hyperparameter tuning tools exist, but understanding the effects manually is invaluable.",
                    "Early stopping prevents overfitting by monitoring validation performance during training. When validation loss stops improving for several epochs (patience parameter), training halts. This prevents the model from memorizing the training set. Checkpoint-based approaches save the best model found during training rather than the final model. These techniques are simple but remarkably effective. Regularization (weight decay, dropout) provides additional overfitting prevention. The combination of early stopping and regularization is standard practice.",
                ]
            }
        ]
    },
    {
        "name": "L11.1_statistics",
        "title": "Statistical Foundations for ML",
        "content_sections": [
            {
                "section": "Probability Fundamentals",
                "paragraphs": [
                    "Probability quantifies uncertainty. A probability is a number between 0 and 1: exactly 0 means impossible, 1 means certain, 0.5 means equally likely or unlikely. Probability mass functions (PMF) describe discrete random variables—assigning probabilities to specific outcomes. Probability density functions (PDF) describe continuous random variables—the integral of the PDF over an interval gives the probability. The cumulative distribution function (CDF) at value x is the probability of being less than or equal to x. Understanding these distributions is fundamental because data is modeled as samples from distributions.",
                    "The mean (expected value) is the average outcome if you repeat an experiment infinitely many times. The variance measures spread around the mean: large variance means outcomes are spread out, small variance means they cluster near the mean. The standard deviation is the square root of variance and has the same units as the variable itself. These statistics summarize distributions and guide preprocessing: normalizing data to zero mean and unit variance is standard practice. Understanding distributions reveals why normalization helps algorithms.",
                    "The normal (Gaussian) distribution is ubiquitous in statistics and ML. Many natural phenomena and measurement errors follow normal distributions due to the central limit theorem: the sum of many independent random variables approximates normal distribution. The normal distribution is completely specified by its mean and variance. Many ML algorithms assume normally distributed errors, and many statistical tests assume normality. Q-Q plots visually assess whether data is normally distributed by comparing quantiles.",
                    "Outliers are values far from the typical range. For normal distributions, 68% of values fall within one standard deviation of the mean, 95% within two, 99.7% within three. Values beyond three standard deviations are rare and often indicate errors or genuinely unusual observations. Different approaches handle outliers: remove them if they're errors, keep them if they're genuine, or use robust methods less sensitive to outliers. Understanding outliers' effects on analysis is crucial.",
                ]
            },
            {
                "section": "Sampling and Estimation",
                "paragraphs": [
                    "Sampling is selecting a subset from a population to study. Without being able to study the entire population, you use samples to estimate population characteristics. The sampling distribution of an estimator describes how the estimate varies across different samples. Standard error quantifies this variation: smaller standard error means more precise estimates. Larger sample sizes generally reduce standard error, following the √n rule: to halve standard error, you need four times more samples. This relationship guides sample size calculations.",
                    "Estimators are statistics (functions of data) that estimate population parameters. An estimator is unbiased if its expected value equals the true parameter. The sample mean is an unbiased estimator of the population mean. The sample variance, using n-1 instead of n in the denominator (Bessel's correction), is an unbiased estimator of population variance. Understanding estimator properties guides choosing good statistics. The maximum likelihood principle chooses the parameter value that makes observed data most probable, a powerful approach for selecting estimators.",
                    "Confidence intervals quantify uncertainty in estimates. A 95% confidence interval is an interval that, if computed repeatedly from independent samples, would contain the true parameter 95% of the time. It doesn't mean there's a 95% probability the true parameter is in this specific interval—the parameter is fixed, the interval is random. Creating confidence intervals requires the sampling distribution, which is often approximated by the bootstrap: resampling with replacement from your data and computing the statistic repeatedly. Bootstrap confidence intervals are simple and generally applicable.",
                    "The law of large numbers states that as sample size increases, sample estimates converge to population parameters. Formal proof requires technical details, but intuitively: with infinite data, you perfectly know the true parameters. In practice, the law of large numbers underlies why ML models improve with more training data (up to a point where other factors become limiting). Data efficiency and understanding how to use limited data effectively is a major ML research area.",
                ]
            },
            {
                "section": "Hypothesis Testing and Statistical Inference",
                "paragraphs": [
                    "Hypothesis testing formalizes comparing observations to expectations. The null hypothesis (H₀) is a baseline claim (usually 'no effect'). The alternative hypothesis (H₁) is what you want to show. Hypothesis tests compute how likely observing your data is if H₀ is true. If very unlikely (p-value < 0.05), you reject H₀. The p-value is the probability of observing data at least as extreme as what you got, assuming H₀ is true. One common misconception: p-value is NOT the probability H₀ is true—it's the probability of the data given H₀.",
                    "Statistical significance (p < 0.05) is a conventional threshold but has limitations. Low p-values can occur from small effects in large samples or from p-hacking (testing many hypotheses). Conversely, true effects might not reach significance in small samples. The significance level (typically 0.05) is arbitrary—the choice should reflect consequences. In safety-critical applications, you might use 0.01; in exploratory research, 0.10 might be appropriate. Reporting effect size (how large the effect is) alongside p-values provides complete information.",
                    "Type I and Type II errors formalize testing mistakes. Type I error: rejecting H₀ when it's true (false positive). Type II error: failing to reject H₀ when it's false (false negative). The significance level α = P(Type I). The power = 1 - P(Type II) is the probability of detecting an effect if it exists. There's a trade-off: improving power requires larger sample sizes. Understanding these errors guides experimental design and interpretation.",
                    "A/B testing applies hypothesis testing to compare two options. Show version A to some users and version B to others, then test if they differ significantly. This enables data-driven decisions in product development. Running A/B tests requires careful experimental design: randomizing assignment, running sufficient duration to collect adequate samples, and predetermining success criteria (not p-hacking). A/B testing's power has led most tech companies to adopt it as standard practice.",
                ]
            }
        ]
    },
    {
        "name": "L11.2_bayesian",
        "title": "Bayesian Methods and Probabilistic ML",
        "content_sections": [
            {
                "section": "Bayes' Theorem and Conditional Probability",
                "paragraphs": [
                    "Conditional probability P(A|B) is the probability of A given that B has occurred. Bayes' theorem relates forward and reverse conditional probabilities: P(A|B) = P(B|A) × P(A) / P(B). This simple formula is incredibly powerful. To compute P(A|B) (posterior belief given evidence B), you use P(B|A) (likelihood of evidence given A) and P(A) (prior belief about A). Bayes' theorem formalizes how to update beliefs as you gather evidence. This principle underlies Bayesian statistics, Bayesian networks, and many ML algorithms.",
                    "Priors encode initial beliefs before seeing data. Weak priors (vague, spread out) represent genuine uncertainty. Strong priors (concentrated) encode confident prior beliefs. The choice of prior affects inference, sometimes heavily—why Bayesian methods require defending prior choices. Improper priors (not true probability distributions) can simplify mathematics but require careful interpretation. Frequentist statisticians critique Bayesian priors as subjective; Bayesians counter that all analysis involves assumptions (priors), and making them explicit is honest.",
                    "Likelihood provides a bridge between data and model. Given model parameters, the likelihood is the probability of observing the data. Maximizing likelihood (maximum likelihood estimation) finds parameters making observed data most probable. Likelihood differs from probability: likelihood describes how probable data is for fixed parameters, probability describes how probable outcomes are for fixed data. In the context 'likelihood of data observed as parameter varies', we're plotting likelihood as a function of parameters.",
                    "The posterior combines prior and likelihood via Bayes' theorem: P(parameters | data) ∝ P(data | parameters) × P(parameters). The posterior is a probability distribution over parameters (unlike frequentism's point estimates). This distribution quantifies uncertainty: you get not just a best-guess parameter but a range reflecting confidence. Posterior credible intervals provide Bayesian analogs to frequentist confidence intervals but with a different interpretation: the true parameter is in the interval with specified probability.",
                ]
            },
            {
                "section": "Bayesian Inference and Computation",
                "paragraphs": [
                    "Computing the posterior exactly is often impossible because the normalizing constant P(data) requires integrating over all parameter values—intractable in high dimensions. Markov Chain Monte Carlo (MCMC) solves this by sampling from the posterior distribution. The Metropolis-Hastings algorithm proposes new parameter values and accepts them based on how much likelihood and prior improve. Over many iterations, samples accumulate around regions of high posterior probability. This computational breakthrough in the 1990s made Bayesian inference practical.",
                    "Gibbs sampling, a specialized MCMC method, samples each parameter conditional on others. It's simple to implement and efficient when conditional distributions are easy to compute. Many hierarchical models have Gibbs sampling implemented. Variational inference approximates the posterior with a simpler distribution by minimizing divergence between approximation and true posterior. It's faster than MCMC but introduces bias from approximation choice. These computational methods make Bayesian methods practical for real problems.",
                    "Empirical Bayes avoids specifying priors by estimating prior parameters from the data. You fit a distribution to observed parameters (if you've repeatedly solved similar problems), then use this fitted distribution as the prior. This pragmatic approach combines prior information from multiple problems while reducing subjectivity. Hierarchical Bayesian models extend this: different groups share a common prior distribution, enabling information sharing while accounting for group variation.",
                    "Bayesian Deep Learning combines neural networks with Bayesian inference. Standard neural networks compute point estimates of weights; Bayesian approaches maintain distributions over weights. This provides uncertainty estimates rather than point predictions. Variational inference approximates weight posterior distributions, enabling practical Bayesian predictions. Though computationally expensive, uncertainty is invaluable in safety-critical applications.",
                ]
            },
            {
                "section": "Applications and Advantages",
                "paragraphs": [
                    "Bayesian networks use directed acyclic graphs to represent conditional dependencies between variables. Nodes are variables, directed edges represent conditioning. This graphical representation clarifies independence assumptions and enables efficient inference. Bayesian networks excel at reasoning with incomplete information: you can query any variable given any evidence. They're used in medical diagnosis, spam detection, and planning. The graph structure formalizes domain knowledge, making assumptions explicit and testable.",
                    "Bayesian methods shine in small-data regimes. With little data, prior information is crucial. Bayesian approaches formally incorporate domain expertise through priors. Frequentist methods, which ignore prior information, often overfit with small samples. Bayesian regularization prevents overfitting through priors that penalize complex models. This is why Bayesian methods remain popular in applied domains with limited data, despite frequentist dominance in academic statistics.",
                    "Decision making under uncertainty is fundamentally Bayesian. You update beliefs as you gather information (following Bayes' theorem), then act to maximize expected utility. In business, this might mean A/B testing to gather data, updating beliefs about which version is better, then deploying the better version. In medicine, it's updating disease probability from test results, then deciding on treatment. The explicit decision framework makes trade-offs clear.",
                    "The Bayesian vs. Frequentist debate reflects different philosophies. Frequentists treat parameters as fixed unknowns, computing probability statements about data. Bayesians treat parameters as random variables, computing probability statements about parameters given data. Frequentism's objectivity (minimized subjectivity in statistical methods) and simplicity are strengths. Bayesianism's ability to incorporate prior knowledge, handle sequential decision-making, and provide intuitive probability statements are strengths. In practice, methods from both schools are useful—'pragmatic Bayesianism' borrows from both.",
                ]
            }
        ]
    }
]

def get_content_for_lesson(lesson_name):
    """Get the enrichment content for a specific lesson"""
    for lesson in LESSONS:
        if lesson["name"] == lesson_name:
            return lesson
    return None

def create_readme_content(lesson):
    """Create comprehensive README content from lesson data"""
    title = lesson["title"]
    content = f"""# {title}
## Comprehensive Learning Guide

## Table of Contents
"""
    
    # Generate TOC
    for section in lesson["content_sections"]:
        content += f"- {section['section']}\n"
    
    content += "\n---\n\n"
    
    # Add sections with paragraphs
    for section in lesson["content_sections"]:
        content += f"## {section['section']}\n\n"
        for para_idx, paragraph in enumerate(section['paragraphs'], 1):
            content += f"{paragraph}\n\n"
    
    return content

def update_lesson_readme(lesson_path, lesson_name, content_sections):
    """Update or create README with new content and commit each paragraph"""
    readme_path = os.path.join(lesson_path, "README.md")
    
    # Read existing content if it exists
    existing_content = ""
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            existing_content = f.read()
    
    # Ensure lesson directory exists
    os.makedirs(lesson_path, exist_ok=True)
    
    commit_count = 0
    
    for section in content_sections:
        section_name = section["section"]
        
        for para_idx, paragraph in enumerate(section["paragraphs"], 1):
            # Create new README with all content up to this point
            readme_content = existing_content + f"\n## {section_name}\n\n{paragraph}\n\n"
            
            # Write the README
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)
            
            # Stage and commit
            try:
                subprocess.run(
                    ["git", "add", readme_path],
                    cwd=os.getcwd(),
                    check=True,
                    capture_output=True
                )
                
                commit_msg = f"[{lesson_name}] Add {section_name} - paragraph {para_idx}"
                subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    cwd=os.getcwd(),
                    check=True,
                    capture_output=True
                )
                
                commit_count += 1
                existing_content = readme_content
                
                print(f"✓ {commit_msg}")
                
            except subprocess.CalledProcessError as e:
                print(f"✗ Error committing {lesson_name}: {e}")
    
    return commit_count

def main():
    """Main function to enrich all lessons in module-02"""
    
    print("=" * 70)
    print("MODULE-02 LESSON ENRICHMENT")
    print("=" * 70)
    
    module_path = "modules/module-02/lessons"
    total_commits = 0
    
    for lesson in LESSONS:
        lesson_name = lesson["name"]
        lesson_path = os.path.join(module_path, lesson_name)
        
        print(f"\n📚 Processing: {lesson_name}")
        print(f"   Title: {lesson['title']}")
        
        # Update README and get commit count
        commits = update_lesson_readme(lesson_path, lesson_name, lesson["content_sections"])
        total_commits += commits
        
        print(f"   Commits created: {commits}")
    
    print("\n" + "=" * 70)
    print(f"✅ ENRICHMENT COMPLETE")
    print(f"Total commits created: {total_commits}")
    print("=" * 70)
    
    # Verify commits
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-n", str(total_commits)],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            check=True
        )
        print("\n📋 Recent commits:")
        print(result.stdout)
    except subprocess.CalledProcessError:
        print("Could not verify commits")

if __name__ == "__main__":
    main()
