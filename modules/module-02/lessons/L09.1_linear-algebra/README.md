# Linear Algebra Fundamentals for ML
## Comprehensive Learning Guide

## Vectors and Matrices

Vectors are ordered collections of numbers representing points in space.

Matrices are rectangular arrays of numbers organizing vectors into a grid structure.

Matrix transpose swaps rows and columns: (A^T)[i,j] = A[j,i].

Special matrices have unique properties: identity, diagonal, orthogonal matrices.

## Vector and Matrix Operations

The dot product measures similarity between vectors summing component-wise products.

Matrix multiplication combines dot products of matrix rows with columns.

Norms quantify vector magnitude: L2 (Euclidean), L1, and max norms are common.

Matrix norms extend vector norms: Frobenius norm, spectral norm for stretching.

## Solving Linear Systems

Linear systems appear constantly in ML: linear regression, constraints, equations.

Gaussian elimination is the foundation transforming systems to upper triangular form.

Least-squares solutions minimize error when exactly solving Ax = b is impossible.

The condition number measures sensitivity to perturbations indicating instability.


## Matrix Decompositions

QR decomposition factors matrix into orthogonal and upper triangular.

Cholesky decomposition works with symmetric positive-definite matrices.

LU decomposition factors into lower and upper triangular matrices.

Schur decomposition reveals matrix structure and stability.

Polar decomposition factors into unitary and positive-semidefinite.

Jordan decomposition helps understand matrix powers and exponentials.


## Applications in Machine Learning

Solving least squares problems efficiently with QR decomposition.

Computing determinants from LU for likelihood calculations.

Matrix exponentials appear in continuous-time models.

Kronecker products combine transformations in high dimensions.

Tensor operations extend linear algebra to higher-order tensors.

Vectorization tricks convert nested loops to matrix operations.


## Numerical Stability

Ill-conditioned matrices amplify small input perturbations.

