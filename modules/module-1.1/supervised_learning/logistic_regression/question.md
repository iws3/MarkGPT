# Logistic Regression - Deep Dive Questions

## Question 1: Sigmoid and Probability
Explain the sigmoid function and why it's used in logistic regression. How does it map unbounded values to the [0,1] probability range? What would happen if you used a linear function instead?

## Question 2: Decision Boundary and Classification Threshold
What is a decision boundary in logistic regression? How does changing the classification threshold (from 0.5 to 0.3 or 0.7) affect precision, recall, and the ROC curve?

## Question 3: Cost Function Deep Dive
Explain binary cross-entropy loss and why it's superior to squared error for classification. Derive the gradient for logistic regression and explain how it differs from linear regression.

## Question 4: Regularization in Classification
How does L1 and L2 regularization affect the learned coefficients in logistic regression? When would you prefer L1 over L2 and vice versa?

## Question 5: Multi-class Extension
Explain the differences between One-vs-Rest and Softmax (multinomial) approaches for multi-class logistic regression. What are the computational and probabilistic differences?

## Question 6: Imbalanced Classification
Discuss the challenges of imbalanced datasets in logistic regression. Why is accuracy a poor metric for imbalanced data? How would you address class imbalance?

## Question 7: Relationship to Other Models
How does logistic regression relate to linear discriminant analysis (LDA) and support vector machines (SVM)? What are the conceptual similarities and differences?

## Question 8: Feature Interactions
Can logistic regression capture feature interactions inherently? How would you add interaction terms? Discuss the trade-off between model complexity and interpretability.

## Question 9: Probability Calibration
What is probability calibration? Why might a logistic regression model produce poorly calibrated probabilities? How would you improve calibration?

## Question 10: Interpretation and Business Context
In a credit risk model using logistic regression, how would you interpret a coefficient of 0.8 for "annual_income"? How would you communicate model predictions to non-technical stakeholders?

