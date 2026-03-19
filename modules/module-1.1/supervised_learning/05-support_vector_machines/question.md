# Support Vector Machines - Deep Dive Questions

## Question 1: Optimization Problem
Formulate the SVM optimization problem mathematically. Explain the decision function and the role of support vectors. What does maximizing the margin achieve theoretically?

## Question 2: Kernel Trick
Explain the kernel trick and why it's computationally efficient. Can you apply a kernel to any algorithm? What are the limitations of kernel methods?

## Question 3: Kernel Selection
How would you choose between linear, RBF, polynomial, and sigmoid kernels? What are the computational and accuracy implications of each choice? Can you design a custom kernel?

## Question 4: C and Gamma Parameters
What do C and gamma parameters control in SVM? How do they affect the bias-variance trade-off? What happens when C or gamma are too large or too small?

## Question 5: Soft vs. Hard Margin
Explain the difference between hard margin and soft margin SVM. When would you use each approach? How does the C parameter relate to margin softness?

## Question 6: Multi-class Extension
How does SVM extend from binary to multi-class classification? Compare One-vs-Rest, One-vs-One, and Crammer-Singer approaches. What are the pros and cons?

## Question 7: Scalability Issues
What are the computational limitations of SVM for large datasets? Why is the training complexity O(n³) or worse? How can you address scalability issues (SGDClassifier, approximations)?

## Question 8: SVM vs. Other Algorithms
Compare SVM with logistic regression, neural networks, and tree-based methods. In what scenarios would you prefer SVM over alternatives?

## Question 9: Interpretability
Are SVMs interpretable? How can you extract feature importance or decision rules from an SVM model? What are the limitations compared to decision trees?

## Question 10: Probabilistic Outputs
SVMs produce decision margins, not probabilities. How can you convert SVM outputs to probability estimates? What's the Platt Scaling approach?

