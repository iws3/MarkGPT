# Gradient Boosting - Deep Dive Questions

## Question 1: Sequential Learning Process
Explain how gradient boosting sequentially trains weak learners. What negative gradient does each new tree fit? How does residual fitting work?

## Question 2: Loss Function and Gradient
How does the choice of loss function affect gradient boosting? Can you use different loss functions? How are gradients derived for custom loss functions?

## Question 3: Learning Rate and Shrinkage
What's the role of learning rate in gradient boosting? How do learning rate and number of estimators interact? Is there an optimal shrinkage strategy?

## Question 4: Weak vs. Strong Learners
What makes a learner "weak"? Why do weak learners work better in boosting? Can you use complex learners (deep trees) instead?

## Question 5: Overfitting and Regularization
What causes overfitting in gradient boosting? How do regularization techniques (max_depth, subsample, colsample_bytree) help prevent overfitting?

## Question 6: Boosting vs. Bagging
Compare gradient boosting and random forest (bagging) conceptually. When would you prefer boosting over bagging and vice versa?

## Question 7: Implementation Differences
Compare scikit-learn's GradientBoostingClassifier with XGBoost and LightGBM. What are the key differences in implementation and efficiency?

## Question 8: Feature Importance in Boosting
How is feature importance calculated in gradient boosting? How does it differ from tree importance? Can importance be misleading?

## Question 9: Computational Complexity
What is the time and space complexity of gradient boosting? How does it scale with number of estimators, features, and samples? Why is it slower than random forests?

## Question 10: Advanced Techniques
Discuss stacking, blending, and other ensemble techniques. How do these extend gradient boosting? When would you combine multiple boosting models?

