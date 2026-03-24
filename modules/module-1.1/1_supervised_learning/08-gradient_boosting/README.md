# Gradient Boosting

## Fundamentals

Gradient Boosting is an ensemble learning technique that sequentially builds weak learners (typically decision trees) and combines them to create a strong predictor. Unlike Random Forest which trains trees independently, Gradient Boosting trains each new tree to correct the errors made by previous trees, hence the name "boosting." The algorithm uses gradient descent to minimize a loss function, making it highly flexible for different problem types. Gradient Boosting is known for its exceptional predictive power and is the backbone of winning solutions in many machine learning competitions. Implementations like XGBoost, LightGBM, and CatBoost have become industry standards due to their efficiency and performance.

## Key Concepts

- **Sequential Training**: Each tree corrects previous errors
- **Gradient Descent**: Minimizes loss function
- **Learning Rate**: Controls step size
- **Trees as Base Learners**: Weak learners combined additively

## Applications

- Competition-winning predictions
- Financial forecasting
- Click-through rate prediction
- Demand forecasting
- Risk assessment

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Boosting Ensemble Strategy and Sequential Models

Gradient Boosting builds an ensemble of weak learners sequentially, where each new learner focuses on correcting the mistakes made by previous learners. Unlike bagging, which trains models in parallel on random subsets, boosting is inherently sequential: each iteration fits a new model to the residuals (negative gradients) of previous predictions. This sequential nature creates strong dependencies between models but dramatically reduces bias. The final prediction is a weighted sum of all weak learner predictions. The boosting principle is powerful: weak learners, individual models that perform only slightly better than random guessing, can be combined into a strong ensemble that achieves excellent performance. This is formalized in boosting theory, which provides bounds on generalization error in terms of training error and the diversity of weak learners.