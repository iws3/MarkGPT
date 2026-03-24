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

### Gradient Descent in Function Space

Gradient Boosting generalizes the boosting framework using gradient descent in function space. Each iteration computes the negative gradient of a loss function with respect to current predictions, identifying the direction that would most reduce loss. A new weak learner (typically shallow decision trees called stumps) is fit to predict these negative gradients. The predictions from this learner are scaled by a learning rate η and added to the ensemble: F_{m+1}(x) = F_m(x) + η·h_m(x), where h_m predicts gradients. The learning rate controls how much each weak learner adjusts the ensemble, providing regularization through slower learning. Smaller learning rates require more iterations but often produce better generalization. The loss function—squared error, absolute error, log loss, or custom functions—can be chosen based on the problem  and determines the gradient direction at each iteration.

### Subsampling and Regularization Strategies

Gradient Boosting is prone to overfitting, especially with high learning rates or many iterations. Several regularization strategies mitigate this risk. Stochastic Gradient Boosting randomly subsamples training data at each iteration (row sampling) and randomly selects features (column sampling) when fitting weak learners. This introduces variance that can offset overfitting while maintaining efficiency. Early stopping monitors validation loss and terminates training when performance plateaus, preventing training on too many iterations. Shrinkage limits the contribution of each weak learner through the learning rate η; smaller values require more iterations but improve generalization. Tree depth constraints limit weak learner complexity, typically using depth 3-8 for stumps. Regularization parameters can be tuned through cross-validation or grid search. The combination of multiple regularization strategies—subsampling, early stopping, shrinkage, and weak learner constraints—is typically necessary to achieve good generalization with Gradient Boosting.

### Practical Implementation and Hyperparameter Tuning

Successful Gradient Boosting requires careful hyperparameter tuning. Key hyperparameters include the number of iterations (trees), learning rate, tree depth, minimum samples per leaf, and subsampling rates. The learning rate and number of iterations are coupled; lower learning rates benefit from more iterations. A common strategy is to fix the learning rate at a reasonable value (e.g., 0.01-0.1) and use early stopping to determine the number of iterations. Tree depth is often set to 3-8; deeper trees increase model complexity and risk overfitting. Modern implementations like XGBoost, LightGBM, and CatBoost provide additional optimizations and regularization options. XGBoost adds an L1/L2 regularization term to the loss function. LightGBM uses leaf-wise tree growth instead of level-wise, improving efficiency. CatBoost handles categorical features automatically without manual encoding. These implementations have made Gradient Boosting one of the most effective and popular algorithms in practice, winning numerous machine learning competitions.