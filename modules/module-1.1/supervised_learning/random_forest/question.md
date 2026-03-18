# Random Forest - Deep Dive Questions

## Question 1: Ensemble Method Fundamentals
Explain bootstrap aggregation (bagging) and how Random Forest uses it. Why does combining multiple trees reduce variance? Derive the variance reduction formula.

## Question 2: Random Feature Selection
Why does Random Forest randomly select features at each split? How does this differ from searching for the globally optimal split? What are the computational and statistical benefits?

## Question 3: Out-of-Bag Error
What is Out-of-Bag (OOB) error and how is it calculated? Why is OOB valuable? Can it replace cross-validation? What are its advantages and limitations?

## Question 4: Feature Importance Interpretation
How are feature importances calculated in Random Forests? What does importance measure? Can you trust feature importance rankings? When might they be misleading?

## Question 5: Bias-Variance Trade-off
How do Random Forests manage the bias-variance trade-off compared to single decision trees? How do hyperparameters like max_depth and n_estimators affect this trade-off?

## Question 6: Parallelization
Why is Random Forest naturally parallelizable? How does parallel training affect training time and final accuracy? Are there any downsides to parallelization?

## Question 7: Computational Complexity
What is the time and space complexity of Random Forest training and inference? How does it scale with number of trees, features, and samples?

## Question 8: Handling Missing Values and Categorical Data
How do Random Forests handle missing values? What strategies can you use for categorical features? How does this compare to other algorithms?

## Question 9: Extrapolation and Generalization
Can Random Forests extrapolate beyond the training data range? Why or why not? How does this affect their use for time-series prediction?

## Question 10: Advanced Topics
Compare Random Forest with XGBoost, LightGBM, and CatBoost. What are the fundamental differences? When would you prefer gradient boosting over Random Forest?

