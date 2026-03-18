# Naive Bayes - Deep Dive Questions

## Question 1: Bayes' Theorem Foundation
Derive Bayes' theorem from first principles. Explain each component: likelihood, prior, evidence, and posterior. How does Naive Bayes apply Bayes' theorem?

## Question 2: Conditional Independence Assumption
What is the naive assumption of conditional independence? Why is it called "naive"? How often is this assumption satisfied in practice? Does violation always hurt performance?

## Question 3: Naive Bayes Variants
When would you use Gaussian vs. Multinomial vs. Bernoulli Naive Bayes? Can you use one variant for a different data type? What happens if you do?

## Question 4: Probability Estimation
How are class priors and feature likelihoods estimated from data? What's the role of smoothing (Laplace smoothing)? Why is smoothing necessary?

## Question 5: Feature Modeling
In Gaussian Naive Bayes, how are features modeled? What assumptions are made about feature distributions? How could you extend this to other distributions?

## Question 6: Interpretability
How can you interpret Naive Bayes predictions probabilistically? Can you extract feature importance? How does interpretability compare to other classifiers?

## Question 7: Scalability and Efficiency
Why is Naive Bayes computationally efficient for training and prediction? How does it scale with number of features and samples? Is it suitable for real-time predictions?

## Question 8: Naive Bayes vs. Logistic Regression
Compare Naive Bayes with logistic regression theoretically. When might Naive Bayes outperform logistic regression despite its independence assumption?

## Question 9: Handling Continuous Features in Multinomial
Naive Bayes assumes discrete features (counts) for the multinomial variant. How would you apply it to continuous text features? What's the connection to TF-IDF?

## Question 10: Advanced Applications
Discuss advanced applications of Naive Bayes: recommendation systems, anomaly detection, information retrieval. How does Naive Bayes extend to these domains?

