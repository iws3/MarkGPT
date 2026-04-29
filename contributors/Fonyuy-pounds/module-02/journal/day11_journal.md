# Day 11: Statistics & Probability for NLP
**Date:** 2026-04-29

## 📝 Learning Objectives
- Understanding Bayes' Theorem and its application in NLP.
- Exploring Maximum Likelihood Estimation (MLE).
- Implementing a Naive Bayes classifier from scratch.
- Understanding the importance of Laplace smoothing for handling OOV (Out Of Vocabulary) words.

## 🏋️‍♂️ Exercises: Naive Bayes from Scratch
**Objective:** Build a classifier to distinguish between Biblical (KJV) and Modern (Non-Biblical) text.

### Implementation Details:
- **Tokenization:** Lowercased text and extracted alphanumeric words.
- **Priors:** Calculated $P(KJV)$ and $P(Modern)$ based on the training set.
- **Likelihoods:** Calculated $P(Word | Class)$ for each word in the vocabulary.
- **Laplace Smoothing:** Added $\alpha=1.0$ to handle words that don't appear in the training data for a specific class.
- **Log-Probabilities:** Used log-probabilities to prevent underflow when multiplying many small probabilities.

### Results:
- **Training Data:** 10 KJV verses and 10 Modern sentences.
- **Test Set Accuracy:** 66.67%.
- **Observation:** The model correctly identified Biblical language like "The spirit of the Lord moved upon the waters" but struggled with modern technical sentences that contained mostly OOV words (e.g., "The software engineer updated the database schema" was misclassified as KJV). This is expected given the extremely small training size where common words like "the" and "of" dominate the probability calculations.

## 🧠 Daily Reflection
**1. Why is Naive Bayes "Naive"?**
> It's called "naive" because it assumes that all features (words in our case) are independent of each other given the class. In reality, language has strong dependencies (e.g., "New York" is more likely than "New London"), but this simplification makes the model extremely efficient and surprisingly effective for text classification.

**2. What is the role of Laplace Smoothing?**
> Without smoothing, if a word appears in the test set but not in the training set for a particular class, its probability $P(Word | Class)$ would be 0. Since we multiply these probabilities (or add log-probs), the entire class probability would become zero (or $-\infty$). Smoothing ensures that every word has a non-zero probability, allowing the model to make predictions even when encountering OOV words.

**3. Maximum Likelihood Estimation (MLE) vs. Bayesian Inference:**
> MLE picks the parameters that maximize the likelihood of the observed data. In our Naive Bayes, the counts were used for MLE. Bayesian inference would involve a prior distribution over parameters, which is essentially what we did by adding the smoothing factor (which can be seen as a Dirichlet prior).
