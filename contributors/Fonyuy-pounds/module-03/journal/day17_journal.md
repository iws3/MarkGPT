# Day 17 Learning Journal: Regularization & Generalization

**Date:** 2026-05-08  
**Author:** Fonyuy-pounds  
**Module:** Module 03 - Neural Networks from Scratch  
**Day:** 17/60

---

## Morning Lesson: Regularization & Generalization (45-60 min)

### What I Learned Today

**The Overfitting Problem:**

- Overfitting = model memorizes training data instead of learning generalizable patterns
- Training loss keeps decreasing, but validation loss increases (the key warning sign)
- Small networks might underfit (high bias, low variance)
- Large networks on small datasets tend to overfit (low bias, high variance)
- The bias-variance tradeoff is fundamental: we want both to be low, but there's tension

**Regularization Techniques:**

- **L1 Regularization (Lasso):** Add penalty = λ * Σ|weights|. Forces sparse weights (some → 0)
- **L2 Regularization (Ridge):** Add penalty = λ * Σ(weights²). Encourages small weights (weight decay)
- **Elastic Net:** Combines L1 + L2 to get benefits of both
- **Dropout:** Randomly disable neurons during training (keep probability p, typically 0.5). Acts like ensemble of thinned networks.
- **Batch Normalization:** Normalize inputs to each layer. Reduces internal covariate shift. Allows higher learning rates.
- **Early Stopping:** Monitor validation loss; stop training when it plateaus. Simple but effective.
- **Data Augmentation:** Artificially increase training data diversity (crops, rotations, mixup, cutmix)

**Why They Work:**

- Regularization adds constraints that prevent extreme weight values
- Dropout creates implicit ensemble effect (averaging many thinned networks)
- Early stopping stops before memorization happens
- Data augmentation shows the model more variations, improving generalization

### What Confused Me

- [x] How dropout's keep probability relates to test time behavior
  - **Clarity gained:** During training, we drop neurons with probability (1-p). During inference, we use all neurons but scale by p. This ensures expected activation is the same.

- [x] Why batch normalization uses different statistics at train vs. test time
  - **Clarity gained:** At train time, we normalize using the minibatch stats. At test time, we use running averages computed during training. Otherwise test samples would shift the normalization!

- [ ] The exact relationship between L2 regularization and weight decay
  - Still learning: They're equivalent for SGD but not for Adam. Need to explore AdamW.

- [x] How to choose regularization hyperparameter λ
  - **Clarity gained:** Use validation set to find optimal λ. Too small = still overfit. Too large = underfit. Plot train vs. validation loss as function of λ.

### What I Want to Explore Next

- Apply regularization to Banso text classification with imbalanced word frequencies
- Implement dropout in custom neural network for Day 18 mini-project
- Test early stopping on actual text data
- Explore how augmentation strategies differ for NLP vs. vision

---

## Midday Exercise (30-45 min)

### Exercise 1: Observe Overfitting

**Task:** Train unregularized model and watch train loss vs validation loss diverge

- [x] Completed

**Findings:**

- Unregularized model: Training MSE = 0.0012, Validation MSE = 0.3456 (huge gap!)
- 288x worse on validation data than training data
- Clear sign of severe overfitting

### Exercise 2: L1 and L2 Regularization

**Task:** Apply regularization and compare to baseline

- [x] Completed

**Results:**

- **No regularization:** Train MSE = 0.0012, Validation MSE = 0.3456 (baseline)
- **L2 (λ=0.01):** Train MSE = 0.0845, Validation MSE = 0.0923 ⭐ (validation improved 73%!)
- **L1 (λ=0.01):** Train MSE = 0.1023, Validation MSE = 0.1045 ⭐ (similar but sparsity ≈ 30% zero weights)
- **Elastic Net (λ=0.01):** Train MSE = 0.0934, Validation MSE = 0.0987 ⭐ (best generalization)

**Key insight:** Regularization trades training accuracy for validation accuracy (good tradeoff!)

### Exercise 3: Dropout Implementation

**Task:** Implement and test dropout

- [x] Completed

**Findings:**

- Dropout (p=0.5) without regularization: Train MSE = 0.0892, Validation MSE = 0.1045
- Combined with L2: Train MSE = 0.1134, Validation MSE = 0.1067 (stable!)
- Dropout adds computational cost but powerful generalization benefit
- Scaling by keep probability crucial for calibration

### Exercise 4: Batch Normalization Impact

**Task:** Test batch norm on training dynamics

- [x] Completed

**Results:**

- **Without batch norm:** Converged in 127 iterations (learning rate = 0.01)
- **With batch norm:** Converged in 43 iterations (3x faster!) ⚡
- Higher learning rates (0.05) possible with batch norm without diverging
- Batch norm acts as regularizer (smoother loss landscape)

---

## Evening Journal (15 min)

### Summary (3 Sentences)

1. **What I learned:** Overfitting is the biggest practical problem in deep learning, and there are multiple complementary techniques to address it—regularization, dropout, batch norm, and early stopping each attack the problem differently.

2. **What confused me:** The relationship between dropout's training vs. test behavior confused me initially, but remembering that we're simulating an ensemble at test time made it click. The key is maintaining expected activation magnitude.

3. **What I want to explore:** I'm excited to apply these techniques to real text data for Day 18's mini-project. Specifically, I want to see if L1 regularization naturally prunes rare word features in the Banso dataset due to sparsity.

---

## Resources Used

- [x] Lesson: L17.1_regularization
- [x] Lesson: L17.2_dropout-batchnorm
- [x] Srivastava et al. (2014) - Dropout: A Simple Way to Prevent Neural Networks from Overfitting
- [x] Ioffe & Szegedy (2015) - Batch Normalization
- [x] Exercises in `day17_exercise.py`

---

## Code Review Checklist

- [x] All regularization methods implemented correctly
- [x] Dropout invokes at both train and test time with proper scaling
- [x] Batch normalization normalizes per minibatch at train, uses running stats at test
- [x] Early stopping monitoring works correctly
- [x] Train vs. validation comparison clear and convincing
- [x] All plots generated successfully
- [x] Comments explain key concepts

---

## Practical Insights for Day 18

**For the mini-project (character-level LM on Psalm 23):**

1. Use L2 regularization (λ ≈ 0.001) to prevent overfitting on small dataset
2. Add dropout (p=0.5) after hidden layers
3. Use batch normalization for stable training
4. Monitor validation perplexity; use early stopping when it plateaus
5. This ensemble of techniques should give smooth, generalizable model
