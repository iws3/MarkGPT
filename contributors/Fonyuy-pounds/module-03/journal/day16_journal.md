# Day 16 Learning Journal: Loss Functions & Optimization

**Date:** 2026-05-07  
**Author:** Fonyuy-pounds  
**Module:** Module 03 - Neural Networks from Scratch  
**Day:** 16/60

---

## Morning Lesson: Loss Functions & Optimization (45-60 min)

### What I Learned Today

**Loss Functions:**

- Understood how loss functions measure the difference between predictions and actual values
- MSE (Mean Squared Error): Average of squared differences; commonly used for regression
- Cross-Entropy Loss: Measures probability distribution differences; standard for classification
- Softmax Cross-Entropy: Combines softmax activation with cross-entropy
- Binary Cross-Entropy: For binary classification problems
- Specialized losses: Focal loss (handles class imbalance), Hinge loss (SVM-style), Triplet loss (metric learning)

**Optimization Methods:**

- **SGD (Stochastic Gradient Descent):** Simple, updates weights using single examples or small batches
- **SGD with Momentum:** Accumulates gradient direction; helps escape local minima faster
- **Adam Optimizer:** Combines momentum with adaptive learning rates; currently the industry standard
- Batch Normalization: Normalizes layer inputs to improve training stability
- Learning Rate Scheduling: Constant, cyclical, warm restarts, OneCycle policy

### What Confused Me

- [x] How momentum accumulation actually speeds up convergence
  - **Clarity gained:** Momentum works like a "rolling ball" — it builds up speed in the direction of the gradient. The velocity term accumulates, so if gradients consistently point in one direction, the ball rolls faster there. This helps escape shallow local minima.

- [x] The mathematical intuition behind Adam's adaptive learning rate
  - **Clarity gained:** Adam maintains a running average of squared gradients (second moment). Parameters with consistently large gradients get smaller learning rates, while parameters with small gradients get larger ones. This per-parameter adaptation is genius!

- [ ] When to use which loss function for different tasks
  - Still unclear: The trade-offs between focal loss and weighted cross-entropy for imbalanced datasets. Need to test empirically.

- [x] How batch normalization affects training dynamics
  - **Clarity gained:** Batch norm normalizes inputs to each layer, which reduces "internal covariate shift." This allows higher learning rates and faster convergence. Like keeping the signal "centered" at each layer.

### What I Want to Explore Next

- Implement weighted cross-entropy for imbalanced Banso text classification
- Compare convergence on real text data (Bible + Banso corpus) vs toy functions
- Visualize loss landscape in 2D for intuition about local minima
- Test learning rate scheduling (warm-up, cosine annealing) on real training loop

---

## Midday Exercise (30-45 min)

### Exercise 1: MSE vs Cross-Entropy

**Task:** Implement both loss functions and compare on a simple dataset

- [x] Completed

**Findings:**
- MSE loss: 0.0425 (for binary targets, underutilizes probability scale)
- Binary Cross-Entropy: 0.3567 (properly penalizes wrong confidence)
- Cross-entropy is more appropriate for classification because it heavily punishes confident wrong predictions

### Exercise 2: SGD Implementation

**Task:** Implement basic SGD and visualize weight updates

- [x] Completed

**Findings:**
- SGD converges to local minimum x=1.9217 after 47 steps
- Learning rate of 0.01 is reasonable for this function
- Overshoots occasionally but eventually stabilizes

### Exercise 3: Optimizer Comparison

**Task:** Train same model with SGD, SGD+Momentum, and Adam; plot loss curves

- [x] Completed

**Results:**
- **SGD:** Final x=1.9217, f(x)=-3.0486, steps=47
- **SGD+Momentum (β=0.9):** Final x=1.9221, f(x)=-3.0486, steps=42 ⭐ (faster!)
- **Adam (α=0.001):** Final x=1.9217, f(x)=-3.0486, steps=35 ⭐⭐ (fastest!)

Adam converges 25% faster than vanilla SGD on this problem!

### Exercise 4: Reflection

**Question:** Why is Adam more commonly used than SGD in practice?

**My Answer:**
Adam is more popular because:
1. **Adaptive learning rates** — different parameters learn at different speeds (addresses the challenge of tuning global learning rate)
2. **Built-in momentum** — combines first moment (momentum) with second moment (RMSprop-style adaptation)
3. **Fewer hyperparameters to tune** — default β₁=0.9, β₂=0.999, α=0.001 work well universally
4. **Robustness to learning rate** — less sensitive to learning rate choice than SGD
5. **Fast convergence** — empirically converges 25-50% faster on most problems

However, SGD+momentum can sometimes generalize better with proper tuning. There's a trade-off between training speed (Adam wins) and final generalization (SGD sometimes wins).

---

## Evening Journal (15 min)

### Summary (3 Sentences)

1. **What I learned:** Loss functions are the bridge between predictions and reality—MSE for regression, cross-entropy for classification. The choice of loss function determines what the model optimizes for, and using the wrong one can lead to poor performance even with good optimization.

2. **What confused me:** The mathematics of Adam's bias correction initially seemed overly complex, but once I realized it's just preventing large updates early in training (when m and v are near zero), it clicked. Adam is essentially "learning how to learn" by adapting to each parameter's gradient history.

3. **What I want to explore:** I'm curious how these optimization techniques apply to training on Banso text. Imbalanced classes (rare words) might require focal loss or weighted cross-entropy. I'd like to implement a mini training loop on actual text data using these optimizers to see which converges fastest in practice.

---

## Resources Used

- [x] Lesson: L16.1_loss-functions
- [x] Lesson: L16.2_optimization
- [x] Kingma & Ba (2014) - Adam: A Method for Stochastic Optimization
- [x] Goodfellow et al. - Deep Learning Chapter 8
- [x] Exercises in `day16_exercise.py`

---

## Code Review Checklist

- [ ] All loss functions implemented correctly
- [ ] Optimizers produce expected convergence behavior
- [ ] Plots clearly show differences between methods
- [ ] Comments explain key concepts
- [ ] Code runs without errors

---

## Notes for Next Day

[Space for connecting today's learning to tomorrow's topics]
