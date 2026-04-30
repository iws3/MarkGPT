# Day 10: Calculus for Optimization
**Date:** 2026-04-28

## 📝 Learning Objectives
- Understanding derivatives and the chain rule (crucial for backpropagation).
- Gaining intuition for gradient descent.
- Differentiating between local vs. global minima.

## 🏋️‍♂️ Exercises: Gradient Descent
**Objective:** Minimize $f(x) = x^4 - 4x^2 + x$ using gradient descent.

**1. Derivative Calculation:**
The derivative of the function is:
$f'(x) = 4x^3 - 8x + 1$

**2. Gradient Descent Experiments:**
By applying the update rule $x_{new} = x_{old} - lr \cdot f'(x_{old})$, we tested different learning rates starting from $x=0$.

- **Which learning rate diverges?**
  A learning rate of `0.5` diverges. The step size is too large, causing the algorithm to overshoot the minimum. The values of $x$ oscillate with increasing magnitude, rapidly heading towards infinity.
  
- **Which converges slowly?**
  A learning rate of `0.001` or `0.01` converges slowly. Because the step sizes are tiny, the algorithm requires many iterations to reach the local minimum.
  
- **Which converges well?**
  A learning rate of `0.1` converges relatively quickly and stably to a local minimum.

## 🧠 Daily Reflection
**1. Why is the chain rule crucial for backpropagation?**
> In neural networks, a prediction is the result of many nested functions (layers). The chain rule allows us to calculate the derivative of the final loss with respect to any individual weight by multiplying the local gradients backwards from the output to the input. Without the chain rule, we couldn't efficiently determine how to adjust early layers based on the final error.

**2. Local vs. Global Minima:**
> The function $f(x) = x^4 - 4x^2 + x$ has two local minima. Gradient descent is a greedy algorithm, so it only moves downhill. Depending on the starting point (e.g., $x=0$), it might converge to a local minimum that is not the global minimum. This highlights why techniques like momentum and stochasticity are useful in training complex neural networks, helping them escape suboptimal local minima.
