# Day 15: Backpropagation: The Heart of Learning
**Date:** 2026-05-05

## 📝 Learning Objectives
- Understand the computation graph and how errors propagate backwards.
- Grasp the chain rule of calculus as the engine of deep learning.
- Derive and implement backpropagation for a Multi-Layer Perceptron from scratch.

## 🏋️‍♂️ Exercises: Implementing Backpropagation
**Objective:** Train the 2-layer MLP from Day 14 on the XOR problem using backpropagation instead of hand-coded weights.

### Implementation Details:
- **Forward Pass Revisited:** We store intermediate values (activations `A` and pre-activations `Z`) during the forward pass. This is crucial because these values are required to compute the gradients later.
- **The Chain Rule:** To update a weight $W_{ij}$ deep in the network, we need to know how a change in $W_{ij}$ affects the final loss $L$. The chain rule tells us we can compute this by multiplying local gradients backward from the output to the input:
  $$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} \cdot \frac{\partial Z_1}{\partial W_1}$$
- **The Backward Pass:** 
    1. Computed the gradient of the loss with respect to the output layer's pre-activation (`dZ2`).
    2. Computed the gradients for the output layer's weights and biases (`dW2`, `db2`).
    3. Backpropagated the error to the hidden layer (`dA1`, `dZ1`).
    4. Computed the gradients for the hidden layer's weights and biases (`dW1`, `db1`).
- **Optimization:** Used standard Gradient Descent (`W = W - learning_rate * dW`) to update the parameters.

### Results:
- The network successfully learned the XOR function! The loss steadily decreased over 10,000 epochs.
- The final predictions were very close to the target values, demonstrating that the network successfully discovered the correct weights for the non-linear boundaries.
- This proves that we don't need to hand-code features; the network can *discover* the necessary internal representations (like AND/OR logic) entirely on its own through backpropagation.

## 🧠 Daily Reflection

**1. What is the "computation graph" and why is it useful?**
> A computation graph visualizes mathematical expressions as a series of nodes (operations) and directed edges (data flow). It's incredibly useful because it breaks down a massive, complex derivative into simple, local operations. Every node only needs to know how to compute its own output and its own local derivative. This modularity is what makes automatic differentiation (like in PyTorch) possible!

**2. Why was implementing backpropagation by hand important?**
> In modern deep learning, we rarely write backprop by hand—frameworks do it for us. However, implementing it from scratch demystifies the "magic." I now understand *exactly* why vanishing gradients happen (multiplying many small `< 1` derivatives together) and why we need to cache intermediate activations during the forward pass (which costs memory).

**3. The Beauty of Credit Assignment**
> Backpropagation solves the fundamental problem of credit assignment: when the network makes a mistake, whose fault is it? Backprop mathematically assigns blame to each weight proportional to its contribution to the final error, telling it precisely which direction to adjust.

## 🚀 Next Steps
Now that the network can learn, I'll move on to more advanced loss functions and optimizers like Adam to make training faster and more stable!
