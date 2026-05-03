# Day 13: Neural Networks from Scratch - The Neuron & Activation Functions
**Date:** 2026-05-03

## 📝 Learning Objectives
- Understanding the biological inspiration for artificial neural networks.
- Implementing the basic unit of deep learning: the Artificial Neuron.
- Exploring different activation functions and their mathematical properties.
- Grasping the necessity of non-linearity in neural networks.

## 🏋️‍♂️ Exercises: Implementing a Neuron
**Objective:** Create a foundation for building multi-layer networks by implementing a single neuron and various activation functions.

### Implementation Details:
- **`Neuron` Class:** I implemented a class that takes $n$ inputs, each with an associated weight $w$, and a bias $b$. The output is calculated as $a = f(\sum w_i x_i + b)$.
- **Activation Functions:** 
    - **Sigmoid:** Maps input to (0, 1). Great for probabilities but suffers from vanishing gradients.
    - **Tanh:** Zero-centered, maps to (-1, 1). Generally better than sigmoid for hidden layers.
    - **ReLU (Rectified Linear Unit):** The modern standard. $f(x) = \max(0, x)$. Extremely efficient and helps mitigate vanishing gradients.
    - **Leaky ReLU:** A variant of ReLU that allows a small gradient when the input is negative, preventing "dead neurons."

### Results:
- Successfully implemented the forward pass logic.
- Tested the neuron with various inputs and confirmed that the activation functions transform the linear sum as expected.
- Observed how different activations squash or threshold the output differently.

## 🧠 Daily Reflection

**1. How does a biological neuron compare to our artificial model?**
> A biological neuron is incredibly complex, involving chemical neurotransmitters and electrical spikes (action potentials). Our artificial model is a simplified mathematical abstraction: dendrites are inputs, synapses are weights, the soma is the summation and activation function, and the axon is the output. While simple, this abstraction is powerful enough to learn complex patterns when stacked in layers.

**2. Why do we need activation functions? Why not just use linear combinations?**
> This is a crucial insight! If we only used linear combinations (matrix multiplications and additions), any number of layers would still be equivalent to a single linear layer. We would only be able to learn linear boundaries. Activation functions introduce **non-linearity**, allowing the network to approximate any complex function (Universal Approximation Theorem).

**3. When should I use ReLU vs. Sigmoid?**
> In modern deep learning, ReLU (or its variants like Leaky ReLU) is the default choice for hidden layers because it trains much faster and doesn't saturate for positive values. Sigmoid is mostly reserved for the output layer of binary classification problems where we need a probability between 0 and 1.

## 🚀 Next Steps
Tomorrow, I'll move from a single neuron to **Multi-Layer Perceptrons (MLPs)** and explore how these units work together to solve non-linear problems like XOR.
