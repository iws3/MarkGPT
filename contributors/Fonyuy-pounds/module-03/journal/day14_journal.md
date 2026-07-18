# Day 14: Neural Networks from Scratch - Multi-Layer Perceptrons (MLPs)
**Date:** 2026-05-04

## 📝 Learning Objectives
- Understanding the limitations of a single neuron.
- Discovering how hidden layers introduce the capacity to learn non-linear decision boundaries.
- Exploring the Universal Approximation Theorem.
- Implementing a Multi-Layer Perceptron (MLP) and performing a forward pass.

## 🏋️‍♂️ Exercises: Solving the XOR Problem
**Objective:** Implement a 2-layer MLP and solve the classic XOR problem to demonstrate the representational power of depth.

### Implementation Details:
- **`Layer` Class:** I implemented a class that handles multiple neurons simultaneously using matrix operations (`z = X · W + b`), significantly speeding up the forward pass compared to iterating through individual neurons.
- **`MLP` Class:** A container class that chains multiple `Layer` instances together. The output of one layer becomes the input to the next.
- **Solving XOR:** A single neuron (linear classifier) cannot solve the XOR problem because the classes are not linearly separable. By adding a hidden layer with 2 neurons, the network essentially learns two new features (e.g., an OR gate and an AND gate). The output layer then combines these features (OR and NOT AND) to solve the XOR problem.
- **Decision Boundary Visualization:** I visualized the predictions of the MLP across a 2D grid, showing a distinct non-linear boundary that successfully separates the true classes from the false ones.

### Results:
- Manually set the weights of the hidden layer and output layer to mimic logical gates (OR and AND).
- The network successfully mapped the XOR inputs:
    - [0, 0] -> ~0
    - [0, 1] -> ~1
    - [1, 0] -> ~1
    - [1, 1] -> ~0
- The decision boundary visually confirmed that the hidden layer warps the input space, allowing the output layer to draw a straight line through the transformed space.

## 🧠 Daily Reflection

**1. Why was XOR such a big deal historically?**
> In the 1960s, Minsky and Papert proved that single-layer perceptrons could not learn XOR. This caused the first "AI Winter" because researchers mistakenly believed that neural networks were fundamentally flawed. It wasn't until backpropagation allowed for the training of *multi-layer* networks that this limitation was overcome, proving that hidden layers give networks immense representational power.

**2. What is the Universal Approximation Theorem?**
> It's the mathematical proof that a neural network with at least one hidden layer and a non-linear activation function can approximate *any* continuous function, provided it has enough neurons. This is why deep learning is so powerful—it can theoretically learn any mapping from inputs to outputs!

**3. What changes if we add a third layer?**
> A 2-layer network (1 hidden layer) can learn a continuous boundary. Adding a third layer (2 hidden layers) allows the network to learn combinations of boundaries, meaning it can approximate highly complex, discontinuous regions or intricate manifolds in the data space. It makes the network more parameter-efficient at representing complex functions.

## 🚀 Next Steps
Now that I have a working MLP capable of forward passes, the next logical step is to figure out how to *train* it automatically. Tomorrow, I'll dive into the heart of modern deep learning: **Backpropagation** and the chain rule!
