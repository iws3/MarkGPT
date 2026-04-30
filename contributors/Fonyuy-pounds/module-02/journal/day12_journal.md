# Day 12: Module 02 Review & Environment Setup
**Date:** 2026-04-30

## 📝 Learning Objectives
- Setting up a Deep Learning environment (PyTorch).
- Understanding the tensor abstraction as the fundamental data structure for Neural Networks.
- Leveraging PyTorch's `autograd` for automatic gradient computation.
- Building a complete text preprocessing pipeline for downstream training.

## 🏋️‍♂️ Exercises: PyTorch Basics
**Objective:** Familiarize with PyTorch tensors and autograd by training a single neuron.

### Key Takeaways:
- **Tensors vs. NumPy Arrays:** Tensors are very similar to NumPy arrays but they can reside on accelerators like GPUs. More importantly, they can track operations via `requires_grad=True`.
- **Autograd:** Calling `.backward()` on a scalar loss automatically computes the gradients for all tensors involved in its computation that require gradients. This is a massive leap from having to calculate derivatives by hand (like in Day 10).
- **Single Neuron Training:** I successfully trained a single neuron (Linear layer) using Mean Squared Error (MSE) and Stochastic Gradient Descent (SGD) to learn a simple linear mapping $y = 2x + 1$. 

## 🚀 Mini-Project 2: Text Preprocessing Pipeline
**Objective:** Build a robust, object-oriented text processing pipeline for the MarkGPT data (Bible + Banso texts).

### Implementation Details:
- **`TextPipeline` Class:** Developed a custom class that handles cleaning, tokenization, vocabulary building, and encoding/decoding.
- **Special Tokens:** Included `<PAD>` (for padding sequences to uniform length) and `<UNK>` (for handling out-of-vocabulary words during testing/inference).
- **Data Splitting:** Implemented a standard ML data split: 60% Training, 20% Validation, and 20% Test sets to ensure robust evaluation later.
- **Result:** The pipeline successfully takes raw text and outputs clean, tokenized integer arrays ready to be fed into PyTorch `Dataset` and `DataLoader` objects.

## 🧠 End of Module 02 Reflection
Module 02 effectively bridged the gap between pure mathematics and code. By implementing PCA, gradient descent, and Naive Bayes from scratch, I gained a deep appreciation for the mechanics "under the hood." Transitioning to PyTorch on Day 12 feels empowering; the mathematical concepts I've learned are now seamlessly handled by the framework, freeing me to focus on architecture and data pipelines. I am ready for Module 03 (Neural Networks from Scratch).
