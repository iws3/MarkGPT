# Day 07: Python for Machine Learning
**Date:** 2026-04-22

## 📝 Learning Objectives
- Understanding Python data types for ML.
- Working efficiently with NumPy arrays and vectorized operations.
- Concept of Array Broadcasting.

## 🏋️‍♂️ Exercises: NumPy vs Manual Loops
**1. Matrix Multiplication Benchmarking:**
- Manual (Nested Loops) Execution Time: ~0.15200 seconds (for 100x100 array)
- NumPy Execution Time: ~0.00120 seconds

*Why is NumPy significantly faster when doing operations computationally?*
> NumPy operations execute in heavily optimized, pre-compiled C routines. This avoids the massive overhead of Python's dynamic type-checking, reference counting, and interpreter loop execution per element. Additionally, it vectorizes operations leveraging SIMD (Single Instruction, Multiple Data) CPU capabilities.

**2. Softmax Implementation:**
*How does Softmax normalization differ from just dividing by the sum? Why does scaling probabilities this way matter in Machine Learning?*
> Softmax scales using exponentiation, which exaggerates differences. Large logits (raw scores) become disproportionately larger compared to smaller ones before normalization, thus driving the highest probability towards 1 rapidly while suppressing the lower ones. This is crucial for models to make distinct, "confident" classifications out of continuous vector spaces.

## 🧠 Daily Reflection
**1. What surprised you the most about the speed scale between Python native iterations and NumPy's C-compiled vector arrays?**
What is most surprising is seeing raw math that takes centuries of CPU cycles strictly iterating through loops finish practically instantly. Recognizing that Python lists are essentially continuous arrays of "pointers" rather than strictly adjacent system memory really highlights the hidden overhead in dynamic languages.

**2. Where might vectorizations apply best within the architecture of a theoretical Language Model?**
Vectorization will be profoundly mandatory in calculating the "Attention Mechanisms" and processing the layers of neurons. Whenever the model computes thousands of dot-products to match tokens against memory contexts, it would be impossible without NumPy/PyTorch massive vectorized matrix computations.
