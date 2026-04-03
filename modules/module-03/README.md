# Module 03 — Neural Networks from Scratch
## Days 13–18 | Intermediate

---

## Module Overview

This module builds your intuition for how neural networks learn. You will implement a neuron, then a multi-layer perceptron, then the magic of backpropagation — all from scratch, in plain Python and NumPy.

By the end of Module 03, you will:
- Understand the neuron as a mathematical object
- Implement forward and backward passes
- Solve non-linear problems (XOR) that n-gram models cannot
- Grasp why depth matters in neural networks

## Learning Objectives

- Understand core ML concepts
- Implement algorithms from scratch
- Relate theory to MarkGPT architecture
- Complete hands-on exercises

## Structure

```
lessons/       - Conceptual explanations with code examples
exercises/     - Practical implementation exercises
projects/      - Larger projects (optional)
resources/     - Additional readings and links
```

## Time Estimate

- Lessons: 4-6 hours
- Exercises: 4-6 hours
- **Total: 8-12 hours per module**

## Key Concepts

[See lesson files for detailed content]

## Completion Checklist

- [ ] Read all lessons (L*_*.md files)
- [ ] Complete all exercises (day*_*.md files)
- [ ] Pass the module quiz (if provided)
- [ ] Understand connections to MarkGPT

## Resources

- Lesson references contain links to papers and tutorials
- http://markgpt-docs.com (forthcoming)
- GitHub discussions: https://github.com/yourusername/MarkGPT-LLM-Curriculum/discussions

## Next Module

See ../module-0$((i+1))/README.md for the next module.


## Neuron Architecture and Fundamentals

### The Biological Perspective

Artificial neurons are inspired by biological neurons:
- **Dendrites**: Receive signals (inputs)
- **Cell Body**: Process information (weighted sum)
- **Axon**: Send output signal
- **Synapse**: Connection strength (weights)

**Firing Mechanism**
Biological neuron fires when activation exceeds threshold.
Artificial neuron: Apply activation function to weighted sum.

### Mathematical Definition of a Neuron

**Linear Combination**
$$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$

Where:
- $x_i$: Input features
- $w_i$: Weights (synaptic strengths)
- $b$: Bias (threshold shift)
- $z$: Pre-activation (logit)

**Activation**
$$a = \sigma(z)$$

Where $\sigma$ is activation function (ReLU, sigmoid, tanh, etc.)

