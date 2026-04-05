# Module 04 — Recurrent Networks & Sequence Modeling
## Days 19–24 | Intermediate

---

## Module Overview

Sequences are everywhere in language: a sentence is a sequence of words, a word is a sequence of characters. This module teaches how to build networks that remember, using RNNs, LSTMs, and the first forms of attention.

By the end of Module 04, you will:
- Understand how RNNs maintain hidden state over time
- Implement the vanishing gradient problem and see it firsthand
- Build an LSTM that actually remembers long sequences
- Use attention to look back selectively through a sequence

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
## Recurrent Neural Networks - Fundamentals

### What is a Recurrent Network?

Standard feedforward: x → h → output
Recurrent: x_t → h_t → output_t
h_t depends on x_t AND h_{t-1}
Hidden state carries information from past.

Processing sequences: One element at a time
Weights shared across time steps (parameter efficiency).
### Hidden State and Time Unrolling

h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
o_t = W_o @ h_t + b_o

Time unrolling: Unfold RNN for T time steps
Creates deep feedforward network (depth = sequence length)
Backprop through time (BPTT): Chain rule across time.

## Sequence to Output

### Many-to-One (e.g., sentiment)

Input: Sequence of 100 words
Process: Apply RNN at each step
Output: Only use final h_T for classification
Loss computes only on last output.
Gradient flows backward through all steps.

### One-to-Many (e.g., image captioning)

Input: Single image
Process: Encode to h_0
Output: Generate caption word by word
h_0 from CNN → fed to RNN
Each RNN step outputs word token.

### Many-to-Many (e.g., NER, machine translation)

Input: Sequence of N tokens
Process: RNN processes all
Output: Sequence of N predictions
Each time step has input and output.
Examples: Sequence labeling, translation.

## The Vanishing Gradient Problem

### The Issue

BPTT: Chain rule multiplies gradients
∂L/∂h_0 = (∂L/∂h_T) * (∂h_T/∂h_{T-1}) * ... * (∂h_1/∂h_0)
Each ∂h_t/∂h_{t-1} < 1 typically
Product of T < 1 terms → exponentially small
Gradient for h_0 becomes nearly 0.

### Why This Matters

Early inputs get negligible gradients.
Model forgets distant past (effective window ~5-20 steps).
Long-range dependencies can't be learned.
Example: Pronoun in position 1, reference at position 50.
RNN unlikely to learn this dependency.

