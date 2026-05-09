# Day 19 Journal: Sequences and the Problem of Memory

## Daily Setup
- **Date**: May 9, 2026
- **Module**: 04 - Recurrent Networks & Sequence Modeling
- **Topic**: Sequences & Memory (The limits of feedforward)
- **Time Spent**: 

## Goals for Today
- [ ] Understand why fixed-context MLPs fail on long sequences
- [ ] Learn the concept of "Hidden State"
- [ ] Implement a manual sequence processor using an MLP
- [ ] Observe "Catastrophic Forgetting" in simple sequential processing

## Notes and Learnings

### The Problem with Fixed Windows
In Day 18, we built an MLP that looked at 5 characters to predict the 6th. While this works for local patterns (like spelling "Lord"), it has no memory of what happened 10, 20, or 100 characters ago.
- If we want to capture long-range dependencies (e.g., a subject at the start of a sentence and its verb at the end), a fixed window would need to be massive.
- Massive windows lead to parameter explosion in MLPs.

### The Hidden State Concept
Instead of a fixed window, what if we maintain a "summary" of everything we've seen so far?
- This summary is called the **Hidden State** ($h$).
- $h_t = f(x_t, h_{t-1})$
- This recursive definition allows information to theoretically flow from the very first token to the current one.

### Unrolling Through Time
Thinking about sequences as a single "unrolled" graph where the same weights are applied at every step. This leads to the idea of Backpropagation Through Time (BPTT).

## Exercises
- **E19.1**: Manual sequence processor. We'll feed a sequence into an MLP step-by-step, updating a state vector, and see how much information is retained at the end.

## Reflection
(To be completed)
