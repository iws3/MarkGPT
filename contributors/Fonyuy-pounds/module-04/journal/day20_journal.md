# Day 20 Journal: Recurrent Neural Networks (RNNs)

## Daily Setup
- **Date**: May 10, 2026
- **Module**: 04 - Recurrent Networks & Sequence Modeling
- **Topic**: Recurrent Neural Networks (RNNs)
- **Time Spent**: 4 hours

## Goals for Today
- [x] Understand the architecture of a basic RNN cell
- [x] Learn about Backpropagation Through Time (BPTT)
- [x] Investigate the vanishing gradient problem in vanilla RNNs
- [x] Implement gradient clipping to handle exploding gradients
- [x] Train a character-level RNN on the Book of John

## Notes and Learnings

### The RNN Cell
A vanilla RNN processes sequences by applying the same transformation at each time step. The hidden state $h_t$ is updated using the current input $x_t$ and the previous hidden state $h_{t-1}$:
$$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$
The output $y_t$ is typically derived from $h_t$:
$$y_t = W_{ho} h_t + b_{ho}$$

### Backpropagation Through Time (BPTT)
BPTT is just standard backprop applied to the unrolled computation graph. Because the same weights are used at every step, the gradient of the loss with respect to a weight is the sum of gradients from all time steps where that weight was used.

### The Vanishing Gradient Problem
As we backpropagate through many steps, we multiply by the weight matrix $W_{hh}$ and the derivative of the activation function repeatedly.
- If eigenvalues of $W_{hh}$ are $< 1$, gradients vanish exponentially.
- If eigenvalues are $> 1$, gradients explode.
This makes it very hard for vanilla RNNs to learn dependencies that span more than a few steps.

### Gradient Clipping
To prevent exploding gradients, we can rescale the gradient if its norm exceeds a threshold:
$$\text{if } \|\mathbf{g}\| > \text{threshold, then } \mathbf{g} = \frac{\text{threshold}}{\|\mathbf{g}\|} \mathbf{g}$$

## Exercises
### E20.1: Character-level RNN on Book of John
I implemented a vanilla RNN using `numpy` and trained it on a extracted version of the Gospel of John from the KJV Bible.

**Training Parameters:**
- Vocab Size: 70 characters
- Hidden Size: 100 units
- Sequence Length: 25 characters
- Learning Rate: 0.1 (AdaGrad)
- Iterations: 5000

**Progress:**
- **Iter 0**: Loss 106.21 (Random noise)
- **Iter 500**: Loss 97.37 (Some "the" and spacing patterns)
- **Iter 2500**: Loss 61.42 (Starting to form words like "said", "into")
- **Iter 5000**: Loss 51.25 (Verse structures appearing: "5:30 Nithand...")

### E20.2: Observations on Generated Text
The text generated after 5000 iterations shows that the model has captured:
1. **Basic Word Structure**: It creates sequences that look like English words, even if they aren't real words (e.g., "hous wiveve de").
2. **Verse Patterns**: It learned the `X:Y` numbering format common in the KJV Bible.
3. **Common Words**: "the", "he", "and", "into" appear frequently and correctly.
4. **Incoherence**: Over long distances, the text is completely nonsensical. This is a classic symptom of the vanishing gradient problem; the model can't remember the beginning of a sentence by the time it reaches the end.

## Reflection
Today's deep dive into RNNs was eye-opening. While the concept of a "hidden state" acting as memory is powerful, the practical limitations of vanilla RNNs are stark. BPTT is conceptually simple but computationally tricky due to the recurring weight multiplications.

Gradient clipping proved essential—without it, the loss would occasionally spike to infinity early in training. However, even with clipping, the model's "memory" is very short-lived. I'm looking forward to Day 21, where we explore Gated Recurrent Units (GRUs) and LSTMs to see how they solve these memory bottlenecks.

