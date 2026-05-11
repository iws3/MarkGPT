# Day 21 Journal: LSTM Cells and Gated Architectures

## Daily Setup
- **Date**: May 11, 2026
- **Module**: 04 - Recurrent Networks & Sequence Modeling
- **Topic**: LSTMs and GRUs
- **Time Spent**: 5 hours

## Goals for Today
- [x] Understand the gating mechanisms in LSTMs (Forget, Input, Output)
- [x] Learn how the "Cell State" acts as a long-term memory conveyor belt
- [x] Implement an LSTM from scratch using NumPy
- [x] Compare LSTM convergence and text generation with vanilla RNNs
- [x] Explore the GRU architecture as a simplified alternative

## Notes and Learnings

### The LSTM Solution
The Long Short-Term Memory (LSTM) network was designed specifically to address the vanishing gradient problem. By using an additive update for the cell state, gradients can flow through time steps without being repeatedly multiplied by weights that might be small.

#### Key Components:
1. **Forget Gate ($f_t$)**: Decides what information to discard from the previous cell state.
2. **Input Gate ($i_t$)**: Decides which new values we'll update the state with.
3. **Candidate ($g_t$ or $\tilde{C}_t$)**: A vector of new values that could be added to the state.
4. **Output Gate ($o_t$)**: Decides what the next hidden state should be based on the cell state.

### LSTM vs. GRU
While LSTMs have three gates and a separate cell state, Gated Recurrent Units (GRUs) simplify this:
- **Reset Gate**: Determines how much of the past to forget.
- **Update Gate**: Combines the forget and input gates.
- No separate cell state; only a hidden state.

GRUs are computationally faster and often perform comparably to LSTMs, though LSTMs are sometimes better at capturing very long-term dependencies.

## Exercises

### E21.1: Character-level LSTM on Book of John
I implemented an LSTM from scratch and trained it on the same dataset as yesterday's RNN.

**Observations:**
- **Convergence**: The LSTM converged slightly slower in terms of clock time per iteration (due to more matrix multiplications), but the loss dropped more consistently.
- **Memory**: The generated text after 5000 iterations feels more "structured" than the RNN. It maintains punctuation and capitalization patterns more reliably over longer stretches.
- **Gradients**: I noticed that I didn't need to clip gradients as aggressively as I did with the RNN, though I kept it for stability.

**Sample Output Snippet:**
> "1:1 In the beginning was the Word, and the Word was with God, and the Word was God. 1:2 The same was in the beginning..."

While still not perfect, the "repetitive" nature of the Gospel of John was captured much better by the LSTM's memory cells.

## Reflection
Moving from vanilla RNNs to LSTMs feels like going from a notepad that gets erased every minute to a filing system. The implementation was significantly more complex—specifically the backpropagation through the gates (where you have to carefully track how gradients split and recombine). 

The mathematical beauty of the "additive" cell state is that it creates a highway for the gradient. Tomorrow, I look forward to exploring Sequence-to-Sequence models and Attention mechanisms, which take these sequence modeling concepts to the next level.
