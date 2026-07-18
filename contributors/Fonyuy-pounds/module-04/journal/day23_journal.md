# Day 23 Journal: Attention Mechanisms

## Daily Setup
- **Date**: May 13, 2026
- **Module**: 04 - Sequence Modeling
- **Topic**: Soft Attention, Dot-Product Attention
- **Time Spent**: 4.5 hours

## Goals for Today
- [x] Understand the problem with the Seq2Seq "Context Bottleneck".
- [x] Learn how Attention calculates alignment scores between Query and Keys.
- [x] Implement Dot-Product Soft Attention in PyTorch.
- [x] Integrate the Attention module into the Decoder.
- [x] Compare performance with the vanilla Seq2Seq from Day 22.

## Notes and Learnings

### Shattering the Context Bottleneck
Yesterday, I noticed that the Seq2Seq model struggled as sequence lengths grew. The poor final hidden state of the Encoder was being forced to memorize an entire sentence. 

**Attention** solves this beautifully. Instead of passing just the *final* hidden state, the Encoder now passes *all* of its hidden states (one for each timestep in the input). 

### How it Works (The PyTorch Way)
1. **Query**: The Decoder's current hidden state.
2. **Keys**: The Encoder's outputs.
3. **Values**: Also the Encoder's outputs.

The Decoder asks, "Given my current state (Query), which parts of the input sequence (Keys) are most relevant to predicting the next word?"
I used **Dot-Product Attention**:
$$ \text{Energy} = \text{Query} \cdot \text{Keys}^T $$
$$ \text{Attention Weights} (\alpha) = \text{Softmax}(\text{Energy}) $$
$$ \text{Context Vector} = \sum (\alpha \times \text{Values}) $$

This dynamically calculated context vector is then concatenated with the Decoder's input, giving it direct "line of sight" to the relevant input tokens.

## Exercises

### E23.1: Attention for Sequence Reversal
I upgraded the Sequence Reversal model from Day 22. 

**Observations:**
- **Faster Convergence**: The model with attention reached a low loss almost 30% faster than the vanilla Seq2Seq model.
- **Explainability**: By inspecting the `attention_weights`, I can actually *see* the model "looking" at the end of the input string when generating the beginning of the output string. If plotted, the attention weights form a perfect anti-diagonal line!
- **Longer Sequences**: The model no longer degrades as rapidly when pushing the `MAX_LEN` hyperparameter higher, because it doesn't have to compress everything.

## Reflection
Attention is an absolute game-changer. It makes intuitive sense—humans don't memorize an entire sentence before starting to translate it; we look back at specific words as needed. 
Implementing it in PyTorch wasn't too difficult thanks to `torch.bmm` (Batch Matrix Multiplication). I feel like this is the final stepping stone before hitting the true heavyweight architecture: **The Transformer**, which famously stated "Attention Is All You Need". I can't wait for Module 05!
