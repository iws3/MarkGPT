# Day 22 Journal: Sequence-to-Sequence (Seq2Seq) Architectures

## Daily Setup
- **Date**: May 12, 2026
- **Module**: 04 - Recurrent Networks & Sequence Modeling
- **Topic**: Seq2Seq and Encoder-Decoder Models
- **Time Spent**: 4 hours

## Goals for Today
- [x] Understand the Encoder-Decoder paradigm.
- [x] Learn how the "Context Vector" acts as an information bottleneck.
- [x] Transition from from-scratch NumPy code to PyTorch.
- [x] Implement a Seq2Seq model using PyTorch GRUs.
- [x] Train the model on a toy sequence reversal task.

## Notes and Learnings

### The Encoder-Decoder Pattern
The Seq2Seq model solves a critical problem: mapping an input sequence to an output sequence where the lengths might differ (e.g., translation, summarization). 
1. **Encoder**: Reads the input sequence item by item and compresses all the information into a final hidden state.
2. **Context Vector**: This final hidden state is handed over to the Decoder. It is expected to contain a summary of the *entire* input sequence.
3. **Decoder**: Takes the context vector as its initial hidden state and starts generating the output sequence step-by-step.

### The PyTorch Transition
After spending the last few days implementing BPTT through RNNs, LSTMs, and GRUs by hand using NumPy, I finally transitioned to using **PyTorch** today. 
The complexity of calculating gradients across two separate RNNs (Encoder and Decoder), managing the context vector gradient flow, and implementing "Teacher Forcing" made NumPy impractical. PyTorch's `autograd` feels like magic after doing the math manually.

### Teacher Forcing
A cool concept I learned today is **Teacher Forcing**. During training, the Decoder might generate a wrong character. If we feed that wrong character back in as the next input, the model spirals out of control. With Teacher Forcing, we randomly pass the *actual* ground truth character as the next input during training, which helps the model converge much faster.

## Exercises

### E22.1: Seq2Seq for Sequence Reversal
I built a Seq2Seq model using `nn.GRU`. Instead of translating English to Banso (which would require a large dataset and complex tokenization), I opted for a clean proof-of-concept: **Sequence Reversal**.

**The Task**: Feed a string like `"hello "`, and the model must output `" olleh"`.

**Observations**:
- The model successfully learned to compress a sequence into the context vector and unpack it in reverse order.
- The `Teacher Forcing Ratio` was crucial. At 0.5, the model learned steadily.
- The bottleneck became obvious at longer sequence lengths (e.g., `MAX_LEN > 15`). The single context vector struggles to hold all the character positional information. 

## Reflection
Moving to PyTorch was a massive relief, allowing me to focus on *architecture* rather than *calculus*. The Seq2Seq model is elegant, but the "context vector bottleneck" is a clear limitation. I can see why the curriculum introduces **Attention** next. If the decoder could "look back" at the encoder's intermediate states, it wouldn't have to rely entirely on that single context vector bottleneck. Looking forward to Day 23!
