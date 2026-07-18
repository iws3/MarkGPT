# Day 24 Journal: Module 04 Review & Mini-Project 4
## Daily Setup
- **Date**: May 15, 2026
- **Module**: 04 - Sequence Modeling
- **Topic**: Mini-Project 4 (LSTM with Attention)
- **Time Spent**: 6 hours

## Goals for Today
- [x] Extract and preprocess the Gospel of Mark as a character-level dataset.
- [x] Build a complete Seq2Seq LSTM with Attention for language generation.
- [x] Train the model on Mark and generate 200-character passages.
- [x] Compare the quality of Attention-based generation with previous vanilla RNN models.

## Notes and Learnings

### Scaling Up to the Gospel of Mark
The Gospel of Mark provided a great middle-ground dataset—large enough to require real learning but small enough to train quickly on local hardware. After extracting it from the KJV Bible and running the character-level preprocessor, I had a vocabulary of about 70-80 unique characters (including punctuation and spaces).

### The Power of Attention in Generation
The biggest breakthrough today was seeing how the **Attention mechanism** (which I implemented yesterday for reversal) applies to open-ended text generation. 

By using the **Encoder** to process a "seed" verse and the **Decoder** to generate the continuation, the model no longer "forgets" the beginning of the sentence halfway through. In the Day 20 RNN (on John), the model would often drift into random character loops. Today's model maintains much better grammatical structure and even picks up on the repetitive Biblical cadence ("And he said...", "Then Jesus...").

### Implementation Details
- **Architecture**: LSTM Encoder -> Dot-Product Attention -> LSTM Decoder.
- **Teacher Forcing**: I used a 0.5 ratio, which helped the model stay on track during early training while still forcing it to learn its own generation paths.
- **Gradient Clipping**: Crucial for LSTMs to prevent the "exploding" gradients that can happen when sequences are long.

## Exercises

### Mini-Project 4: MarkLSTM
I implemented the model in `day24_project.py`. After 20 epochs, the loss dropped significantly, and the model started generating recognizable phrases.

**Sample Output:**
> **Seed**: "The beginning of the gospel of Jesus Christ, the Son of God; 1:2 As it is written in the prophets,"
> **Generated**: "Behold, I send my messenger before thy face, which shall prepare thy way before thee. The voice of one crying in the wilderness..."

## Reflection
Module 04 has been intense. Moving from simple hidden states to LSTMs and finally to Attention has shown me exactly how we reached the modern LLM era. The "bottleneck" problem is real, and Attention is the bridge that crossed it. I'm feeling confident and ready to tackle **Module 05** and deep-dive into advanced NLP foundations like BPE and Word Embeddings!
