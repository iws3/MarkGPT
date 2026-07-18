# Day 18 Journal: Module 03 Review & Mini-Project 3

## Daily Setup
- **Date**: May 8, 2026
- **Module**: 03 - Neural Networks from Scratch
- **Topic**: Review & Mini-Project 3 (Character-level MLP)
- **Time Spent**: 

## Goals for Today
- [x] Review Module 03 concepts (Neurons, MLPs, Backprop, Loss/Optimization, Regularization)
- [x] Build a character-level language model using an MLP
- [x] Train the model on Psalm 23 in English
- [x] Train the model on Psalm 23 in Banso
- [x] Generate plausible continuations for both
- [x] Compare the character-level outputs between the two languages
- [x] Write reflection and evaluation

## Results (Mini-Project 3)

| Metric | English (KJV) | Banso (Lamnso') |
| :--- | :--- | :--- |
| Vocabulary Size | 24 characters | 24 characters |
| Training Samples | 572 | 480 |
| Final Loss (500 Epochs) | 0.1374 | 0.2049 |

### Sample Generations (T=0.7)
- **English**: `the lord for though i walk through the still waters he restoreth me be doin me beside the presence of the lord is my soul he...`
- **Banso**: `nyuiy boo moo mi gha la moo a menjiy ma bvuu fo shin nyam yi ndaa moo moo nkang wan boo moo wan gha la moo a mbaa nduu shi yi...`

## Reflection
### Key Observations:
1. **Vocabulary Consistency**: Interestingly, both the English and Banso versions of Psalm 23 used exactly 24 unique characters. This is likely due to the phonetic nature of the Banso translation provided, which uses a limited set of Latin characters.
2. **Convergence**: The English model reached a lower loss (0.1374) compared to the Banso model (0.2049). This might be due to slightly more training data in the English version (572 vs 480 samples) or more repetitive patterns in the KJV text that the MLP could latch onto more easily.
3. **Generation Quality**: For a character-level MLP with a context window of only 5, the results are surprisingly readable. It successfully learned common words like "lord", "shepherd", "nyuiy" (God), and "moo" (me). However, it occasionally struggles with long-range coherence, which is expected for an MLP compared to an RNN or Transformer.
4. **Banso Structure**: The Banso model successfully captured common Banso particles and repetitive structures like "a gha la moo" and "wan gha".
5. **Temperature**: At lower temperatures (0.7), the models were quite stable and followed the training text closely. At higher temperatures (1.1+), they started inventing new "biblical-sounding" but nonsensical words.

### Module 03 Summary:
Module 03 has been foundational. Moving from a single neuron to multi-layer perceptrons, and then implementing backpropagation and regularization from scratch, has demystified how neural networks actually "learn". Building this character-level model was a great way to cap off the module before moving into sequence modeling (RNNs/LSTMs) in Module 04.

![Loss Comparison](file:///c:/Users/the%20eye%20informatique/Desktop/ML/AI/MarkGPT/contributors/Fonyuy-pounds/module-03/exercises/day18_loss_comparison.png)


