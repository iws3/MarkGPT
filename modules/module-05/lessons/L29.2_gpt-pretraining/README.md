# GPT and Language Model Pretraining
## Comprehensive Learning Guide

## Transformer Architecture

Transformers use self-attention for context.

Parallel processing enables efficiency.

Positional encoding captures sequence order.

Multi-head attention models diverse relationships.

Feed-forward networks increase expressiveness.

Layer normalization stabilizes training.

Residual connections improve gradient flow.

## Language Model Pretraining

Next token prediction provides learning signal.

Causal attention prevents future information access.

Large text corpora provide training data.

Unsupervised learning requires no annotations.

Emergent capabilities appear with scale.

In-context learning enables few-shot performance.

Instruction tuning improves usability.

## GPT Capabilities

Text generation produces coherent sequences.

Few-shot learning adapts quickly.

Prompt engineering guides model behavior.

Chain-of-thought improves reasoning.

Instruction following enables task specification.

Knowledge stored implicitly in parameters.

Scaling laws predict performance with size.

## Advanced Language Models

Temperature and top-k sampling control generation diversity.

## Autoregressive vs Bidirectional

### Autoregressive (GPT)

Generate left-to-right
P(w_t | w_1, ..., w_{t-1})
Natural: Text generation
Can't see future
Used by GPT, GPT-2, GPT-3

### Bidirectional (BERT)

P(w_t | all words except w_t)
Can see both directions
Better for classification
Can't generate naturally
Used by BERT, RoBERTa

### Masked Language Model

[MASK] token in input
Predict what's masked
Bidirectional context
Allows both directions
BERT pre-training task

### Next Sentence Prediction

Predict if B follows A
Related sentences: Yes
Random sentences: No
Binary classification
BERT auxiliary task

