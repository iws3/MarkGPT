# ELMo and Contextualized Word Representations
## Comprehensive Learning Guide

## Contextual Embeddings

ELMo generates context-dependent word representations.

BiLSTM processes text bidirectionally.

Word meaning varies with context.

Fixed embeddings cannot capture variation.

Character-level inputs enable morphological transfer.

Multiple layers capture different linguistic properties.

Layer combination weights determine representation.

## Training Mechanism

Language modeling predicts next token.

Bidirectional prediction provides full context.

Shared weights increase efficiency.

Large-scale pretraining enables good representations.

Transfer learning applies learned representations.

Fine-tuning adapts to specific tasks.

Representation quality improves downstream performance.

## Applications and Extensions

NLP tasks benefit from contextual representations.

Multilingual models handle code-switching.

Domain-specific models improve specialized tasks.

Lightweight variants reduce memory footprint.

Real-time systems need fast inference.

Combination with static embeddings improves robustness.

Integration with downstream models straightforward.

## Contextualized Representation Extensions

BERT uses masked language modeling for deeper context.

## ELMo Deep Dive

### Training Data

1B token corpus
Wikipedia + news crawl
Diverse language
Large-scale pre-training
Weeks to train

### Bidirectional Processing

Forward LSTM: Left-to-right
Backward LSTM: Right-to-left
Concatenate: Both directions
Context from both sides
Better than forward only

### Layer Combination

Learned weights per layer
γ * (w_input*input + w_fwd*fwd + w_bwd*bwd)
Task-specific combination
Different tasks use different layers
Adaptive representation

### Evaluation Results

NER: +1-2% F1
SQuAD: +2-3% F1
Sentiment: +1% accuracy
Consistent but modest
Foundation, not solution

### Computational Cost

Large model: 93.6M params
Slow inference: 100ms per sentence
Not practical for real-time
Became BERT's problem
BERT 12x more expensive!

## Contextualized Embeddings Theory

### Why Contextualization?

Same word, different contexts
"bank" financial vs river
Fixed embeddings lose sense
Context -> sense-specific
Dynamic representation

### Shallow vs Deep Contextualization

2-layer LSTM: ELMo
12-layer Transformer: BERT
More layers: Better
Depth enables abstraction
Computational cost

### Task-Specific Weighting

Different tasks use different layers
NER: Maybe layer 1
Sentiment: Maybe layer 2
Learned weights per layer
Adaptive representation

### Computational Efficiency

ELMo: Expensive forward pass
Cache embeddings: Pre-compute
BERT: Also slow
Quantization helps
Distillation: Compression

