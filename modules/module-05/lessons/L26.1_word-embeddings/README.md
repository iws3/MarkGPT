# Word Embeddings and Representation Learning
## Comprehensive Learning Guide

## Embedding Fundamentals

Word embeddings map words to dense vectors.

Vector dimensions capture semantic properties.

Similar words have similar embeddings.

Embedding space enables mathematical operations.

Analogies solvable through vector arithmetic.

Dimensionality reduction preserves relationships.

Distributed representations improve generalization.

## Embedding Properties

Semantically similar words cluster in space.

Related concepts form coherent regions.

Direction captures semantic relationships.

Magnitude affects similarity metrics.

Vector operations reflect linguistic properties.

Additive compositionality enables phrase vectors.

Geometry of embedding space interpretable.

## Learning Embeddings

Embeddings learned from distributional context.

Surrounding words provide training signal.

Prediction tasks drive embedding learning.

Frequency weighting emphasizes common words.

Context window determines learned relationships.

Training objective shapes embedding properties.

Initialization affects convergence speed.

## Advanced Embedding Techniques

Retrofitting embeddings to external knowledge bases.

Cross-lingual embeddings enable multilingual transfer.

## Embedding Properties

### Compositionality

Can combine embeddings?
"new" + "york" ≈ "new york"
Sometimes works, sometimes not
Non-linear: Addition too simple
Better: Learn composition

### Polysemy (Multiple Meanings)

"bank" = financial or river
Single embedding loses info
Contextualized embeddings help
Or: Prototype + sense vectors
Hard problem

### Hypernym-Hyponym Relations

"dog" is hyponym of "animal"
Hierarchical semantic structure
Embeddings reflect hierarchy
Vector direction matters
Emergent property

### Cultural Bias

"doctor" more similar to "he"
"nurse" more similar to "she"
Reflects training data bias
Problematic for applications
Debias methods exist

### Dimensionality Sweet Spot

50D: Too small, poor quality
100D: Decent for small tasks
300D: Standard, good balance
1000D: Overkill for most
Find empirically

## Embedding Quality Evaluation

### Intrinsic vs Extrinsic

Intrinsic: Standalone tests
Word similarity datasets
Fast to compute
Extrinsic: Downstream tasks
Real application performance

### Analogy Evaluation

"king" - "man" + "woman" = "queen"
Semantic + syntactic
Google word test
Often not reliable
Debate on validity

### Nearest Neighbor Analysis

Find k-nearest neighbors
Qualitative inspection
Should be semantically similar
Reveals embedding quality
Most interpretable

### Alignment to Human Judgments

Human rate word pairs
1-10 similarity scale
RareWord-353, SimLex
Spearman correlation
Standard benchmarks

### Speed vs Quality Tradeoff

Large embedding: Better quality
Fast to compute
Large embedding: Slow inference
Need:Space tradeoff
Application dependent

