# Word2Vec and Skip-Gram Models
## Comprehensive Learning Guide

## Skip-Gram Architecture

Word2Vec predicts context words from target.

Skip-gram uses target to predict surrounding words.

Continuous bag of words predicts target from context.

Single hidden layer keeps model simple.

Embedding matrix weights become word vectors.

Output layer predicts word probabilities.

Negative sampling approximates softmax.

## Training Mechanisms

Stochastic gradient descent optimizes embeddings.

Context window slides across text.

Random window size varies context diversity.

Subsampling removes frequent words.

Negative sampling speeds training.

Hierarchical softmax reduces computation.

Learning rate scheduling improves convergence.

## Word2Vec Properties

Captures semantic and syntactic relationships.

Analogy tasks reveal geometric structure.

Words and contexts have dual representations.

Frequency affects embedding quality.

Context size controls local vs global patterns.

Negative samples define what vectors avoid.

Simple architecture enables scalability.

## Word2Vec Extensions

Fasttext extends word2vec with character n-grams.

## Word2Vec Implementation Details

### Skip-gram Objective

Maximize: Σ log P(context|word)
Use Softmax or Negative Sampling
Gradient updates embeddings
Simple but effective
Parallelizable

### Negative Sampling Benefits

Avoids expensive softmax
K = 5-15 negative samples
Binary classification task
10-15x speedup typical
Practical necessity

### Context Window

Window size = 5: Look ±5 words
Larger window: More context
Smaller window: More local
Trade-off: Task-dependent
Typical: 5-10

### Multi-word Phrases

Identify phrases: "new york"
Treat as single token
Improves semantic quality
Statistic-based detection
Multiple passes

### Training Tips

Learning rate: 0.025 starting
Decay over time
Multiple epochs: 5-10
Larger corpus: Better
Parallelization on cores

## Skip-Gram Model Deep Dive

### Noise Contrastive Estimation

Simplify softmax calculation
Binary classification instead
Real: true word, Fake: random
log-sigmoid approximation
Efficiency trick

### Context Window Dynamics

Variable window: Helps
Weight by distance: Closer > farther
Dynamic sampling: More variety
Trade exploration vs stability
Empirically better

### Subword Patterns

Word2vec on subword tokens
"beautiful" = "beau" + "tiful"
Shares similarity structure
Helps morphologically similar
Factorizes meaning

### Sampling Strategies

Uniform negative sampling
High-frequency negatives: More common
Unigram^0.75: Balance
Affects convergence
Task dependent

### Initialization Impact

Random small values
Xavier initialization
Affects convergence speed
Final quality similar
Early iterations differ

