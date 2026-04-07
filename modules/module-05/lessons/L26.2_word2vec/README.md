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

