# Text Classification and Document Representation
## Comprehensive Learning Guide

## Classification Approaches

Text classification assigns documents to categories.

Bag-of-words ignores word order.

Neural approaches learn representations.

CNN for text captures n-gram patterns.

RNN for text captures sequential information.

Attention mechanisms focus on important words.

Hierarchical models process documents in sections.

## Feature Representation

Lexical features capture word information.

Syntactic features encode sentence structure.

Semantic features represent meaning.

Discourse features capture document flow.

Character n-grams handle morphology.

Pre-trained embeddings transfer knowledge.

Feature engineering requires domain knowledge.

## Training Strategies

Balanced datasets ensure fair learning.

Class weighting handles imbalance.

Data augmentation increases training data.

Regularization prevents overfitting.

Validation monitors generalization.

Hyperparameter tuning optimizes performance.

Ensemble methods combine multiple models.

## Advanced Classification Methods

Zero-shot learning classifies without labeled examples.

Multi-label classification assigns multiple categories.

## Classification Architectures

### TextCNN

CNN on text sequences
1D convolution over words
Multiple filter sizes: 2, 3, 4
Max-over-time pooling
Simple, fast, effective

### FastText Classifier

Bag-of-words embeddings
Average word vectors
Hierarchical softmax
Extremely fast
Decent accuracy

### Attention-based

Attend to important words
Learn weights per word
Context-dependent importance
Interpretable decisions
Better performance

### Class Imbalance Handling

Reweight by class frequency
Oversampling minority
SMOTE: Synthetic examples
Adjust decision threshold
Multiple strategies

### Multi-label Classification

Multiple labels per document
"Action" and "adventure"
Not mutually exclusive
Different loss (cross-entropy per label)
Different evaluation metrics

## Classification Loss Functions

### Cross-Entropy

-Σ y_i * log(p_i)
Single label setting
Standard for classification
Differentiable
Numerically stable variants

### Focal Loss

Down-weight easy examples
PT = probability of true label
Loss = -alpha * (1-PT)^gamma * log(PT)
Focus on hard negatives
Helps imbalanced data

### Contrastive Loss

Triplet: anchor, positive, negative
Margin-based ranking
min(sim(a,p) - sim(a,n) + margin)
Forces semantic structure
Better embeddings

### Metrics for Classification

Accuracy: %correct
Precision: TP/(TP+FP)
Recall: TP/(TP+FN)
F1: Harmonic mean
ROC-AUC: Ranking metric

### Threshold Optimization

Default: 0.5
Move for cost-sensitive
Optimize F1 or custom metric
Search threshold
Data-dependent

