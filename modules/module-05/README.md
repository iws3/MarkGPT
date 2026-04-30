# Module 05 — NLP Foundations: Text as Data
## Days 25–30 | Intermediate

---

## Module Overview

How do machines see language? This module teaches the practical skills for converting text into numbers: tokenization, embeddings, and the craft of feature engineering. You'll also learn about pre-Transformer language models that ruled the era 2018–2019.

By the end of Module 05, you will:
- Implement tokenization from Byte Pair Encoding
- Train word embeddings and visualize them
- Build a text classifier
- Understand ELMo and the shift toward contextual representations

## Learning Objectives

- Understand core ML concepts
- Implement algorithms from scratch
- Relate theory to MarkGPT architecture
- Complete hands-on exercises

## Structure

```
lessons/       - Conceptual explanations with code examples
exercises/     - Practical implementation exercises
projects/      - Larger projects (optional)
resources/     - Additional readings and links
```

## Time Estimate

- Lessons: 4-6 hours
- Exercises: 4-6 hours
- **Total: 8-12 hours per module**

## Key Concepts

[See lesson files for detailed content]

## Completion Checklist

- [ ] Read all lessons (L*_*.md files)
- [ ] Complete all exercises (day*_*.md files)
- [ ] Pass the module quiz (if provided)
- [ ] Understand connections to MarkGPT

## Resources

- Lesson references contain links to papers and tutorials
- http://markgpt-docs.com (forthcoming)
- GitHub discussions: https://github.com/yourusername/MarkGPT-LLM-Curriculum/discussions

## Next Module

See ../module-0$((i+1))/README.md for the next module.
## Tokenization Fundamentals

### What is Tokenization?

Converting text → tokens (numbers)
Token: Smallest unit (word, subword, character)
Required step for all NLP models
Quality crucial: Affects downstream tasks
Trade-off: Granularity vs vocabulary size

### Character-level Tokenization

Simplest: Each character = token
Alphabet size: 26 + digits + punct ≈ 100
Pros: Handles any text (misspellings, OOV)
Cons: Sequences very long, harder to learn
Example: "Hello" → [H,e,l,l,o]
Used in: Character-level language models

### Word-level Tokenization

Split on whitespace and punctuation
Vocab size: 10K-100K typical
Pros: Interpretable, reasonable length
Cons: OOV problem (unknown words)
Example: "Hello world!" → ["Hello", "world", "!"]
Problem: "hello" vs "Hello" = different tokens

### Subword Tokenization

Middle ground: Parts of words
Vocab size: 1K-100K
Examples: Byte-Pair Encoding, WordPiece
Balances word and character levels
Standard in modern NLP
"Hello" → ["He", "llo"] or ["Hel", "lo"]

## Byte-Pair Encoding (BPE)

### Algorithm

1. Start with characters + special symbols
2. Count all adjacent pairs
3. Merge most frequent pair
4. Repeat until vocab size reached
Simple greedy algorithm
Very effective in practice

### BPE Example

Text: "hello hello"
Initial: [h,e,l,l,o, ,h,e,l,l,o]
Step 1: "l" "l" frequent → [h,e,ll,o, ,h,e,ll,o]
Step 2: "h" "e" frequent → [he,ll,o, ,he,ll,o]
Step 3: "he" "ll" frequent → [hell,o, ,hell,o]
Result: [hell, o, </s>, hell, o]

### BPE Advantages

Handles misspellings: "helo" → ["he", "lo"]
Compression: Frequent words = single token
Vocabulary: Finite size (predictable memory)
Language independent: Works on any language
Reversible: Can decode back
Reproducible: Same text → same tokens

## WordPiece Tokenization

### Differences from BPE

Merge criterion: Likelihood maximization
Not just frequency
Used in: BERT, RoBERTa
Similar results to BPE
Slightly different algorithm
Both work well in practice

## SentencePiece

### Language Agnostic

Works on raw text (no preprocessing)
No language-specific logic
Combines BPE and unigram language model
Treats space as token
Great for non-Latin scripts
Used in: T5, mBERT, many recent models

## Vocabulary Size Impact

### Small Vocab (1K)

Sequence length: Very long
Memory per sample: High
Training time: Slow
Parameter count: Lower (embedding matrix)
Typical: Character-level models

### Large Vocab (100K)

Sequence length: Short
Memory per sample: Low
Training time: Fast
Parameter count: Very high
Typical: BERT, GPT
Trade-off: Memory vs speed

## Special Tokens

### Standard Special Tokens

[CLS]: Classification token (start)
[SEP]: Separator (between sentences)
[PAD]: Padding (fill short sequences)
[UNK]: Unknown (OOV words)
[MASK]: Masked token (BERT pre-training)
</s>: End of sequence
<s>: Start of sequence

### Custom Tokens

Task-specific: [QUESTION], [ANSWER]
Entity types: [PER], [LOC], [ORG]
Domain-specific: [CODE], [EQUATION]
Improves performance
Requires fine-tuning
Common in production systems

## Handling OOV Words

### Problem

Word not in vocabulary → [UNK]
Loses information
Subword tokenization helps
"unrecognizable" → ["unrecognizable"]
BPE: ["unrecogniz", "able"]
Preserves information!

### Solutions

1. Subword tokenization (BPE, WordPiece)
2. Character-level fallback
3. Morphological analysis
4. Expand vocabulary
5. Back-off smoothing (pre-training trick)
Best: Combine approaches

## Tokenization Quality Metrics

### Compression Ratio

Average tokens per word
1.0: Perfect (1 token per word)
1.3: Good (3 tokens per 10 words)
2.0: Poor (half as many words)
Impact: Memory and compute
Typical: 1.1-1.3 for English

### Vocabulary Coverage

% of corpus tokens that are in-vocabulary
BERT (30K vocab): 98%+
GPT-2 (50K vocab): 99%+
Smaller vocab: Lower coverage
Affects performance
Trade-off: Size vs coverage

## Contextual Tokenization

### Problem: Ambiguity

"bank" = financial vs river bank
Single tokenization misses context
Morphologically: Same
Solution: Same token, different embeddings
Transformers learn contextual meaning

## Tokenization in Code

### Using HuggingFace

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base')
tokens = tokenizer.encode('Hello world')
# [101, 7592, 2088, 102]
text = tokenizer.decode(tokens)
# 'Hello world'
```

### Custom Tokenizers

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())
# Train on your data
tokenizer.train_from_iterator(texts, vocab_size=50000)
# Use it
encoding = tokenizer.encode('Your text')
```

## Multilingual Tokenization

### Challenges

100+ languages, different scripts
Chinese: Words not separated
Arabic: Right-to-left
Agglutinative: Turkish, Finnish
Solution: Universal tokenizers
SentencePiece: Language-agnostic

### mBERT Tokenization

Shared tokenizer across 104 languages
110K vocabulary
WordPiece trained on multilingual corpus
Works reasonably well
Trade-off: Per-language quality
Enables zero-shot cross-lingual transfer

## Tokenization Speed

### Practical Performance

BERT tokenizer: 10K tokens/s
SentencePiece: 100K tokens/s
Character-level: 1M tokens/s
Bottleneck: Usually data loading
Cache tokens during preprocessing
Pre-tokenize for speed

## Preprocessing Pipeline

### Best Practices

1. Normalize: Lowercasing (task-dependent)
2. Remove: Accents, special chars (careful!)
3. Tokenize: Use standard tokenizer
4. Truncate: Limit length
5. Pad: Make same length for batching
6. Convert to IDs: Vocabulary lookup

## Tokenization Errors

### Common Issues

1. **Case sensitivity**: "Hello" ≠ "hello"
   Solution: Lowercase before tokenizing

2. **Punctuation**: Attached vs separate
   Solution: Check tokenizer behavior

3. **Contractions**: "don't" → [
   Solution: Expand or handle in tokenizer

4. **Whitespace**: Multiple spaces
   Solution: Normalize whitespace

## Vocabulary Learning

### From Unlabeled Data

Train on large corpus
No labels needed
Learn frequent patterns
Adapt to domain
Example: Train on Wikipedia → Reddit
Learn Reddit-specific slang

## Word Embeddings Fundamentals

### What are Word Embeddings?

Dense vectors representing words
Dimension: 50-300 typical
Learned from large text corpus
Similar words → similar vectors
Foundation of modern NLP
Input to neural networks

### Distributional Hypothesis

"You shall know a word by the company it keeps"
Context determines meaning
Words in similar contexts → similar meanings
Learning principle: Co-occurrence statistics
Basis for all embedding methods
Remarkably effective!

### Embedding Dimension

50D: Very small, fast, limited expressiveness
100D: Minimal, basic tasks
300D: Standard for word embeddings
1000D: Large, rich, slow
Larger: More expressive, more parameters
Typical: 300D word2vec, 768D BERT

## Word2Vec

### Skip-gram Model

Predict context from word
Input: Center word
Output: Surrounding words (window)
Loss: Cross-entropy
Objective: Maximize P(context|word)
Simple but powerful

### CBOW (Continuous Bag of Words)

Opposite of Skip-gram
Input: Context words
Output: Center word
Faster to train
Better for frequent words
Generally worse performance than Skip-gram

### Negative Sampling

Problem: Softmax over entire vocabulary
Huge vocabulary: 1M+ words
Computing softmax: O(V)
Solution: Negative sampling
Sample K negative examples
Loss: Binary classification (positive vs negatives)
10-15x speedup!

### Hierarchical Softmax

Alternative to negative sampling
Binary tree of vocabulary
Log depth ≈ log(V)
Each path: Binary decisions
Faster than full softmax
Slower than negative sampling
Used in some implementations

### Word2Vec Training

1. Initialize embeddings randomly
2. Iterate through corpus
3. For each word-context pair:
   - Compute output probability
   - Compute loss
   - Update embeddings
4. Repeat multiple epochs
Convergence: ~10-20 billion words

### Analogies: Additive Property

"king - man + woman ≈ queen"
Vector arithmetic works!
v(king) - v(man) + v(woman) ≈ v(queen)
Shows semantic structure
Not always perfect
Remarkable emergent property

### Why it Works

Skip-gram objective: Embed words
Minimize distance for similar contexts
Words in similar positions → close vectors
Unsupervised learning from corpus statistics
No manual annotations needed
Scales to huge corpora

## GloVe (Global Vectors)

### Motivation

Word2Vec: Local context only
GloVe: Combine local and global
Use word co-occurrence matrix
Factor matrix → embeddings
Linear transformation
Faster training than Word2Vec

### Co-occurrence Matrix

Count how often words co-occur
X_ij = count(word_i near word_j)
Huge and sparse: V x V
100K vocab: 10 billion entries!
Solution: Only store non-zeros
Factorize: X ≈ W @ W^T

### GloVe Objective

Weighted least squares:
L = Σ_ij f(X_ij) (w_i . w_j - log X_ij)^2
f(X_ij): Weight function
Prevents rare co-occurrences dominating
Closed form optimization possible
Faster convergence than Word2Vec

### GloVe vs Word2Vec

GloVe: Explicit global statistics
Word2Vec: Implicit via optimization
GloVe: Typically better (slightly)
Word2Vec: Simpler, faster
Both: Great in practice
Modern: Transformers replace both

## fastText

### Subword Vectors

Problem: Word2Vec can't handle OOV
Solution: Learn character n-gram embeddings
Word = sum of character n-grams
v(hello) = v(he) + v(el) + v(ll) + v(lo) + ...
Handles unseen words!
Handles morphology

### Character N-grams

Typical: 3-6 character n-grams
"hello" (n=3): [hel, ell, llo]
"hello" (n=4): [hell, ello]
"hello" (n=5): [hello]
Plus special markers for word start/end
Sum: Gives word embedding

### fastText Training

1. Compute character n-gram vectors
2. Word vector = sum of n-grams
3. Train like Word2Vec (Skip-gram)
4. Learn both: word vectors and n-grams
5. At inference: Can compute for OOV

Handling spelling: Similar n-grams

### fastText Language Support

Multilingual: 176+ languages
Pre-trained vectors available
Works great for morphologically rich
Turkish, Finnish, Czech, etc.
Better than Word2Vec for these
Practical advantage for low-resource

## Embedding Visualization

### t-SNE Projection

300D embedding → 2D visualization
Non-linear dimensional reduction
Preserves local structure
Shows word clusters
Similar words close together
Beautiful emergent structure

### UMAP

Uniform Manifold Approximation
Faster than t-SNE
Preserves global structure better
Two parameters: n_neighbors, min_dist
Good for large embedding sets
Can scale to millions of vectors

## Training Embeddings

### Using gensim

```python
from gensim.models import Word2Vec

model = Word2Vec(sentences, size=300, window=5)
vec = model.wv['hello']
similar = model.wv.most_similar('king')
```

### Using fastText

```python
import fasttext
model = fasttext.train_unsupervised('data.txt')
vec = model.get_word_vector('hello')
```

## Pre-trained Embeddings

### When to Use Pre-trained

Limited data: Always use
Data on same domain: Cold-start faster
Transfer learning: Few examples needed
Stability: Better than random init
Typical: Download from model zoos
Fine-tune on task data

### Domain Specificity

Generic (Wikipedia): Good baseline
Medical (PubMed): Better for medical
Code (GitHub): Better for code
Domain adaptation: Fine-tune
Task-specific: Train from scratch if big data
Usually: Domain > task-specific

## Embedding Evaluation

### Intrinsic Evaluation

Word analogies: King - man + woman = queen
Similarity correlation: Compare to human ratings
Relatedness tasks: RareWord, SimLex
Direct measure of embedding quality
Fast to compute
Not always predictive of downstream

### Extrinsic Evaluation

Downstream task performance
Text classification
Named entity recognition
Sentiment analysis
Best predictor of real usefulness
But: Slower and task-dependent

## Embedding Training Tips

### Hyperparameters

Size (dimension): 100-300 typical
Window size: 2-15 (larger = preserves syntax)
Negative samples: 5-15
Epochs: 5-30 (more if larger corpus)
Learning rate: Usually 0.025 default
Minimum count: 5, drop rare words

### Dataset Size

Small (1M words): Overfitting risk
Medium (100M words): Good quality
Large (1B+ words): Best quality
Diminishing returns after 1B
Typical: Train on crawled web text
Wikipedia (3.5B words): Excellent base

### Computational Efficiency

Word2Vec: O(V) with negative sampling
Can train on single machine
100M words: Minutes
1B words: Hours
Parallel training: Multiple threads
Fast compared to deep learning

## Contextualized Embeddings

### Static vs Dynamic

Static (Word2Vec): Same vector for all contexts
Dynamic (ELMo, BERT): Different per context
Problem with static: "bank" is ambiguous
ELMo: First dynamic method (2018)
Revolution in NLP
Transition to transformers

## Common Mistakes

### Don't

- Train on tiny corpus (<100K words)
- Use embedding size > 500 without need
- Ignore OOV problem
- Fine-tune randomly (use pre-trained)
- Ignore domain mismatch
- Train embeddings from scratch if data limited

### Do

- Use pre-trained when possible
- Match embedding dim to task
- Fine-tune on task data
- Evaluate on downstream tasks
- Consider subword (fastText) for OOV
- Use contextualized for modern tasks

## Embedding Arithmetic

### Word Analogies

Task: a is to b as c is to d
Find d: d ≈ b - a + c
Example: Paris - France + Germany = Berlin
Remarkable: Works surprisingly well
Emergent property of vector space
Not always accurate but conceptually rich

### Semantic Relationships

Synonyms: Similar vectors
Antonyms: Opposite vectors
Hypernyms: Generalization direction
Part-of: Also somewhat reflected
Relational properties: Encoded implicitly
Remarkable unsupervised learning!

## ELMo: Embeddings from Language Models

### The Great Insight (2018)

Language modeling captures context
Hidden states encode rich information
Use as embeddings instead of words
Different representation per context
Solves word sense disambiguation
Revolutionized NLP in 2018

### Architecture

Bidirectional LSTM language model
Forward LSTM: Left to right
Backward LSTM: Right to left
3 layers: Input + 2 hidden
Each token: Multiple representation options
Combine all: Weighted sum of layers

### Training

Language modeling task:
Predict next token given context
Unsupervised: Only need text
Scale: 1B token corpus
Result: Rich contextual representations
Weights: Learned task-specifically

### Usage

1. Pre-train LSTM language model
2. Extract hidden states for token
3. Concatenate/weight combinations
4. Use as features for downstream
5. Fine-tune weights on task
Simple layer-weighting learned

### Results

NER: +2% F1 on CoNLL
Sentiment: +1-2% accuracy
QA: +2-3% on SQuAD
Consistent improvements
Modest but reliable gains
Foundation for BERT

### Limitations

Unidirectional for future context
Backward pass separate
Slow to compute
Large model size
Replaced by BERT (bidirectional)
But: Key insight remains (contextual)

## Pre-Transformer Context

### Timeline

2013: Word2Vec
2014: GloVe
2016: fastText
2018: ELMo (game changer)
2018: BERT (even better)
2018: GPT (autoregressive)
2019: Beyond

## Feature Engineering

### Traditional NLP Pipeline

1. Tokenization
2. POS tagging
3. Parsing
4. Named entities
5. Manual feature extraction
6. Machine learning classifier
Was state-of-art pre-deep learning

### Feature Types

Lexical: Word, lemma, stem
Morphological: POS tag, suffixes
Syntactic: Dependency relations
Semantic: Entity types, meanings
Contextual: Surrounding words
External: Knowledge bases

### TF-IDF

Term Frequency-Inverse Document Freq
TF(t, d) = count(t in d) / |d|
IDF(t) = log(N / count(docs with t))
TF-IDF(t, d) = TF(t, d) * IDF(t)
Downweights common words
Simple but effective baseline

### Bag of Words

Simplest representation
Count each word occurrence
Create feature vector
Loses word order
High-dimensional but sparse
Still baseline for many tasks

### N-grams

Unigrams: Individual words
Bigrams: Two consecutive words
Trigrams: Three consecutive
Encodes local word order
Partially solves word order loss
Increase dimensionality

## Text Classification

### Problem Definition

Input: Document (text)
Output: Category label (discrete)
Task: Learn function mapping
Many real applications
Sentiment, spam, topic, intent
Foundation for NLP

### Simple Baseline

1. TF-IDF vectorization
2. Logistic regression
3. Get predictions
4. Evaluate on test set
Often surprisingly good
Fast, interpretable, reliable

### Deep Learning Approach

1. Tokenize text
2. Lookup embeddings
3. Pool/RNN over sequence
4. Dense layers
5. Softmax for class probs
Better with large labeled data

### Datasets

20 newsgroups: 20 categories
Movie reviews: Sentiment (2 class)
AG News: News categorization
DBpedia: Wikipedia categories
TREC: Question type classification
Standard benchmarks for evaluation

### Evaluation Metrics

Accuracy: % correct (balanced data)
Precision: TP / (TP + FP)
Recall: TP / (TP + FN)
F1: 2 * (P*R) / (P+R) (unbalanced)
Macro F1: Average per class
Choose based on use case

## Information Extraction

### Named Entity Recognition

Identify entities: People, places, orgs
Tag each token with entity type
Sequence labeling task
BIO tagging scheme
B: Begin, I: Inside, O: Outside
Fundamental IE task

### Relation Extraction

Identify relationships between entities
Example: Company_X founded_in Year_Y
Given: Named entities
Find: Relationship type
Applications: Knowledge base construction
Harder than NER

### Dependency Parsing

Identify grammatical structure
Who depends on whom
Arc labels: Subject, object, etc.
Tree structure (mostly)
Transition-based systems
Graph-based systems

## Sequence Labeling

### Problem Setup

Input: Sequence of tokens
Output: Label per token
Constraints: Local dependencies
Examples: POS, NER, chunking
More complex than classification
Order and structure matter

### CRF (Conditional Random Fields)

Probabilistic sequence model
Conditions on observations
Globally normalized
Handles dependencies
Can't predict impossible sequences
Good baseline for labeling

### BiLSTM-CRF

BiLSTM: Encode sequence
CRF: Decode with constraints
State-of-art pre-Transformer
High performance
Good for structured prediction
Popular in industry

## Language Models (Pre-Transformer)

### N-gram Language Models

P(w_t | w_{t-n+1}, ..., w_{t-1})
Count-based: Simple and effective
Backoff: Handle OOV
Smoothing: Unseen n-grams
Fast inference
Limited by sparsity

### RNN Language Models

Recurrent: Process token by token
Hidden state: Encodes context
Unbounded context
Much better than n-grams
Training: Backprop through time
Slow but powerful

### Perplexity

How surprised model is
PP = 2^(cross-entropy)
Lower is better
Baseline: Unigram LM (~1000)
LSTM: ~100-200
Good metric for LM quality

## Text Similarity

### Cosine Similarity

For embeddings or TF-IDF vectors
cos(u, v) = (u . v) / (||u|| ||v||)
Range: [-1, 1], typically [0, 1]
1: Identical direction
0: Orthogonal
-1: Opposite

### Word Movers Distance

Treat documents as distributions
Distance between word clouds
Uses word embeddings
More semantic than BoW
Computationally expensive
Good for short documents

### Semantic Similarity Tasks

STS Benchmark: Sentence pairs
Rate 0-5 similarity
450K+ sentence pairs
Standard evaluation benchmark
Predicts perceptual similarity
Challenging: Requires understanding

## Paraphrase Detection

### Problem

Identify if sentences mean same
Binary classification
Different words, same meaning
Or: Similarity rating
Useful for plagiarism detection
Helps question answering

### Approaches

1. Edit distance: Too simple
2. TF-IDF similarity: Better
3. Word embeddings: Much better
4. ELMo/BERT: State-of-art
5. Fine-tuned transformer: Best
Modern: >> 95% accuracy

## Machine Translation (Pre-Seq2seq)

### Statistical Machine Translation

P(t|s) = P(t) * P(s|t) / P(s)
Language model: P(t)
Translation model: P(s|t)
Phrase-based capturing
Reordering models
Dominated until 2016

### Seq2Seq Breakthrough

Encoder-decoder RNNs
End-to-end learning
Outperformed SMT dramatically
BLEU score: 35+ vs SMT 25+
Simpler system
Single model, not pipeline

## Sentiment Analysis

### Task Definition

Classify positive vs negative
Or: Rate on scale 1-5
Common application
Good benchmark problem
Many datasets available
Progression: BoW → embeddings → transformers

### Challenges

Sarcasm: "This is great!" (negative)
Negation: "not bad" = positive
Aspect sentiment: "Good food, bad service"
Multi-class: Fine-grained opinions
Domain: Language varies by domain
More complex than it seems

### Datasets

Movie reviews: Binary, straightforward
IMDB: 25K training examples
SemEval: Multiple languages
Amazon reviews: Large scale
Twitter: Informal, sarcasm
Different difficulty levels

## Vector Space Models

### Distributional Similarity

Words as points in space
Distance encodes similarity
Clusters emerge automatically
Unsupervised discovery
Remarkable regularities
Foundation of modern NLP

### Locality Sensitive Hashing

Fast nearest neighbor search
Hash similar vectors together
Approximate but efficient
Million vector queries: Milliseconds
Used in production systems
Scales to web

## Knowledge Distillation Pre-Transformers

### Compressing Models

Large model: Better performance
Small model: Faster inference
Distillation: Transfer knowledge
Student learns from teacher
Soft targets via temperature
Pre-dates transformers

## Attention Before Transformers

### Attention for Seq2Seq

Bottleneck: Single vector from encoder
Attention: Look at all encoder states
Different attention per decoder step
Context vector: Weighted sum
Huge improvement for translation
Next: Make entire model attention

## Embedding Analysis

### Bias in Embeddings

man:programmer ≈ woman:homemaker
Reflects training data bias
Problematic for applications
Detection: Analogy tests
Mitigation: Debias embeddings
Ongoing research

### Word Sense Disambiguation

"Bank" = financial vs river
Single embedding fails
Solution: Contextualized (ELMo+)
Different vectors per context
Better semantic understanding
Jumping off point to BERT

## Cross-lingual Transfer

### Multilingual Embeddings

Single space across languages
Similar concepts align
Zero-shot translation
mBERT enables this
Word2Vec: Language-specific
BERT: Unified space

## Module 05 Summary

**Concepts Learned**
- Tokenization methods and trade-offs
- Word embeddings (Word2Vec, GloVe, fastText)
- Static vs contextualized embeddings
- ELMo and pre-Transformer models
- Classical NLP features (TF-IDF, n-grams)
- Text classification and sequence labeling
- Information extraction
- Language modeling
- Machine translation and seq2seq
- Production systems

## Advanced Text Representations

### Contextual vs Non-contextual

Non-contextual: "Hello" always same vec
Contextual: "Hello" different per context
ELMo: First contextual (2018)
BERT: Bidirectional contextual
GPT: Autoregressive contextual
Modern requirement

### Layer-wise Analysis

Lower layers: Syntax (POS, chunking)
Middle layers: Low-level semantics
Upper layers: High-level (task-specific)
Transfer learning exploits this
Fine-tune upper layers for task
Freeze lower for general knowledge

## Dimensionality Reduction

### PCA (Principal Component Analysis)

Linear projection to k dimensions
Preserves variance maximum
Fast: SVD computation
Interpretable: Components are words
Works okay for embeddings
Ok visualization

### t-SNE Deep Dive

Non-linear dimension reduction
Preserves local structure
Clustered visualization
Slow for large datasets
Parameter sensitive
Best for exploration

### UMAP Advantages

Preserves both local and global
Much faster than t-SNE
Scales to millions
Theoretically principled
Better for production
Modern choice

## Linguistic Structure

### Syntax vs Semantics

Syntax: Grammar, word order
Semantics: Meaning
Embeddings capture both
Lower layers: More syntactic
Attention patterns reveal structure
Can probe for structure

### Probing Tasks

Train classifier on hidden states
Predict POS tags from embeddings
If high accuracy: Encodes POS
Multiple properties simultaneously
Trade-off: Information vs capacity
Reveals learned representations

## Morphology

### Morphological Analysis

Words have structure
Stems and affixes
"Running" = "run" + "-ing"
Embeddings somewhat capture
More explicit: Morphological models
Helps low-resource languages

### Morphologically Rich Languages

Turkish: Agglutinative (many suffixes)
Finnish: Very complex
Czech: Many cases
Arabic: Root patterns
Challenges: Many word forms
fastText helps via subword

## Domain Adaptation

### The Problem

Train on Wikipedia (general)
Test on biomedical (specific)
Distribution shift
Generic embeddings miss domain
Solution: Adapt embeddings
Or train model on domain

### Approaches

1. Retrain on domain corpus
2. Fine-tune general embeddings
3. Domain-specific vocabulary
4. Combine with domain knowledge
5. Active learning: Query hard examples
Typically: Fine-tune + domain vocab

## Data Augmentation

### Backtranslation

Translate to another language
Translate back to original
Creates paraphrases
Double data size
Improves robustness
Common for small datasets

### Synonym Replacement

Replace words with synonyms
EDA: Easy Data Augmentation
Random and controlled
Preserves label but varies text
Simple baseline
Works surprisingly well

### Mixup and Cutoff

Mixup: Linear combination of embeddings
Cutoff: Drop random tokens/subwords
Regularization via augmentation
Helps with small datasets
Modern practice

## Low-Resource NLP

### Transfer Learning

Pre-train on general (billions tokens)
Fine-tune on task (thousands examples)
Much better than training from scratch
50-story/0-shot with good model
Modern necessity
Enables low-resource work

### Few-Shot Learning

Learn from very few examples
5-10 labeled examples
Meta-learning approaches
MAML: Model-agnostic meta-learning
Prototypical networks
Enables rapid adaptation

### Zero-Shot Learning

No labeled examples
Use class descriptions
Match to class semantic vectors
Requires good embeddings
Prompt-based approach
Emerging with large models

## Handling Imbalanced Data

### Class Imbalance

90% negative, 10% positive
Accuracy misleading
Always predict majority
Accuracy: 90% but useless!
Need: Balanced metrics

### Solutions

1. Oversampling: Duplicate minority
2. Undersampling: Drop majority
3. Cost-sensitive: Penalize errors
4. SMOTE: Synthetic minority examples
5. Threshold tuning: Adjust decision
6. Use better metrics: F1, RoC-AUC

## Error Analysis

### Systematic Analysis

1. Look at wrong predictions
2. Categorize error types
3. Count by category
4. Focus on top errors
5. Address root causes
Better than tuning:
Understanding > parameters

### Confusion Matrix

TP: True Positive
FN: False Negative (missed)
FP: False Positive (false alarm)
TN: True Negative
Reveals what model struggles with
Guides data collection

## Active Learning

### Selective Labeling

Don't label everything
Label most informative examples
Model-uncertain examples
Reduces annotation cost
20% labels → 90% of full accuracy
Smart data collection

### Query Strategies

Uncertainty: Model least confident
Diversity: Examples unlike labeled
Expected gradient length: Big gradient
Committee: Ensemble disagreement
Typical: Start with diverse, refine uncertain

## Interpretability

### Feature Attribution

Which words matter for prediction?
Gradient-based: ∂L/∂x
Attention: Which words attended?
LIME: Local explanations
SHAP: Game theory-based
Understanding models is crucial

### Adversarial Robustness

Small input changes flip prediction
Spelling errors: "resteraunt"
Paraphrases: Change wording
Typos: Should be robust
Adversarial training helps
Data augmentation helps

## Production Deployment

### Model Serving

Model trained - now what?
API endpoint for predictions
Batch processing for bulk
Real-time requirements
Infrastructure: Docker, Kubernetes
Monitoring: Accuracy, latency

### Edge Cases

Empty input: Handle gracefully
Very long text: Truncate intelligently
OOV words: Use subword fallback
Multiple languages: Route to language-specific
Malformed: Preprocess carefully
Production requirement

### Monitoring and Metrics

Accuracy: Baseline metric
Latency: Response time
Throughput: Requests/sec
Cost: Computational efficiency
Drift: Does performance degrade?
User feedback: Ground truth check

### Updating Models

Data changes over time
Retraining schedule
A/B test: New vs old model
Gradual rollout: 1% → 10% → 100%
Rollback: If issues detected
Continuous improvement

## Dataset Construction

### Annotation Guidelines

Clear definitions
Examples with explanations
Disagreement resolution
Quality control
Inter-annotator agreement
Cohen's kappa metric

### Crowdsourcing

Hire many annotators
Low cost per example
Redundancy for quality
Platforms: Mechanical Turk, Upwork
Worker qualification
Often enables large-scale

### Dataset Biases

Selection bias: Who collected?
Labeler bias: Subjective decisions
Distribution: Representative sample?
Can affect model fairness
Audit datasets routinely
Better: Diversify sources

## Benchmarking

### Important Benchmarks

GLUE: 9 language understanding tasks
SQuAD: Reading comprehension
MNIST: Classic (simple now)
ImageNet: Huge vision benchmark
Common measures: Compare fairly
Leaderboards: Competitive drive

### Benchmark Gaming

Overfitting to benchmark
Not generalizing
Contamination: Test in training?
Solution: New datasets, real applications
Better metric: Few-shot on new task
Robust evaluation needed

## Computational Efficiency

### Model Size

BERT-base: 110M parameters
BERT-large: 340M parameters
Inference: Larger = slower
Mobile: Need lightweight
DistilBERT: 40% smaller, 97% accuracy
Pruning: Remove unimportant weights

### Quantization Revisited

FP32 → FP16: 2x faster
FP32 → INT8: 4x faster, 1% loss
TPU: INT8 only
Practical in production
Libraries: TensorFlow Lite, ONNX
Worth doing

### Inference Optimization

Batch requests: 10x speedup
Cache: Precompute embeddings
Model selection: Right size for task
Hardware: GPU if available
Serving: TensorFlow Serving, triton
Startup cost matters sometimes

## Responsible AI

### Fairness

Demographic parity: Same accuracy
Equalized odds: Similar errors
Calibration: Confidence = accuracy
Bias detection: Audit models
Mitigation: Better data, constraints
Ongoing work

### Privacy

Memorization: Models can memorize
Membership inference: Can extract training data?
Differential privacy: Formal guarantee
Federated learning: Train on device
Data retention: Keep only needed
Regulation: GDPR, local laws

### Transparency

Model cards: Document model
Data sheets: Document data
Limitations: Be honest
Failure cases: When it breaks
Bias statement: Known issues
Good practice

## Research Frontiers

### Open Problems

1. Longer context: Transformers limited
2. Reasoning: Models don't reason well
3. Grounding: Connect to reality
4. Efficiency: Smaller models
5. Multimodal: Text + images + audio
6. Interpretability: Why decisions?
7. Fairness: Autonomous bias

### Future Directions

Scaling laws: Bigger = better
Retrieval augmentation: Access knowledge
Sparse models: Only activate parts
Mixture of experts: Specialized models
Prompt engineering: Better queries
Few-shot learning: Rapid adaptation

## Capstone: End-to-End Project

**Build a Text Classification System**

1. Find dataset (download or create)
2. Explore data distribution
3. Baseline: TF-IDF + Logistic Regression
4. Embeddings: Word2Vec or fastText
5. Deep model: CNN/RNN
6. Error analysis and debugging
7. Optimize and deploy
8. Monitor performance
9. Write thorough report

## Suggested Capstones

1. Sentiment analysis (movie reviews)
2. Spam detection (emails)
3. Intent classification (chatbot)
4. Toxicity detection (comments)
5. Topic classification (news)
6. Language detection (multilingual)
7. Domain-specific classifier (your domain)

## Key Takeaways

**Understanding**
1. Text is high-dimensional data
2. Embeddings capture semantics
3. Context matters profoundly
4. Transfer learning enables low-resource
5. Models have biases and limitations
6. Interpretability is underrated
7. Production != research

**Technical Skills**
1. Text preprocessing
2. Embedding training and usage
3. Classification and sequence labeling
4. Error analysis
5. Model deployment
6. Performance monitoring
7. Responsible AI practices

## Module 05 Final Thoughts

This module covers NLP foundations.
From raw text to semantic understanding.
Tokenization through embeddings.
Classical methods still useful.
Foundation for transformer era.

**Timeline**
2013-2017: Embeddings reign
2018: ELMo changes everything
2018+: Transformer era
2023+: Large models dominate

You've learned the fundamentals.
Ready for modern NLP (module-06).
Ready for production systems.
Ready to contribute to research.

You know text as data. Excellent! 🎉
## Word2Vec Architecture Deep Dive

### Skip-gram Model

Goal: Predict context words from target word
Input: Word embedding
Output: Probability of context words
Loss: Cross-entropy over context window
Optimization: SGD with negative sampling

### Continuous Bag of Words (CBOW)

Opposite of Skip-gram
Input: Context words
Output: Predict center word
Faster training than skip-gram
Better for frequent words
Trade-off: Slightly lower quality

### Negative Sampling

Full softmax: O(V) expensive
Negative sampling: Sample K negative examples
Binary logistic regression instead
K typical values: 5-15
Faster: O(K) vs O(V)
Works surprisingly well!

### Hierarchical Softmax

Binary tree over vocabulary
Depth: O(log V)
Path probability product
Better for rare words
Slower than negative sampling
Historical importance

### Subword Information

FastText improvement
Represent as sum of character n-grams
Handle OOV words (morphology)
Example: "running" = run + ing + ...n-grams
Better performance on morphologically rich languages
Handle misspellings gracefully

## GloVe: Global Vectors

### Motivation

Word2Vec: Local context only
GloVe: Global corpus statistics + local context
Matrix factorization approach
Combine benefits of both
Excellent empirical results

### Co-occurrence Matrix

Count word co-occurrences in context window
V x V matrix (V = vocab size)
X_ij = count of word i with word j in context
Sparse: ~0.1% non-zero typical
Compress via SVD (but loses info)

### GloVe Loss Function

Weighted least squares
Loss = Σ f(X_ij) * (w_i · w_j + b_i + b_j - log X_ij)^2
f(X_ij): Weighting function (dampens rare pairs)
Global statistics captured
Better on analogies than Word2Vec

### Word Analogies

Test: king - man + woman = queen
Equation: w_king - w_man + w_woman ≈ w_queen
Linear relationships in embedding space
Remarkable property!
GloVe better than Word2Vec on this
But both still imperfect

### Context-dependent vs Static

Word2Vec/GloVe: Single vector per word
Problem: Homonyms (bank = river vs institution)
One vector can't capture both
Solution: Contextualized embeddings (ELMo, BERT)
Modern approach: Dynamic based on context

## ELMo: Embeddings from Language Models

### Key Insight

Train bidirectional LSTM language model
Extract hidden states
Weighted combination = ELMo
Context-dependent (solves homonym problem)
Transfer learning: Improves downstream tasks

### Bidirectional Language Model

Forward: Predict word from left context
Backward: Predict word from right context
Both directions: Full context
LSTM layers: 2 (original ELMo)
Hidden size: 4096
Fast: ~1B token corpus

### Representation Extraction

Layer 1: Character convolutions
Layers 2-3: Biphone LSTM
Extract: All 3 layer outputs
Concatenate: [char, lstm1, lstm2]
Weighted sum: λ * (γ * Σ s_k * h_k)
λ, γ, s_k: Task-specific learnable

### Fine-tuning Process

Downstream task: NER, SRL, etc.
Freeze ELMo weights
Learn weights for layer combination
Concatenate ELMo with task embeddings
Results: +2-4% F1 improvement typical

### Limitations

Slow: 1100M parameters
RNN-based: Sequential (not parallelizable)
Training time: 1-2 weeks
Next: Transformers (faster, better)
But ELMo was breakthrough (2018)

## BERT: Bidirectional Encoder Representations from Transformers

### Pre-training Objectives

1. Masked Language Model (MLM)
   - Replace 15% tokens with [MASK]
   - Predict original token
   - 80% [MASK], 10% random, 10% original
2. Next Sentence Prediction (NSP)
   - Given 2 sentences
   - Predict if adjacent in corpus
   - Binary classification

### Why Masking Works

Bidirectional context: See all words
Force model to understand meaning
Not cheating with position info
Deeper representations learned
Unlike GPT (causal, left-to-right)

### Pre-training Details

Corpus: BookCorpus + Wikipedia
3.3B tokens total
Vocab: 30K WordPiece tokens
Training: 16 TPUs, 4 days
Batch: 256 (large!)
Optimizer: Adam, LR=1e-4

### Fine-tuning

Add task-specific head
Train 2-4 epochs
Learning rate: 2e-5 (small!)
Batch: 16-32
Results: SOTA on GLUE
Simple but effective

### Variants

RoBERTa: Better pre-training (improved +1-3% accuracy)
ALBERT: Shared parameters (smaller)
DistilBERT: 40% smaller, 60% faster
ELECTRA: Different pre-training objective
Hundreds of variants now

## GPT: Generative Pre-trained Transformer

### Causal Language Modeling

Predict next token given history
P(w_t | w_1, ..., w_{t-1})
Autoregressive: Generate token by token
Masked attention: Can't see future
Simple but powerful objective

### Differences from BERT

BERT: Masked, bidirectional (understand)
GPT: Causal, left-to-right (generate)
BERT: Encoder-only (no generation)
GPT: Decoder-only (can generate)
Different strengths for different tasks

### Scaling Laws

Compute = 6ND (N=params, D=data tokens)
Loss ∝ N^(-a) where a ≈ 0.07-0.1
Language model scaling shows predictable trends
GPT-2: 1.5B params, impressive
GPT-3: 175B params, few-shot magic
Emergent abilities at scale

### Few-shot Learning

GPT-3: No fine-tuning needed
Task definition in prompt
Examples: 0-shot, 1-shot, few-shot
Works on translation, QA, reasoning
Remarkable generalization

### In-context Learning

Implicit fine-tuning in prompt
Model adapts to examples
No gradient updates
Purely in-context (during forward pass)
Mysterious mechanism (active research)

## Instruction Tuning and Alignment

### The Problem

GPT-3: Powerful but unpredictable
Can refuse tasks or be verbose
Harmful outputs possible
Solution: Fine-tune on instructions
Instruction-following is learnable skill

### Data Collection

Human-written instructions
Each with expected output
High quality: ~100K examples
Diversity: Many task types
Costs: $100K+ for good data
Or use model-generated (weaker)

### Training Process

Fine-tune base model on instructions
2-5 epochs typical
Learning rate: 1e-5 (small)
Quick: 1-2 hours on single GPU
Huge effect on capability
ChatGPT: Fine-tuned GPT-3.5

### Reinforcement Learning from Human Feedback

Step 1: Collect comparisons
Step 2: Train reward model
Step 3: Use PPO to optimize
Step 4: Generate better outputs
Iterative improvement
ChatGPT training approach

### Safety and Alignment

Constitutional AI: Principles-based
Red-teaming: Test for failures
Adversarial examples: Find weaknesses
Ongoing challenge
No silver bullet yet
Active research area (critical!)

## Prompt Engineering

### Zero-shot Prompting

No examples, just instructions
Simple case
Works surprisingly often
Baseline for comparison
Limits: Complex tasks fail

### Few-shot Prompting

Provide 1-5 examples
Model learns pattern from examples
Not actual fine-tuning
In-context learning
Often better than zero-shot

### Chain-of-Thought

Ask model to reason step-by-step
Improves accuracy on math +40%
Works with few examples
Forces explicit reasoning
Slower inference (longer)
But higher quality

### Self-Consistency

Generate multiple reasoning paths
Take majority vote
Improves accuracy on complex tasks
Cost: K times slower inference
K=5 typical (5x cost, 5-10% accuracy gain)

### Prompt Optimization

Gradient-based: LLM-based optimization
Discrete: Genetic algorithms
Manual: Human expertise
Prompt templates: Reusable patterns
Active research: AutoPrompt, etc.

## Semantic Search and Dense Retrieval

### Dense Passage Retrieval

Encode queries and documents
Retrieve by similarity
FAISS: Fast similarity search
100M documents: Milliseconds
Much better than BM25 keyword matching

### Bi-Encoders

Query encoder and document encoder
Independent (can batch both separately)
Fast inference
Similarity: Dot product or cosine
Usually good enough

### Cross-Encoders

Jointly encode query and document
Higher quality than bi-encoder
Slower: Must evaluate all pairs
Use as re-ranker
Workflow: Bi-encoder → cross-encoder

### Contrastive Learning

Positive: Query with relevant doc
Negative: Query with irrelevant doc
Loss: Maximize positive, minimize negative
SimCLR: Self-supervised version
Data efficient

### Knowledge Distillation for Search

Large cross-encoder → small bi-encoder
Student learns from teacher
Fast inference with quality
Production approach
Typical: 5x speedup, 90% quality

## Word2Vec Architecture Deep Dive

### Skip-gram Model

Goal: Predict context words from target word
Input: Word embedding
Output: Probability of context words
Loss: Cross-entropy over context window
Optimization: SGD with negative sampling

### Continuous Bag of Words (CBOW)

Opposite of Skip-gram
Input: Context words
Output: Predict center word
Faster training than skip-gram
Better for frequent words
Trade-off: Slightly lower quality

### Negative Sampling

Full softmax: O(V) expensive
Negative sampling: Sample K negative examples
Binary logistic regression instead
K typical values: 5-15
Faster: O(K) vs O(V)
Works surprisingly well!

### Hierarchical Softmax

Binary tree over vocabulary
Depth: O(log V)
Path probability product
Better for rare words
Slower than negative sampling
Historical importance

### Subword Information

FastText improvement
Represent as sum of character n-grams
Handle OOV words (morphology)
Example: "running" = run + ing + ...n-grams
Better performance on morphologically rich languages
Handle misspellings gracefully

## GloVe: Global Vectors

### Motivation

Word2Vec: Local context only
GloVe: Global corpus statistics + local context
Matrix factorization approach
Combine benefits of both
Excellent empirical results

### Co-occurrence Matrix

Count word co-occurrences in context window
V x V matrix (V = vocab size)
X_ij = count of word i with word j in context
Sparse: ~0.1% non-zero typical
Compress via SVD (but loses info)

### GloVe Loss Function

Weighted least squares
Loss = Σ f(X_ij) * (w_i · w_j + b_i + b_j - log X_ij)^2
f(X_ij): Weighting function (dampens rare pairs)
Global statistics captured
Better on analogies than Word2Vec

### Word Analogies

Test: king - man + woman = queen
Equation: w_king - w_man + w_woman ≈ w_queen
Linear relationships in embedding space
Remarkable property!
GloVe better than Word2Vec on this
But both still imperfect

### Context-dependent vs Static

Word2Vec/GloVe: Single vector per word
Problem: Homonyms (bank = river vs institution)
One vector can't capture both
Solution: Contextualized embeddings (ELMo, BERT)
Modern approach: Dynamic based on context

## Word2Vec Architecture Deep Dive

### Skip-gram Model

Goal: Predict context words from target word
Input: Word embedding
Output: Probability of context words
Loss: Cross-entropy over context window
Optimization: SGD with negative sampling

### Continuous Bag of Words (CBOW)

Opposite of Skip-gram
Input: Context words
Output: Predict center word
Faster training than skip-gram
Better for frequent words
Trade-off: Slightly lower quality

### Negative Sampling

Full softmax: O(V) expensive
Negative sampling: Sample K negative examples
Binary logistic regression instead
K typical values: 5-15
Faster: O(K) vs O(V)
Works surprisingly well!

### Hierarchical Softmax

Binary tree over vocabulary
Depth: O(log V)
Path probability product
Better for rare words
Slower than negative sampling
Historical importance

### Subword Information

FastText improvement
Represent as sum of character n-grams
Handle OOV words (morphology)
Example: "running" = run + ing + ...n-grams
Better performance on morphologically rich languages
Handle misspellings gracefully

## GloVe: Global Vectors

### Motivation

Word2Vec: Local context only
GloVe: Global corpus statistics + local context
Matrix factorization approach
Combine benefits of both
Excellent empirical results

### Co-occurrence Matrix

Count word co-occurrences in context window
V x V matrix (V = vocab size)
X_ij = count of word i with word j in context
Sparse: ~0.1% non-zero typical
Compress via SVD (but loses info)

### GloVe Loss Function

Weighted least squares
Loss = Σ f(X_ij) * (w_i · w_j + b_i + b_j - log X_ij)^2
f(X_ij): Weighting function (dampens rare pairs)
Global statistics captured
Better on analogies than Word2Vec

### Word Analogies

Test: king - man + woman = queen
Equation: w_king - w_man + w_woman ≈ w_queen
Linear relationships in embedding space
Remarkable property!
GloVe better than Word2Vec on this
But both still imperfect

### Context-dependent vs Static

Word2Vec/GloVe: Single vector per word
Problem: Homonyms (bank = river vs institution)
One vector can't capture both
Solution: Contextualized embeddings (ELMo, BERT)
Modern approach: Dynamic based on context

## ELMo: Embeddings from Language Models

### Key Insight

Train bidirectional LSTM language model
Extract hidden states
Weighted combination = ELMo
Context-dependent (solves homonym problem)
Transfer learning: Improves downstream tasks

### Bidirectional Language Model

Forward: Predict word from left context
Backward: Predict word from right context
Both directions: Full context
LSTM layers: 2 (original ELMo)
Hidden size: 4096
Fast: ~1B token corpus

### Representation Extraction

Layer 1: Character convolutions
Layers 2-3: Biphone LSTM
Extract: All 3 layer outputs
Concatenate: [char, lstm1, lstm2]
Weighted sum: λ * (γ * Σ s_k * h_k)
λ, γ, s_k: Task-specific learnable

### Fine-tuning Process

Downstream task: NER, SRL, etc.
Freeze ELMo weights
Learn weights for layer combination
Concatenate ELMo with task embeddings
Results: +2-4% F1 improvement typical

### Limitations

Slow: 1100M parameters
RNN-based: Sequential (not parallelizable)
Training time: 1-2 weeks
Next: Transformers (faster, better)
But ELMo was breakthrough (2018)

## BERT: Bidirectional Encoder Representations from Transformers

### Pre-training Objectives

1. Masked Language Model (MLM)
   - Replace 15% tokens with [MASK]
   - Predict original token
   - 80% [MASK], 10% random, 10% original
2. Next Sentence Prediction (NSP)
   - Given 2 sentences
   - Predict if adjacent in corpus
   - Binary classification

### Why Masking Works

Bidirectional context: See all words
Force model to understand meaning
Not cheating with position info
Deeper representations learned
Unlike GPT (causal, left-to-right)

### Pre-training Details

Corpus: BookCorpus + Wikipedia
3.3B tokens total
Vocab: 30K WordPiece tokens
Training: 16 TPUs, 4 days
Batch: 256 (large!)
Optimizer: Adam, LR=1e-4

### Fine-tuning

Add task-specific head
Train 2-4 epochs
Learning rate: 2e-5 (small!)
Batch: 16-32
Results: SOTA on GLUE
Simple but effective

### Variants

RoBERTa: Better pre-training (improved +1-3% accuracy)
ALBERT: Shared parameters (smaller)
DistilBERT: 40% smaller, 60% faster
ELECTRA: Different pre-training objective
Hundreds of variants now

## GPT: Generative Pre-trained Transformer

### Causal Language Modeling

Predict next token given history
P(w_t | w_1, ..., w_{t-1})
Autoregressive: Generate token by token
Masked attention: Can't see future
Simple but powerful objective

### Differences from BERT

BERT: Masked, bidirectional (understand)
GPT: Causal, left-to-right (generate)
BERT: Encoder-only (no generation)
GPT: Decoder-only (can generate)
Different strengths for different tasks

### Scaling Laws

Compute = 6ND (N=params, D=data tokens)
Loss ∝ N^(-a) where a ≈ 0.07-0.1
Language model scaling shows predictable trends
GPT-2: 1.5B params, impressive
GPT-3: 175B params, few-shot magic
Emergent abilities at scale

### Few-shot Learning

GPT-3: No fine-tuning needed
Task definition in prompt
Examples: 0-shot, 1-shot, few-shot
Works on translation, QA, reasoning
Remarkable generalization

### In-context Learning

Implicit fine-tuning in prompt
Model adapts to examples
No gradient updates
Purely in-context (during forward pass)
Mysterious mechanism (active research)

## Instruction Tuning and Alignment

### The Problem

GPT-3: Powerful but unpredictable
Can refuse tasks or be verbose
Harmful outputs possible
Solution: Fine-tune on instructions
Instruction-following is learnable skill

### Data Collection

Human-written instructions
Each with expected output
High quality: ~100K examples
Diversity: Many task types
Costs: $100K+ for good data
Or use model-generated (weaker)

### Training Process

Fine-tune base model on instructions
2-5 epochs typical
Learning rate: 1e-5 (small)
Quick: 1-2 hours on single GPU
Huge effect on capability
ChatGPT: Fine-tuned GPT-3.5

### Reinforcement Learning from Human Feedback

Step 1: Collect comparisons
Step 2: Train reward model
Step 3: Use PPO to optimize
Step 4: Generate better outputs
Iterative improvement
ChatGPT training approach

### Safety and Alignment

Constitutional AI: Principles-based
Red-teaming: Test for failures
Adversarial examples: Find weaknesses
Ongoing challenge
No silver bullet yet
Active research area (critical!)

## Prompt Engineering

### Zero-shot Prompting

No examples, just instructions
Simple case
Works surprisingly often
Baseline for comparison
Limits: Complex tasks fail

### Few-shot Prompting

Provide 1-5 examples
Model learns pattern from examples
Not actual fine-tuning
In-context learning
Often better than zero-shot

### Chain-of-Thought

Ask model to reason step-by-step
Improves accuracy on math +40%
Works with few examples
Forces explicit reasoning
Slower inference (longer)
But higher quality

### Self-Consistency

Generate multiple reasoning paths
Take majority vote
Improves accuracy on complex tasks
Cost: K times slower inference
K=5 typical (5x cost, 5-10% accuracy gain)

### Prompt Optimization

Gradient-based: LLM-based optimization
Discrete: Genetic algorithms
Manual: Human expertise
Prompt templates: Reusable patterns
Active research: AutoPrompt, etc.

## Semantic Search and Dense Retrieval

### Dense Passage Retrieval

Encode queries and documents
Retrieve by similarity
FAISS: Fast similarity search
100M documents: Milliseconds
Much better than BM25 keyword matching

