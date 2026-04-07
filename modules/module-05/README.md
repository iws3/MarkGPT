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

