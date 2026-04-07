# Tokenization Deep Dive: Understanding BPE

## Why Subword Tokenization?

Three levels of tokenization:
1. **Character-level**: Every character is a token
   - Pros: Small vocabulary (256 tokens)
   - Cons: Long sequences, hard to learn patterns
2. **Word-level**: Every word is a token
   - Pros: Compact sequences
   - Cons: Large vocabulary (10K-100K), OOV (out-of-vocabulary) substrings
3. **Subword-level** (BPE): Best of both worlds
   - Vocabulary: 5K-50K tokens
   - Learns common substrings as single tokens

## BPE Algorithm

```
Step 1: Initialize: Each character is a token
Step 2: Count bigram frequencies (adjacent token pairs)
Step 3: Merge the most frequent bigram
Step 4: Repeat steps 2-3 until vocab size reached
```

### Worked Example

Initial (character-level):
```
"dog", "cat", "dog"
→ d o g c a t d o g
```

Most frequent bigram: "o" + "g" (appears 2x as "og")

After merge 1:
```
d og c a t d og
```

Continue merging...

## Fertility: A Key Metric

**Fertility** = average tokens per word

Lower is better!

Example:
- Character-level on "beautiful": 9 tokens → fertility 1.0 (1 word, 9 tokens)
- Word-level on "beautiful": 1 token → fertility 1.0
- BPE (8K vocab) on "beautiful": 2-3 tokens → fertility ~1.3

Typical values:
- English KJV: 1.3-1.5
- Banso text: 1.4-1.8 (complex morphology)

## Banso Language Challenges

1. **Phonotactics**: Nso' has specific consonant clusters uncommon in English
2. **Affixation**: Rich prefix/suffix system (e.g., a-, ba-, -i, -a)
3. **Tone markers**: Optional but linguistically important
4. **Morphological complexity**: Single words can encode much information

## References

- Sennrich et al. (2016): "Neural Machine Translation of Rare Words with Subword Units"
- Kudo & Richardson (2018): "SentencePiece: A simple and language independent subword tokenizer"
- Bostrom & Durrett (2020): "Byte Pair Encoding is Suboptimal for Language Model Pretraining"
## Deep Dive: Tokenization Algorithms

### Longest Match First

Greedy: Always take longest token
Efficient: O(1) lookup
Problem: Suboptimal ("hello" as char)
Used in: Early systems

### Maximum Likelihood

Probability of tokenization
P(t1, t2, ..., tn) = P(t1)*P(t2|t1)*...
Viterbi: Find maximum path
Better quality
Computationally expensive

### Unigram Language Model

Simplest LM
P(token) = count(token) / total
Works surprisingly well
Very fast
Foundation of many systems

### Byte-Level Tokenization

UTF-8 bytes: 0-255 vocab
Can represent any text
No OOV!
Very long sequences
Used in: GPT-2

### Unicode Normalization

é = e + combining accent
Different byte sequences, same character
NFC vs NFD formats
Should normalize first
Affects tokenization

### Language-Specific Tokenization

Chinese: No spaces (use jieba)
Arabic: Right-to-left
Vietnamese: Diacritics important
Thai: No word boundaries
Japanese: Kanji vs hiragana
Language matters!

### Trie-Based Tokenization

Prefix tree structure
O(n) tokenization
Efficient matching
Works with variable vocab
Used in BERT

### Entropy-Based Analysis

Tokenization ambiguity
Multiple valid segmentations
Entropy measures uncertainty
High entropy: Fewer good options
Evaluates tokenizer quality

### Dynamic Programming

Optimal substructure
Best(0..n) = min over splits
Viterbi-like algorithm
Guaranteed optimal
O(n^2) time

### Morphologically-Aware

Segment respecting morphology
"running" = "run" + "ing"
Preserves linguistic structure
Better for low-resource
Multilingual advantage

### Domain-Specific Tokenization

Medical: Keep domain terms
Programming: Split operators
Social media: Hashtags, @mentions
Customize for domain
Improves downstream

### Character n-gram Fallback

No exact match?
Use character n-grams
Can represent anything
Slow but complete
Hybrid systems

### Reversibility and Reconstruction

Can we reconstruct original text?
Tokenization is lossy
Some tokens merge forever
"New" vs "Ne" + "w"
Important for applications

### Token Embeddings vs Word Embeddings

Subword tokens: Different embedding space
Composition for words
Average or max pooling
Better OOV handling
Trade-off: representation vs efficiency

