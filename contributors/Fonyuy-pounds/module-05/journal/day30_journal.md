# Day 30: Semantic Search Engine - Learning Journal

## Overview

Day 30 completes Module 05 with a comprehensive mini-project: building a **semantic search engine** that integrates all concepts from Days 25-29. This marks the transition from learning isolated NLP techniques to building production-grade systems.

---

## 1. Information Retrieval Foundations

### 1.1 Document Ranking Problem

Given:
- Query: $q = \{w_1, w_2, \ldots, w_m\}$ (terms)
- Documents: $D = \{d_1, d_2, \ldots, d_n\}$

Goal: Rank documents by relevance score $\text{score}(q, d_i)$

### 1.2 Vector Space Model

Represent documents and queries as vectors in high-dimensional space:
$$d_i = [x_1, x_2, \ldots, x_V] \in \mathbb{R}^V$$

where $V$ is vocabulary size and each dimension represents a term or feature.

Similarity between query and document:
$$\text{cosine\_sim}(q, d) = \frac{q \cdot d}{\|q\| \cdot \|d\|} = \frac{\sum_{j=1}^{V} q_j d_j}{\sqrt{\sum q_j^2} \sqrt{\sum d_j^2}}$$

**Properties:**
- Range: $[-1, 1]$ (typically $[0, 1]$ for normalized vectors)
- 1.0 = identical direction (perfect match)
- 0.0 = orthogonal (no overlap)

---

## 2. TF-IDF Baseline

### 2.1 Term Frequency (TF)

Measures how frequently a term appears in a document:
$$\text{TF}(t, d) = \frac{\text{count}(t, d)}{|d|}$$

where:
- $\text{count}(t, d)$ = occurrences of term $t$ in document $d$
- $|d|$ = total terms in document $d$

**Intuition:** Common terms within a document are more relevant.

### 2.2 Inverse Document Frequency (IDF)

Measures how rare a term is across all documents:
$$\text{IDF}(t) = \log\left(\frac{N}{\text{df}(t)}\right)$$

where:
- $N$ = total number of documents
- $\text{df}(t)$ = number of documents containing term $t$

**Intuition:** Rare terms are more discriminative (better for ranking).

### 2.3 TF-IDF Weighting

Combine TF and IDF:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

**Example:** For a corpus with 1000 documents:
- Term "the" appears in 900 documents: $\text{IDF} = \log(1000/900) \approx 0.046$ (low weight)
- Term "seraph" appears in 2 documents: $\text{IDF} = \log(1000/2) \approx 2.70$ (high weight)

---

## 3. Semantic Similarity with Embeddings

### 3.1 Limitations of TF-IDF

TF-IDF is **discrete** (exact term matching):
- Query: "celestial beings"
- Document contains: "angels" but not "beings"
- TF-IDF similarity: 0 (despite semantic overlap)

### 3.2 Embedding-Based Similarity

Use learned representations (Day 26: Word2Vec, Day 29: ELMo, GPT):

**Document embedding:**
$$\vec{d} = \frac{1}{|d|} \sum_{w \in d} \vec{w}$$

(Average word embeddings)

**Cosine similarity:**
$$\text{sim}(q, d) = \cos(\vec{q}, \vec{d})$$

**Advantages:**
- Captures semantic similarity (synonyms get similar embeddings)
- Continuous scores (not just binary term matching)
- Works with unseen terms (if OOV handling is implemented)

---

## 4. Named Entity Recognition (NER) in Search

### 4.1 Entity-Enhanced Indexing

Index entities separately:

$$\text{EntityIndex}[e] = \{\text{doc indices containing entity } e\}$$

Example:
```
EntityIndex["God"] = {0, 3, 4, 12, 13, 14, 16, 18}
EntityIndex["Light"] = {3, 4, 13, 14, 15, 16, 17, 18}
EntityIndex["Waters"] = {1, 8, 9}
```

### 4.2 Hybrid Retrieval

Combine semantic and entity-based search:

1. **Semantic phase:** Find top-k documents by embedding similarity
2. **Entity phase:** Find documents containing query entities
3. **Merge:** Combine results, prioritizing semantic matches

$$\text{score}_{\text{hybrid}}(q, d) = \alpha \cdot \text{sim}(q, d) + (1-\alpha) \cdot \text{entity\_score}(q, d)$$

where $\alpha \in [0, 1]$ weights semantic vs. entity components.

---

## 5. Retrieval Evaluation Metrics

### 5.1 Precision and Recall

For a query $q$, given:
- $R_q$ = set of relevant documents (ground truth)
- $A_q$ = set of retrieved documents (ranked at position 1 to k)

**Precision@k:**
$$P@k = \frac{|R_q \cap A_q|}{k}$$

Measures: "Of the k results returned, how many are relevant?"

**Recall@k:**
$$\text{Recall}@k = \frac{|R_q \cap A_q|}{|R_q|}$$

Measures: "Of all relevant documents, how many did we find?"

### 5.2 F1 Score

Harmonic mean of precision and recall:
$$F1 = 2 \cdot \frac{P \cdot R}{P + R}$$

- High precision, low recall: System is conservative (few false positives)
- Low precision, high recall: System is aggressive (many false negatives)
- F1 balances both

### 5.3 Mean Average Precision (MAP)

Aggregate metric across queries:

For query $q$ at position $i$:
$$\text{AP}_q = \frac{1}{|R_q|} \sum_{i=1}^{k} P(i) \cdot \text{rel}(i)$$

where $\text{rel}(i) \in \{0, 1\}$ indicates if result at position $i$ is relevant.

$$\text{MAP} = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}_q$$

---

## 6. Information Retrieval Pipeline

### 6.1 Indexing Phase

1. **Document collection:** Gather all documents
2. **Preprocessing:** Tokenize, lowercase, remove stopwords
3. **Vectorization:** Convert documents to embeddings
4. **Index storage:** Build efficient data structures (vector index, entity index)

### 6.2 Query Phase

1. **Query preprocessing:** Apply same preprocessing as indexing
2. **Query embedding:** Convert query to embedding
3. **Similarity computation:** Calculate similarity with all indexed documents
4. **Ranking:** Sort by similarity score
5. **Result filtering:** Apply optional metadata filters (date, category, etc.)

### 6.3 Scalability Considerations

**Challenge:** Computing similarity with all $N$ documents is $O(N \cdot d)$ where $d$ = embedding dimension.

**Solutions:**
- **Approximate Nearest Neighbors (ANN):** Locality-sensitive hashing (LSH), FAISS, HNSW
- **Hierarchical indexing:** Cluster documents into groups
- **Caching:** Pre-compute similarities for frequent queries

---

## 7. Module 05 Integration

### 7.1 Full NLP Pipeline

Day 30 consolidates Days 25-29:

| Day | Concept | Component |
|-----|---------|-----------|
| 25 | Text preprocessing | Tokenization, lowercase |
| 26 | Word embeddings | TF-IDF baseline, Word2Vec fallback |
| 27 | Sequence modeling | Document embedding (average pooling) |
| 28 | NER | Entity indexing, hybrid search |
| 29 | Contextual embeddings | Optional ELMo/GPT-2 for better quality |
| 30 | Search engine | Complete IR system |

### 7.2 Example: "Light Creation" Query

1. **Preprocess query:** "light creation" → ["light", "creation"]
2. **Embed query:** 
   - TF-IDF: sparse vector with non-zero values at "light" and "creation" indices
   - Word2Vec: dense vector = average of word embeddings
3. **Extract entities:** ["Light", "Creation"] (capitalized)
4. **Entity search:** Find documents with these words
5. **Semantic search:** Compute cosine similarity with all documents
6. **Hybrid merge:** Combine results
7. **Return top-5:** Rank documents by combined score

---

## 8. Banso Linguistic Integration

### 8.1 Theological Vocabulary

Banso terms in Genesis translation:
- **nsi** (creation, world): Genesis 1:1, 1:6, 1:11, etc.
- **bibor** (spirit, wind): Genesis 1:2, 1:3
- **ayaa** (light, clarity): Genesis 1:3, 1:4, 1:14-1:18
- **chi** (water, life force): Genesis 1:2, 1:6, 1:9-1:10
- **mbang** (sky, heaven, expanse): Genesis 1:7, 1:8, 1:14-1:16
- **kibor** (power, strength, dominion): Genesis 1:26, 1:28

### 8.2 Contextual Analysis

Query: "creation power"

Without contextualization (TF-IDF):
- Matches documents with both "creation" and "power"

With contextualization (Banso-aware):
- Matches documents with **nsi** (creation) and **kibor** (power)
- Ranks Genesis 1:26-1:28 (Banso cosmology focus) higher

### 8.3 Cross-Linguistic Transfer

Principles apply to any low-resource language:
1. Build semantic search system with available embeddings
2. Integrate linguistic domain knowledge (theological, cultural terms)
3. Evaluate with native speaker judgments
4. Iteratively improve rankings

---

## 9. Advanced Topics (Future Extensions)

### 9.1 Dense Passage Retrieval (DPR)

Replace TF-IDF/Word2Vec with contextual embeddings:
- Encode queries and passages separately using transformers
- Retrieve by embedding similarity in dense space
- Fine-tune on task-specific data

### 9.2 Re-ranking

Two-stage pipeline:
1. **Retriever:** Fast approximate search (ANN)
2. **Re-ranker:** Expensive but accurate ranking of top-100

Example re-ranker: Cross-encoder that scores (query, document) pairs

### 9.3 Multilingual Retrieval

Use multilingual embeddings (e.g., multilingual BERT) for:
- Query in English, search corpus in Banso
- Cross-lingual semantic matching

### 9.4 Query Expansion

Augment query with synonyms/related terms:

Query: "light" → "light creation illumination revelation"

Improves recall by matching documents with synonyms.

---

## 10. Summary

**Semantic search engine** combines:
- **Preprocessing:** Clean and tokenize text
- **Embeddings:** Dense representations (TF-IDF or neural)
- **Similarity:** Cosine distance in embedding space
- **NER:** Entity-enhanced indexing
- **Ranking:** Evaluate precision, recall, F1
- **Contextualization:** Leverage Banso theological vocabulary

**Key insight:** Modern NLP systems are **pipelines**, not isolated models. Day 30 demonstrates integrating techniques from Days 25-29 into a coherent, production-grade information retrieval system.

---

## References

1. **Vector Space Model:** Salton, G., Wong, A., Yang, C. S. (1975). "A vector space model for automatic indexing."
2. **TF-IDF:** Spärck Jones, K. (1972). "A statistical interpretation of term specificity and its application in retrieval."
3. **Embedding-based Retrieval:** Gillick, D., et al. (2019). "Learning Dense Representations for Entity Retrieval."
4. **NER in IR:** Bhattacharya, I., & Getoor, L. (2007). "Collective entity resolution in relational data."
5. **Banso Documentation:** MarkGPT Module 05, Days 25-29
