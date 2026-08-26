# [Day 30] Semantic Search Engine - Module 05 Mini-Project (Fonyuy-pounds)

Implements the Day 30 Module 05 capstone: a complete semantic search system that integrates text preprocessing, embeddings, sequence modeling, NER, and contextual representations into a production-grade information retrieval pipeline.

Key contributions:

- **TF-IDF Baseline:** Efficient offline term-weighting vector space model
- **Embedding-based Retrieval:** Word2Vec (fallback) or dense contextual embeddings
- **Cosine Similarity Ranking:** Vector space model for semantic matching
- **Named Entity Recognition Integration:** Entity-indexed hybrid retrieval
- **Corpus Indexing:** Efficient document vectorization and storage
- **Retrieval Evaluation:** Precision, recall, F1, MAP metrics
- **Hybrid Search:** Combine semantic and entity-based ranking
- **Banso Bible Corpus:** Genesis verses with Banso theological vocabulary
- **Cross-Linguistic Analysis:** Demonstrate applicability to low-resource languages
- **Module 05 Integration:** Consolidate Days 25-29 into unified system
- **Scalability Considerations:** Discussion of ANN, re-ranking, multilingual retrieval
- **Dynamic Workspace Detection:** Local + Google Colab compatible

Contributor: Fonyuy-pounds

## 📝 Description

This PR presents the **Module 05 capstone mini-project**: a semantic search engine that retrieves documents semantically similar to a query. Rather than exact keyword matching, the system uses learned embeddings and NER to understand meaning and context.

**What is Semantic Search?**

Traditional search (Google, Elasticsearch):
- Query: "Who is the creator of the universe?"
- Matches documents with exact terms: "creator", "universe"
- Misses: "architect of cosmos", "maker of all things"

Semantic search:
- Understands meaning, not just terms
- Query: "Who is the creator of the universe?"
- Matches: "architect of cosmos" (synonym match), "In the beginning God created..." (semantic overlap)

**System Architecture:**

1. **Indexing Phase:**
   - Preprocess documents (tokenize, lowercase)
   - Compute embeddings (TF-IDF or Word2Vec)
   - Extract named entities
   - Store vectors and metadata

2. **Query Phase:**
   - Preprocess query
   - Embed query
   - Compute similarity with all indexed documents
   - Rank by similarity
   - Return top-k results

3. **Evaluation:**
   - Measure precision, recall, F1
   - Compare semantic vs. entity-based retrieval
   - Analyze performance by query type

**Key Features:**

- **SimpleEmbedder Class:** Unified interface for TF-IDF and Word2Vec embeddings
- **SemanticSearchEngine:** Full IR system with indexing and ranking
- **EntityAwareSearchEngine:** Enhanced retrieval with NER-based filtering
- **Hybrid Retrieval:** Combines semantic and entity-based ranking
- **Banso Bible Corpus:** Genesis verses with cross-linguistic annotations
- **Evaluation Metrics:** Precision@k, Recall@k, F1, Mean Average Precision
- **Linguistic Analysis:** How Banso theological terms contextualize meanings

**Mathematical Foundation:**

- Vector Space Model: Represent documents as vectors
- Cosine Similarity: $\cos(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}| |\vec{d}|}$
- TF-IDF Weighting: $\text{TF-IDF} = \text{TF} \times \text{IDF}$
- Precision/Recall: Evaluate ranking quality

**Learning Objectives:**

After Day 30, students should understand:
1. How to build an end-to-end information retrieval system
2. Difference between exact matching (TF-IDF) and semantic matching (embeddings)
3. How NER enhances search precision
4. Evaluation metrics for ranking systems
5. How modern search engines combine multiple signals

---

## 🎯 Type of Change

- [x] 🎓 New lesson or exercise content
- [x] ✨ New feature
- [x] 📚 Documentation improvement
- [ ] 🐛 Bug fix (non-breaking)
- [ ] ♻️ Code refactor
- [ ] 🧪 Test addition
- [ ] 🔧 Other: ____________

## 📖 Related Module(s)

- Module 05: NLP Foundations (Days 25-30, Capstone)

## 🧪 Testing

- [x] All python scripts run without errors locally and in Google Colab
- [x] Indexing pipeline tested on sample corpus (Genesis corpus, 19 verses)
- [x] Semantic search produces ranked results ordered by similarity
- [x] Entity extraction identifies capitalized words correctly
- [x] Hybrid search combines semantic and entity results appropriately
- [x] Evaluation metrics computed correctly (precision, recall, F1)
- [x] TF-IDF baseline produces expected similarity scores
- [x] Word2Vec optional dependency handled gracefully
- [x] Banso theological vocabulary integrated in corpus metadata
- [x] Cosine similarity values in range [0, 1]

## ✅ Checklist

- [x] Followed style guidelines ([BEST_PRACTICES.md](../BEST_PRACTICES.md))
- [x] Added comprehensive docstrings and comments
- [x] No hardcoded file paths (dynamic workspace detection)
- [x] Commit messages follow conventional format
- [x] No large files committed
- [x] Clear learning objectives documented
- [x] Full working examples included
- [x] Updated README with Day 30 milestone
- [x] Mathematical derivations in learning journal
- [x] Integration with previous days (25-29) explained

## 📸 Code Examples

**Basic Semantic Search:**
```python
engine = SemanticSearchEngine()
engine.index_documents(documents, metadata)

results = engine.search("light creation", top_k=5)
for idx, doc, similarity in results:
    print(f"[{similarity:.3f}] {doc}")
```

**Entity-Enhanced Search:**
```python
entity_engine = EntityAwareSearchEngine()
entity_engine.index_documents(documents, metadata)

results = entity_engine.hybrid_search("God and light", top_k=3)
for result in results:
    print(f"[{result['search_type']}] {result['document']}")
```

**Evaluation:**
```python
precision = true_positives / retrieved_count
recall = true_positives / relevant_count
f1 = 2 * (precision * recall) / (precision + recall)
```

## 📌 Additional Notes

- **TF-IDF vs. Embeddings:** TF-IDF is fast and interpretable; embeddings capture semantic similarity
- **Scalability:** For large corpora, use approximate nearest neighbors (FAISS, HNSW)
- **Re-ranking:** Use expensive models (cross-encoders) to re-rank top-k results from fast retriever
- **Multilingual:** Multilingual embeddings enable cross-lingual search
- **Query Expansion:** Augment queries with synonyms to improve recall
- **Banso Integration:** Demonstrates applicability to low-resource languages
- **Module 05 Summary:**
  - Day 25: Preprocessing & tokenization
  - Day 26: Word embeddings (Word2Vec, GloVe)
  - Day 27: Sequence models (RNNs for document encoding)
  - Day 28: NER with CRF (entity extraction)
  - Day 29: Contextual embeddings (ELMo, GPT)
  - Day 30: Complete IR system integrating all components
- Fully compatible with Google Colab environment

---

**Thanks for contributing to MarkGPT!** 🚀
See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.
