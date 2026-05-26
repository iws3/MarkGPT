"""
Day 30: Module 05 Review & Mini-Project - Semantic Search Engine
================================================================

Build a complete semantic search system that consolidates concepts from Days 25-29:
- Text preprocessing (Day 25)
- Word embeddings (Day 26)  
- Sequence models (Day 27)
- Named Entity Recognition (Day 28)
- Contextual embeddings (Day 29)

This mini-project demonstrates a realistic NLP pipeline: ingesting documents,
vectorizing them with contextual embeddings, and retrieving semantically similar
results using the Banso Bible corpus as a knowledge base.

Author: Fonyuy-pounds
Date: 2026
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

# Detect workspace root (local or Colab)
if "google.colab" in sys.modules:
    workspace_root = "/content/drive/MyDrive/MarkGPT"
else:
    workspace_root = Path(__file__).resolve().parent.parent.parent.parent.parent

sys.path.insert(0, str(workspace_root))

print(f"Workspace root: {workspace_root}")


# =============================================================================
# PART 1: SIMPLE EMBEDDING MODEL (TF-IDF + Word2Vec Fallback)
# =============================================================================

class SimpleEmbedder:
    """
    Lightweight embedding model that works offline without external dependencies.
    Falls back to TF-IDF if Word2Vec not available.
    """
    
    def __init__(self, embedding_dim: int = 100):
        """
        Initialize embedder.
        
        Args:
            embedding_dim: Dimensionality of embeddings (used for Word2Vec fallback)
        """
        self.embedding_dim = embedding_dim
        self.vocab = {}
        self.idf_scores = {}
        self.documents = []
        self.embeddings = {}
        self.use_gensim = False
        
        # Try to import gensim for Word2Vec
        try:
            from gensim.models import Word2Vec
            self.Word2Vec = Word2Vec
            self.use_gensim = True
            print("✓ Using Gensim Word2Vec for embeddings")
        except ImportError:
            print("⊘ Gensim not available; using TF-IDF fallback")
    
    def preprocess(self, text: str) -> List[str]:
        """
        Simple tokenization and lowercasing.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        import re
        text = text.lower()
        # Remove non-alphanumeric except spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = text.split()
        return [t for t in tokens if t]  # Remove empty
    
    def build_vocab(self, documents: List[str]):
        """
        Build vocabulary from documents.
        
        Args:
            documents: List of document strings
        """
        self.documents = documents
        tokenized = [self.preprocess(doc) for doc in documents]
        
        # Build vocab
        for tokens in tokenized:
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        
        # Compute IDF scores
        doc_freq = defaultdict(int)
        for tokens in tokenized:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        num_docs = len(documents)
        for token, freq in doc_freq.items():
            # IDF = log(N / df)
            self.idf_scores[token] = np.log(num_docs / freq) if freq > 0 else 0
        
        print(f"Built vocabulary with {len(self.vocab)} unique tokens")
    
    def get_tfidf_vector(self, text: str) -> np.ndarray:
        """
        Compute TF-IDF vector for text.
        
        Args:
            text: Input text
            
        Returns:
            TF-IDF vector
        """
        tokens = self.preprocess(text)
        vec = np.zeros(len(self.vocab))
        
        # Compute TF
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        
        # Compute TF-IDF
        for token, count in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                tfidf = (count / len(tokens)) * self.idf_scores.get(token, 0)
                vec[idx] = tfidf
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed text using available method.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if self.use_gensim:
            return self.get_word2vec_vector(text)
        else:
            return self.get_tfidf_vector(text)
    
    def get_word2vec_vector(self, text: str) -> np.ndarray:
        """
        Compute Word2Vec vector by averaging token embeddings.
        
        Args:
            text: Input text
            
        Returns:
            Average Word2Vec vector
        """
        tokens = self.preprocess(text)
        if not tokens:
            return np.zeros(self.embedding_dim)
        
        vectors = []
        for token in tokens:
            try:
                vectors.append(self.model.wv[token])
            except KeyError:
                # OOV token - use random vector
                vectors.append(np.random.randn(self.embedding_dim))
        
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.embedding_dim)
    
    def train_word2vec(self):
        """Train Word2Vec model on documents."""
        if not self.use_gensim:
            return
        
        tokenized = [self.preprocess(doc) for doc in self.documents]
        self.model = self.Word2Vec(
            sentences=tokenized,
            vector_size=self.embedding_dim,
            window=5,
            min_count=1,
            epochs=5,
            sg=1  # Skip-gram
        )
        print(f"✓ Trained Word2Vec with {len(self.model.wv)} vocabulary")


# =============================================================================
# PART 2: SEMANTIC SEARCH ENGINE
# =============================================================================

class SemanticSearchEngine:
    """
    Complete semantic search system for retrieving similar documents.
    Uses embeddings and cosine similarity.
    """
    
    def __init__(self, embedding_dim: int = 100):
        """
        Initialize search engine.
        
        Args:
            embedding_dim: Dimensionality of embeddings
        """
        self.embedder = SimpleEmbedder(embedding_dim)
        self.documents = []
        self.embeddings = []
        self.metadata = {}  # Store document metadata (e.g., source, date)
    
    def index_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Index documents for searching.
        
        Args:
            documents: List of document strings
            metadata: Optional metadata for each document
        """
        self.documents = documents
        self.metadata = metadata or [{} for _ in documents]
        
        # Build embedder vocab and train
        self.embedder.build_vocab(documents)
        if self.embedder.use_gensim:
            self.embedder.train_word2vec()
        
        # Embed all documents
        print("Indexing documents...")
        self.embeddings = []
        for i, doc in enumerate(documents):
            emb = self.embedder.embed(doc)
            self.embeddings.append(emb)
            if (i + 1) % max(1, len(documents) // 10) == 0:
                print(f"  Indexed {i + 1}/{len(documents)} documents")
        
        self.embeddings = np.array(self.embeddings)
        print(f"✓ Indexed {len(documents)} documents")
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score [-1, 1]
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Search for top-k semantically similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (doc_index, document_text, similarity_score) tuples
        """
        query_emb = self.embedder.embed(query)
        
        # Compute similarities
        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = self.cosine_similarity(query_emb, doc_emb)
            similarities.append((i, self.documents[i], sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        return similarities[:top_k]
    
    def search_with_metadata(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search and return results with metadata.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with metadata
        """
        results = self.search(query, top_k)
        
        output = []
        for idx, doc, sim in results:
            output.append({
                'rank': len(output) + 1,
                'document': doc,
                'similarity': float(sim),
                'metadata': self.metadata[idx]
            })
        
        return output


# =============================================================================
# PART 3: NER-ENHANCED SEARCH (Entity Queries)
# =============================================================================

class EntityAwareSearchEngine(SemanticSearchEngine):
    """
    Extended search engine that extracts named entities from queries
    and documents for more precise retrieval.
    """
    
    def __init__(self, embedding_dim: int = 100):
        """Initialize entity-aware search engine."""
        super().__init__(embedding_dim)
        self.entity_index = defaultdict(list)  # entity -> [doc_indices]
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Simple entity extraction using heuristics.
        Real implementation would use Day 28 NER model.
        
        Args:
            text: Input text
            
        Returns:
            List of potential entities
        """
        # Heuristic: Capitalized words are potential entities
        tokens = text.split()
        entities = [t for t in tokens if t and t[0].isupper()]
        return entities
    
    def index_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Index documents and extract entities.
        
        Args:
            documents: List of document strings
            metadata: Optional metadata for each document
        """
        super().index_documents(documents, metadata)
        
        # Build entity index
        print("Extracting entities...")
        for i, doc in enumerate(documents):
            entities = self.extract_entities(doc)
            for entity in entities:
                self.entity_index[entity].append(i)
        
        print(f"✓ Extracted entities from {len(documents)} documents")
        print(f"  Total unique entities: {len(self.entity_index)}")
    
    def entity_search(self, entity: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Search for documents containing specific entity.
        
        Args:
            entity: Entity name
            top_k: Number of results to return
            
        Returns:
            List of (doc_index, document_text, score) tuples
        """
        if entity not in self.entity_index:
            return []
        
        doc_indices = self.entity_index[entity][:top_k]
        results = [(i, self.documents[i], 1.0) for i in doc_indices]
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Combine semantic and entity-based search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of hybrid search results
        """
        # Semantic search
        semantic = self.search(query, top_k)
        
        # Entity search
        entities = self.extract_entities(query)
        entity_results = []
        for entity in entities:
            entity_results.extend(self.entity_search(entity, top_k))
        
        # Deduplicate and combine
        seen = set()
        combined = []
        
        # Prioritize semantic results
        for idx, doc, sim in semantic:
            if idx not in seen:
                combined.append({
                    'rank': len(combined) + 1,
                    'document': doc,
                    'similarity': float(sim),
                    'search_type': 'semantic',
                    'metadata': self.metadata[idx]
                })
                seen.add(idx)
        
        # Add entity results
        for idx, doc, score in entity_results:
            if idx not in seen:
                combined.append({
                    'rank': len(combined) + 1,
                    'document': doc,
                    'similarity': float(score),
                    'search_type': 'entity',
                    'metadata': self.metadata[idx]
                })
                seen.add(idx)
        
        return combined[:top_k]


# =============================================================================
# PART 4: DEMO WITH BANSO BIBLE CORPUS
# =============================================================================

def create_sample_corpus() -> Tuple[List[str], List[Dict]]:
    """
    Create sample corpus of Bible verses (Genesis in English + Banso references).
    
    Returns:
        Tuple of (documents, metadata)
    """
    documents = [
        "In the beginning God created the heavens and the earth.",
        "Now the earth was formless and empty, darkness was over the surface of the deep.",
        "And the Spirit of God was hovering over the waters.",
        "And God said, Let there be light: and there was light.",
        "God saw that the light was good, and he separated the light from the darkness.",
        "And there was evening, and there was morning—the first day.",
        "And God said, Let there be a vault between the waters to separate water from water.",
        "So God made the vault and separated the water under the vault from the water above it.",
        "And God called the vault sky. And there was evening, and there was morning—the second day.",
        "And God said, Let the water under the sky be gathered to one place, and let dry ground appear.",
        "And God said, Let the land produce vegetation: seed-bearing plants and trees.",
        "The land produced vegetation: plants bearing seed according to their kinds and trees.",
        "God saw that it was good. And there was evening, and there was morning—the third day.",
        "And God said, Let there be lights in the vault of the sky to separate the day from the night.",
        "Let them serve as signs to mark sacred times, and days and years.",
        "And let them be lights in the vault of the sky to give light on the earth.",
        "God made two great lights—the greater light to govern the day and the lesser light.",
        "He also made the stars. God set them in the vault of the sky to give light on the earth.",
        "And God saw that it was good. And there was evening, and there was morning—the fourth day.",
    ]
    
    metadata = [
        {'chapter': 1, 'verse': 1, 'book': 'Genesis', 'banso_theme': 'nsi (creation)'},
        {'chapter': 1, 'verse': 2, 'book': 'Genesis', 'banso_theme': 'vu (void)'},
        {'chapter': 1, 'verse': 3, 'book': 'Genesis', 'banso_theme': 'bibor (spirit)'},
        {'chapter': 1, 'verse': 4, 'book': 'Genesis', 'banso_theme': 'ayaa (light)'},
        {'chapter': 1, 'verse': 5, 'book': 'Genesis', 'banso_theme': 'ayaa (light)'},
        {'chapter': 1, 'verse': 6, 'book': 'Genesis', 'banso_theme': 'nsi (creation)'},
        {'chapter': 1, 'verse': 7, 'book': 'Genesis', 'banso_theme': 'mbang (sky)'},
        {'chapter': 1, 'verse': 8, 'book': 'Genesis', 'banso_theme': 'mbang (sky)'},
        {'chapter': 1, 'verse': 9, 'book': 'Genesis', 'banso_theme': 'chi (water)'},
        {'chapter': 1, 'verse': 10, 'book': 'Genesis', 'banso_theme': 'chi (water)'},
        {'chapter': 1, 'verse': 11, 'book': 'Genesis', 'banso_theme': 'nsi (creation)'},
        {'chapter': 1, 'verse': 12, 'book': 'Genesis', 'banso_theme': 'nsi (creation)'},
        {'chapter': 1, 'verse': 13, 'book': 'Genesis', 'banso_theme': 'nsi (creation)'},
        {'chapter': 1, 'verse': 14, 'book': 'Genesis', 'banso_theme': 'ayaa (light)'},
        {'chapter': 1, 'verse': 15, 'book': 'Genesis', 'banso_theme': 'ayaa (light)'},
        {'chapter': 1, 'verse': 16, 'book': 'Genesis', 'banso_theme': 'ayaa (light)'},
        {'chapter': 1, 'verse': 17, 'book': 'Genesis', 'banso_theme': 'mbang (sky)'},
        {'chapter': 1, 'verse': 18, 'book': 'Genesis', 'banso_theme': 'mbang (sky)'},
        {'chapter': 1, 'verse': 19, 'book': 'Genesis', 'banso_theme': 'nsi (creation)'},
    ]
    
    return documents, metadata


def benchmark_search_engine():
    """
    Benchmark semantic search engine performance.
    """
    print("\n" + "="*70)
    print("SEMANTIC SEARCH ENGINE BENCHMARK")
    print("="*70)
    
    # Create corpus
    documents, metadata = create_sample_corpus()
    
    # Test 1: Basic semantic search
    print("\n[TEST 1] Basic Semantic Search")
    print("-" * 70)
    engine = SemanticSearchEngine(embedding_dim=100)
    engine.index_documents(documents, metadata)
    
    queries = [
        "light and darkness",
        "water and sky",
        "creation of vegetation",
        "heavens and earth",
    ]
    
    for query in queries:
        print(f"\nQuery: \"{query}\"")
        results = engine.search(query, top_k=3)
        for rank, (idx, doc, sim) in enumerate(results, 1):
            print(f"  {rank}. [{sim:.3f}] {doc[:60]}...")
    
    # Test 2: Entity-aware search
    print("\n[TEST 2] Entity-Aware Search")
    print("-" * 70)
    entity_engine = EntityAwareSearchEngine(embedding_dim=100)
    entity_engine.index_documents(documents, metadata)
    
    entity_queries = [
        "God and light",
        "sky and water",
        "land and plants",
    ]
    
    for query in entity_queries:
        print(f"\nQuery: \"{query}\"")
        results = entity_engine.hybrid_search(query, top_k=3)
        for result in results:
            print(f"  {result['rank']}. [{result['similarity']:.3f}] "
                  f"({result['search_type']}) {result['document'][:50]}...")
    
    # Test 3: Metadata retrieval
    print("\n[TEST 3] Metadata-Enhanced Search")
    print("-" * 70)
    query = "light creation"
    results = engine.search_with_metadata(query, top_k=3)
    for result in results:
        print(f"\n  Rank {result['rank']}: {result['similarity']:.3f}")
        print(f"    Text: {result['document'][:60]}...")
        print(f"    Location: {result['metadata']['book']} {result['metadata']['chapter']}:{result['metadata']['verse']}")
        print(f"    Theme: {result['metadata']['banso_theme']}")


def demonstrate_retrieval_metrics():
    """
    Demonstrate evaluation metrics for search engine.
    """
    print("\n" + "="*70)
    print("RETRIEVAL EVALUATION METRICS")
    print("="*70)
    
    documents, metadata = create_sample_corpus()
    engine = SemanticSearchEngine(embedding_dim=100)
    engine.index_documents(documents, metadata)
    
    # Simulated relevance judgments (query -> relevant doc indices)
    relevance_judgments = {
        "light and darkness": {0, 3, 4, 12, 13},
        "water and sky": {1, 8, 9},
        "creation": {0, 10, 11, 12, 13, 18},
    }
    
    print("\nQuery-Level Metrics:")
    print("-" * 70)
    
    all_precisions = []
    all_recalls = []
    
    for query, relevant in relevance_judgments.items():
        results = engine.search(query, top_k=5)
        retrieved_indices = {idx for idx, _, _ in results}
        
        # Precision: |relevant ∩ retrieved| / |retrieved|
        true_positives = len(relevant & retrieved_indices)
        precision = true_positives / len(retrieved_indices) if retrieved_indices else 0
        
        # Recall: |relevant ∩ retrieved| / |relevant|
        recall = true_positives / len(relevant) if relevant else 0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        
        print(f"\nQuery: \"{query}\"")
        print(f"  Precision@5: {precision:.3f}")
        print(f"  Recall@5: {recall:.3f}")
        print(f"  F1@5: {f1:.3f}")
    
    # Macro-averaged metrics
    print("\n" + "-"*70)
    print("Macro-Averaged Metrics:")
    print(f"  Mean Precision: {np.mean(all_precisions):.3f}")
    print(f"  Mean Recall: {np.mean(all_recalls):.3f}")


def banso_linguistic_analysis():
    """
    Analyze contextual word sense in Banso theological vocabulary.
    """
    print("\n" + "="*70)
    print("BANSO LINGUISTIC INTEGRATION")
    print("="*70)
    
    banso_terms = {
        'nsi': 'Creation, world, cosmos',
        'bibor': 'Spirit, wind, breath',
        'ayaa': 'Light, clarity, revelation',
        'chi': 'Water, flow, life force',
        'mbang': 'Sky, heaven, expanse',
        'kibor': 'Power, strength, dominion',
    }
    
    print("\nBanso Theological Terms:")
    for term, meaning in banso_terms.items():
        print(f"  • {term}: {meaning}")
    
    # Analyze how terms appear in corpus
    documents, metadata = create_sample_corpus()
    
    print("\nTerm Distribution in Corpus:")
    term_dist = defaultdict(int)
    for meta in metadata:
        if 'banso_theme' in meta:
            theme = meta['banso_theme'].split('(')[0].strip()
            term_dist[theme] += 1
    
    for term, count in sorted(term_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {term}: {count} occurrences")
    
    print("\nContextual Word Sense:")
    print("  'Light' appears in creation context (verse 4, 14-18)")
    print("  'Sky' appears in separation context (verse 7, 8, 14-16)")
    print("  'Water' appears in gathering context (verse 9, 10)")
    print("\n  → Banso terms contextualize these meanings within cosmology")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DAY 30: SEMANTIC SEARCH ENGINE - MODULE 05 REVIEW")
    print("="*70)
    
    # Run benchmarks
    benchmark_search_engine()
    
    # Evaluation metrics
    demonstrate_retrieval_metrics()
    
    # Banso linguistic analysis
    banso_linguistic_analysis()
    
    print("\n" + "="*70)
    print("✓ Day 30 exercises complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Semantic search combines embeddings with similarity metrics")
    print("  2. TF-IDF provides offline baseline; contextual embeddings improve quality")
    print("  3. Entity recognition enhances precision for named entity queries")
    print("  4. Hybrid retrieval (semantic + entity) balances recall and precision")
    print("  5. Banso corpus demonstrates cross-linguistic applicability")
    print("  6. Module 05 integrated: preprocessing → embeddings → sequences → NER → contextualization")
