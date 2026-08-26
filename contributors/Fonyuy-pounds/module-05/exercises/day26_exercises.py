"""
Day 26 Exercise: Word Embeddings & Word2Vec from Scratch (NumPy Edition)
Module 05: NLP Foundations
=======================================================================

Objective:
- Implement the Word2Vec Skip-gram architecture with Negative Sampling in pure NumPy from scratch.
- Integrate the Banso cultural term "nfor" (God) using parallel context and Banso proverbs.
- Train the model on a composite corpus of the KJV Bible + Banso linguistic inputs.
- Retrieve the nearest neighbors in the learned embedding space for key terms ("grace", "covenant", "shepherd", "nfor").
- Visualize the semantic vector spaces in 2D using t-SNE with scikit-learn.

Tasks:
1. Load and preprocess KJV Bible text, Banso proverbs, and synthetic cultural sentences.
2. Build vocabulary and map words to indices.
3. Construct a Negative Sampling Skip-gram dataset and noise distribution in NumPy.
4. Implement a custom Skip-gram Model with weight matrices and forward/backward updates in NumPy.
5. Train the model using Stochastic Gradient Descent (SGD) and evaluate word vector similarities.
6. Generate and save a 2D t-SNE visualization plot.
"""

import os
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Set random seed for reproducibility
np.random.seed(42)

# --- PART 1: Data Acquisition & Preprocessing ---

def load_data(workspace_root):
    """
    Load raw KJV Bible text and Banso proverbs, then blend them with 
    synthetic Banso-English context sentences for the deity word 'nfor'.
    """
    bible_path = os.path.join(workspace_root, 'data', 'raw', 'kjv_bible.txt')
    proverbs_path = os.path.join(workspace_root, 'data', 'banso-vernacular', 'proverbs.txt')
    
    corpus_parts = []
    
    # 1. Load Bible sample
    if os.path.exists(bible_path):
        with open(bible_path, 'r', encoding='utf-8') as f:
            # We take a 20KB chunk to ensure fast training and clear convergence on CPU
            bible_text = f.read(20000)
            corpus_parts.append(bible_text)
            print(f"Loaded {len(bible_text)} characters from KJV Bible.")
    else:
        print("KJV Bible not found, using baseline placeholder text.")
        corpus_parts.append("in the beginning god created the heaven and the earth")
        corpus_parts.append("the lord is my shepherd i shall not want")
        corpus_parts.append("for by grace are ye saved through faith")
        corpus_parts.append("this is the covenant that i will make with them after those days saith the lord")

    # 2. Load Banso Proverbs
    if os.path.exists(proverbs_path):
        with open(proverbs_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('lamnso_text') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) > 0:
                    corpus_parts.append(parts[0]) # Add Lamnso' vernacular sentence
                    if len(parts) > 1:
                        corpus_parts.append(parts[1]) # Add English translation
            print("Loaded Banso proverbs and combined them into the corpus.")
            
    # 3. Add custom parallel Banso-English sentences for 'nfor' to establish high-quality context
    banso_sentences = [
        "nfor is the creator and lord of all",
        "nfor has infinite grace and love",
        "nfor is our shepherd who guides us",
        "the covenant of nfor with his people",
        "nfor blesses the land and the people",
        "the wisdom of the elders comes from nfor",
        "nfor is god in the high heaven",
        "we worship nfor and seek his grace",
        "the shepherd feeds his flock under the covenant of nfor",
        "grace and mercy are given by nfor",
        "nfor is the king and ruler",
        "nfor is good and his covenant endures"
    ]
    corpus_parts.extend(banso_sentences)
    
    # Combine all parts
    full_text = " ".join(corpus_parts)
    return full_text

def preprocess_text(text):
    """Clean text by converting to lowercase, removing punctuation, and tokenizing."""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = [word for word in text.split() if word]
    return tokens

def build_vocabulary(tokens, min_count=1):
    """Build word to index mapping, keeping track of frequencies."""
    word_counts = Counter(tokens)
    
    # Explicitly ensure target words are in vocabulary
    target_words = ["grace", "covenant", "shepherd", "nfor", "god", "lord"]
    
    # Filter words based on min_count, but keep all target words
    filtered_words = [word for word, count in word_counts.items() if count >= min_count or word in target_words]
    
    word_to_ix = {word: i for i, word in enumerate(filtered_words)}
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    
    vocab_counts = {word: word_counts[word] for word in filtered_words}
    
    return word_to_ix, ix_to_word, vocab_counts

# --- PART 2: Skip-gram Pair Generation ---

def generate_skipgrams(tokens, word_to_ix, window_size=3):
    """Generate (target_word, context_word) index pairs within a sliding window."""
    pairs = []
    token_ids = [word_to_ix[t] for t in tokens if t in word_to_ix]
    
    for i, target_id in enumerate(token_ids):
        start = max(0, i - window_size)
        end = min(len(token_ids), i + window_size + 1)
        
        for j in range(start, end):
            if i == j:
                continue
            pairs.append((target_id, token_ids[j]))
            
    return pairs

# --- PART 3: Skip-gram NumPy Trainer ---

def sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

class NumPyWord2Vec:
    """Skip-gram model trained with SGD and Negative Sampling using pure NumPy."""
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize target (W_in) and context (W_out) embedding matrices
        initrange = 0.5 / embedding_dim
        self.W_in = np.random.uniform(-initrange, initrange, (vocab_size, embedding_dim))
        self.W_out = np.zeros((vocab_size, embedding_dim))
        
    def train_step(self, target, context, negatives, lr):
        """
        Perform a single training step on a single target word, context word, 
        and its sampled negative context indices.
        
        Returns:
            loss (float): Combined loss for this instance.
        """
        # Look up target vector
        v_t = self.W_in[target] # Shape: (embedding_dim,)
        
        # Positive context vector
        v_c = self.W_out[context] # Shape: (embedding_dim,)
        
        # Positive score
        pos_score = np.dot(v_t, v_c)
        pos_sig = sigmoid(pos_score)
        
        # Positive loss contribution
        loss = -np.log(pos_sig + 1e-10)
        
        # Negative vectors
        v_neg = self.W_out[negatives] # Shape: (num_negatives, embedding_dim)
        
        # Negative scores
        neg_scores = np.dot(v_neg, v_t) # Shape: (num_negatives,)
        neg_sigs = sigmoid(-neg_scores) # Shape: (num_negatives,)
        
        # Negative loss contribution
        loss += -np.sum(np.log(neg_sigs + 1e-10))
        
        # Backpropagation: Gradients
        # Context positive gradient: dL/dv_c = (sigmoid(v_t^T v_c) - 1) * v_t
        grad_v_c = (pos_sig - 1) * v_t
        
        # Context negative gradients: dL/dv_n = sigmoid(v_t^T v_n) * v_t
        # neg_sigs_pos = sigmoid(v_t^T v_n) = 1 - neg_sigs
        neg_sigs_pos = 1.0 - neg_sigs
        grad_v_neg = neg_sigs_pos.reshape(-1, 1) * v_t.reshape(1, -1) # Shape: (num_negatives, embedding_dim)
        
        # Target gradient: dL/dv_t = (sigmoid(v_t^T v_c) - 1)*v_c + sum(sigmoid(v_t^T v_n)*v_n)
        grad_v_t = (pos_sig - 1) * v_c + np.sum(neg_sigs_pos.reshape(-1, 1) * v_neg, axis=0)
        
        # Weight updates
        self.W_out[context] -= lr * grad_v_c
        self.W_out[negatives] -= lr * grad_v_neg
        self.W_in[target] -= lr * grad_v_t
        
        return loss

# --- PART 4: Evaluation and Visualization ---

def find_nearest_neighbors(word, model, word_to_ix, ix_to_word, top_k=5):
    """Calculate the top-k nearest neighbors of a word using Cosine Similarity."""
    if word not in word_to_ix:
        return [("NOT_IN_VOCAB", 0.0)]
    
    target_idx = word_to_ix[word]
    v_t = model.W_in[target_idx]
    
    # Normalize vectors
    norm_target = v_t / (np.linalg.norm(v_t) + 1e-10)
    
    norms = np.linalg.norm(model.W_in, axis=1, keepdims=True)
    norm_all = model.W_in / (norms + 1e-10)
    
    # Calculate cosine similarity
    similarities = np.dot(norm_all, norm_target)
    
    # Sort similarities
    top_indices = np.argsort(similarities)[::-1]
    
    results = []
    count = 0
    for idx in top_indices:
        word_str = ix_to_word[idx]
        if word_str == word:
            continue
        results.append((word_str, similarities[idx]))
        count += 1
        if count >= top_k:
            break
            
    return results

def visualize_embeddings(model, word_to_ix, words_to_plot, save_path):
    """Project the word embeddings into 2D using t-SNE and save a professional graph."""
    indices = []
    valid_words = []
    for w in words_to_plot:
        if w in word_to_ix:
            indices.append(word_to_ix[w])
            valid_words.append(w)
            
    if not indices:
        print("No valid words for t-SNE visualization.")
        return
        
    embeddings = model.W_in[indices]
    
    # Run t-SNE
    perplexity = min(5, len(valid_words) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embeddings)
    
    # Set up styling
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid(True, linestyle='--', alpha=0.5)
    
    target_words = ["grace", "covenant", "shepherd", "nfor", "god", "lord"]
    
    for i, word in enumerate(valid_words):
        x, y = coords[i, 0], coords[i, 1]
        
        if word in target_words:
            # Highlight targets
            ax.scatter(x, y, color='#dc2626', edgecolors='black', s=200, zorder=5)
            ax.text(x + 0.1, y + 0.1, word.upper(), fontsize=13, fontweight='bold', color='#991b1b', zorder=6)
        else:
            # Context words
            ax.scatter(x, y, color='#0d9488', edgecolors='black', s=100, zorder=3)
            ax.text(x + 0.1, y + 0.1, word, fontsize=10, color='#374151', alpha=0.9, zorder=4)
            
    # Title & styling
    ax.set_title("Word Embeddings t-SNE 2D Projection (MarkGPT Day 26 - NumPy)", fontsize=16, fontweight='bold', pad=20, color='#1e293b')
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12, color='#475569')
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12, color='#475569')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"t-SNE visualization successfully saved to {save_path}")

# --- PART 5: Main Orchestrator ---

def main():
    print("=" * 60)
    print("MARK GPT — DAY 26: Word Embeddings NumPy Pipeline Initiated")
    print("=" * 60)
    
    # Set workspace path
    workspace_root = r'c:\Users\the eye informatique\Desktop\ML\AI\MarkGPT'
    
    # 1. Load raw resources
    raw_text = load_data(workspace_root)
    
    # 2. Tokenize and clean
    tokens = preprocess_text(raw_text)
    print(f"Total tokens in raw cleaned corpus: {len(tokens)}")
    
    # 3. Build Vocabulary
    word_to_ix, ix_to_word, vocab_counts = build_vocabulary(tokens, min_count=2)
    vocab_size = len(word_to_ix)
    print(f"Vocabulary Size: {vocab_size}")
    
    # 4. Draw Skip-grams
    window_size = 3
    pairs = generate_skipgrams(tokens, word_to_ix, window_size)
    print(f"Total Skip-gram Pairs generated: {len(pairs)}")
    
    # 5. Define Noise Distribution for Negative Sampling
    word_freqs = np.array([vocab_counts[ix_to_word[i]] for i in range(vocab_size)])
    noise_dist = word_freqs ** 0.75
    noise_dist /= noise_dist.sum()
    
    # 6. Instantiate Model
    embedding_dim = 64
    num_negatives = 5
    model = NumPyWord2Vec(vocab_size, embedding_dim)
    
    # 7. Training loop using pure NumPy SGD
    epochs = 8
    initial_lr = 0.025
    print(f"\nTraining Word2Vec model for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Linear learning rate decay
        lr = initial_lr * (1.0 - (epoch / epochs))
        epoch_loss = 0.0
        
        # Shuffle pairs for SGD
        np.random.shuffle(pairs)
        
        # Pre-sample all negative context indices for this epoch in a single vectorized call
        all_negatives = np.random.choice(vocab_size, size=(len(pairs), num_negatives), p=noise_dist)
        
        for step, (target, context) in enumerate(pairs):
            negatives = all_negatives[step]
            loss = model.train_step(target, context, negatives, lr)
            epoch_loss += loss
            
        avg_loss = epoch_loss / len(pairs)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Avg SGD Loss: {avg_loss:.4f} | LR: {lr:.4f}")
            
    print("Training finished!\n" + "-" * 40)
    
    # 8. Find Nearest Neighbors
    queries = ["grace", "covenant", "shepherd", "nfor", "god", "lord"]
    
    # Write a quick report
    report_lines = [
        "# Day 26 Similarity Report",
        f"Corpus size: {len(tokens)} tokens",
        f"Vocab size: {vocab_size} words",
        f"Embedding Dimension: {embedding_dim}",
        "\n## Word Nearest Neighbors (Cosine Similarity)\n"
    ]
    
    for query in queries:
        print(f"\nNearest neighbors for '{query}':")
        neighbors = find_nearest_neighbors(query, model, word_to_ix, ix_to_word, top_k=5)
        
        report_lines.append(f"### {query.upper()}")
        for neighbor, score in neighbors:
            line_str = f"- **{neighbor}**: {score:.4f}"
            print(f"  {line_str}")
            report_lines.append(line_str)
        report_lines.append("")
        
    # Write report file
    report_path = os.path.join(workspace_root, 'contributors', 'Fonyuy-pounds', 'module-05', 'exercises', 'similarity_report.md')
    with open(report_path, 'w', encoding='utf-8') as rf:
        rf.write("\n".join(report_lines))
    print(f"\nSimilarity report successfully written to {report_path}")
    
    # 9. Visualize with t-SNE
    words_to_plot = [
        "grace", "covenant", "shepherd", "nfor", "god", "lord",
        "jesus", "moses", "david", "faith", "spirit", "king", 
        "heaven", "earth", "wisdom", "elders", "proverb", "community", 
        "banso", "blessing", "truth", "heart", "commandment", "law", 
        "peace", "love"
    ]
    
    save_img_path = os.path.join(workspace_root, 'contributors', 'Fonyuy-pounds', 'module-05', 'exercises', 'word_embeddings_tsne.png')
    visualize_embeddings(model, word_to_ix, words_to_plot, save_img_path)
    
    print("\nDAY 26 Pipeline Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
