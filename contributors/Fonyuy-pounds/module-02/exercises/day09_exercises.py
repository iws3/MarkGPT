import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import os
import csv

def pca_from_scratch(X, n_components=2):
    """
    Perform PCA from scratch on data X.
    """
    # 1. Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # 2. Compute the Covariance Matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # 3. Compute Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort eigenvalues and keep the top n_components
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    top_components = eigenvectors[:, :n_components]
    
    # 5. Project the data
    X_projected = np.dot(X_centered, top_components)
    
    return X_projected, top_components

def build_cooccurrence_matrix(corpus, window_size=2):
    """
    Build a co-occurrence matrix from a corpus.
    """
    vocab = sorted(list(set([word for sentence in corpus for word in sentence])))
    word2idx = {word: i for i, word in enumerate(vocab)}
    matrix = np.zeros((len(vocab), len(vocab)))
    
    for sentence in corpus:
        for i, word in enumerate(sentence):
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    matrix[word2idx[word], word2idx[sentence[j]]] += 1
    
    return matrix, vocab

def clean_text(text):
    text = text.lower()
    # Support basic Latin and common Banso chars (simplified)
    text = re.sub(r'[^a-z\'\s]', '', text) 
    return text.split()

def main():
    print("Welcome to Day 9: Linear Algebra Deep Dive (Contributed by Fonyuy-pounds)")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
    
    print(f"Project root identified as: {project_root}")
    
    kjv_path = os.path.join(project_root, "data", "raw", "kjv_sample.csv")
    banso_path = os.path.join(project_root, "data", "banso-vernacular", "proverbs.txt")
    
    corpus = []
    
    # Load KJV Sample manually
    if os.path.exists(kjv_path):
        print("Loading KJV data...")
        try:
            with open(kjv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    corpus.append(clean_text(row['text']))
            print(f"Loaded KJV Bible verses.")
        except Exception as e:
            print(f"Error loading KJV: {e}")

    # Load Banso Proverbs manually
    if os.path.exists(banso_path):
        print("Loading Banso data...")
        try:
            with open(banso_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    if 'english_translation' in row:
                        corpus.append(clean_text(row['english_translation']))
                    if 'lamnso_text' in row:
                        corpus.append(clean_text(row['lamnso_text']))
            print(f"Loaded Banso proverbs.")
        except Exception as e:
            print(f"Error loading Banso: {e}")

    if not corpus:
        print("Error: Corpus is empty. Generating toy data fallback.")
        corpus = [clean_text("God created the heaven and the earth"), 
                  clean_text("The Spirit of God moved upon the face of the waters")]

    # Build matrix
    print("Building co-occurrence matrix...")
    matrix, vocab = build_cooccurrence_matrix(corpus, window_size=3)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Run PCA
    if len(vocab) > 2:
        print("Running PCA from scratch...")
        X_projected, components = pca_from_scratch(matrix, n_components=2)
        
        # Visualization
        plt.figure(figsize=(12, 8))
        plt.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.5, c='darkblue')
        
        interest_words = ['god', 'earth', 'light', 'water', 'spirit', 'wisdom', 'hand', 'life', 'lord', 'word', 'world']
        
        for i, word in enumerate(vocab):
            if word in interest_words or (len(vocab) < 50) or i % (len(vocab)//20 + 1) == 0:
                plt.annotate(word, (X_projected[i, 0], X_projected[i, 1]), fontsize=9, alpha=0.7)
                
        plt.title("Word Co-occurrence PCA Projection (MarkGPT Day 9)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True, linestyle='--', alpha=0.3)
        
        output_plot = 'word_clusters_pca.png'
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        print(f"Saved {output_plot}")
    else:
        print("Not enough words in vocabulary for PCA.")

if __name__ == "__main__":
    main()
