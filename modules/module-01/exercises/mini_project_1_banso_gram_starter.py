"""
Mini-Project 1: BansoGram
=========================
Module 01: Foundations | Day 06

Goal: Build a Bigram and Trigram language model from scratch to compare 
English (KJV Bible) and Banso (Lamnso' proverbs/Bible).

Instructions:
1. Complete the NgramModel class.
2. Implement Laplace Smoothing (Add-Alpha).
3. Train two models: one on English, one on Banso.
4. Generate text from both and compare the results.
5. Calculate and report perplexity for both.

Author: [Your Name]
Date: 2024
"""

import random
import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 🛠️ DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> List[str]:
    """
    Clean text and convert to a list of tokens.
    - Lowercase
    - Remove punctuation (except <EOS>)
    - Add <EOS> at the end of each logical line
    """
    tokens = []
    # Your code here: split text into sentences/lines, clean, and tokenize
    # Tip: Use re.sub(r'[^\w\s]', '', line) to remove punctuation
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# 🏗️ THE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class BansoGram:
    """
    A generic N-Gram Language Model.
    Can be configured as a Unigram (N=1), Bigram (N=2), or Trigram (N=3).
    """
    
    def __init__(self, n: int = 2, smoothing: float = 1.0):
        self.n = n
        self.smoothing = smoothing
        self.counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocabulary = set()
        
    def train(self, tokens: List[str]):
        """
        Train the model on a list of tokens.
        Build the counts for (context) -> next_word.
        """
        self.vocabulary = set(tokens)
        
        # Your code here: 
        # 1. Iterate through tokens
        # 2. Extract N-sized windows
        # 3. Update self.counts[context][next_word] and self.context_counts[context]
        pass

    def get_probability(self, word: str, context: tuple) -> float:
        """
        Calculate P(word | context) with Laplace smoothing.
        """
        # Your code here:
        # P = (count(context, word) + alpha) / (count(context) + alpha * |V|)
        return 0.0

    def generate(self, seed: List[str], max_len: int = 30) -> str:
        """
        Generate text starting from the seed phrase.
        """
        # Your code here:
        # 1. Start with the seed
        # 2. Use random.choices or similar to sample the next word
        # 3. Stop if <EOS> is reached or max_len is hit
        return "Generated text..."

    def calculate_perplexity(self, tokens: List[str]) -> float:
        """
        Evaluate the model on a test set (held-out data).
        """
        # Your code here:
        # Perplexity = exp( -1/N * sum(log(P(w|context))) )
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 🚀 MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # PATHS (Adjust these paths if necessary)
    BANSO_DATA_PATH = "data/banso-vernacular/proverbs.txt"
    ENGLISH_DATA_PATH = "data/raw/kjv_sample.txt" # Or use a string literal for now

    # 1. Load Data
    # 2. Preprocess
    # 3. Train models (Banso vs English)
    # 4. Generate & Compare
    # 5. Report Metrics
    
    print("Welcome to BansoGram!")
    print("-" * 20)

if __name__ == "__main__":
    main()
