"""
Day 04 Exercise: Bigram Language Model Solution
================================================
Module 01 | Exercise E04.2 & E04.3

Author: Fonyuy-pounds
Date: 2026-04-14
"""

import random
import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(text: str) -> List[str]:
    """
    Convert raw Bible text into a clean list of lowercase word tokens.
    """
    tokens = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Step 1: Convert to lowercase
        line = line.lower()
        
        # Step 2: Remove punctuation
        line = re.sub(r'[^\w\s]', '', line)
        
        # Step 3: Split on whitespace
        words = line.split()
        
        # Step 4: Filter out empty strings
        words = [w for w in words if w]
        
        # Step 5: Add tokens and <EOS>
        tokens.extend(words)
        tokens.append('<EOS>')
    
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: BUILD THE BIGRAM MODEL
# ─────────────────────────────────────────────────────────────────────────────

class BigramLanguageModel:
    """
    A bigram language model that captures P(word_i | word_{i-1}).
    """
    
    def __init__(self, smoothing: float = 0.0):
        self.counts = defaultdict(Counter)
        self.probs = {}
        self.smoothing = smoothing
        self.vocabulary = set()
    
    def train(self, tokens: List[str]) -> None:
        """
        Count all bigrams and build the probability table.
        """
        self.vocabulary = set(tokens)
        V = len(self.vocabulary)
        
        # Step 2: Count all consecutive pairs
        for i in range(len(tokens) - 1):
            word_a = tokens[i]
            word_b = tokens[i+1]
            self.counts[word_a][word_b] += 1
        
        # Step 3: Normalize counts to probabilities
        for word_a, follow_counts in self.counts.items():
            total = sum(follow_counts.values())
            self.probs[word_a] = {}
            for word_b, count in follow_counts.items():
                # Apply Laplace smoothing
                # P(B | A) = (count(A,B) + alpha) / (count(A) + alpha * V)
                self.probs[word_a][word_b] = (count + self.smoothing) / (total + self.smoothing * V)
        
        print(f"Model trained on {len(tokens)} tokens.")
        print(f"Vocabulary size: {V} unique tokens.")
    
    def probability(self, word_b: str, given_word_a: str) -> float:
        """Return P(word_b | given_word_a)."""
        if given_word_a not in self.probs:
            return 1e-10
        
        if word_b not in self.probs[given_word_a]:
            if self.smoothing > 0:
                total = sum(self.counts[given_word_a].values())
                V = len(self.vocabulary)
                return self.smoothing / (total + self.smoothing * V)
            return 1e-10
        
        return self.probs[given_word_a][word_b]
    
    def generate(self, seed: str, max_tokens: int = 50) -> str:
        """Generate text by sampling from the bigram distribution."""
        current_word = seed.lower()
        output_tokens = [current_word]
        
        for _ in range(max_tokens):
            if current_word not in self.probs:
                break
            
            next_word_probs = self.probs[current_word]
            words = list(next_word_probs.keys())
            weights = list(next_word_probs.values())
            
            next_word = random.choices(words, weights=weights, k=1)[0]
            
            if next_word == '<EOS>':
                break
            
            output_tokens.append(next_word)
            current_word = next_word
        
        return ' '.join(output_tokens)
    
    def perplexity(self, tokens: List[str]) -> float:
        """Compute the perplexity of this model on a sequence of tokens."""
        log_prob_sum = 0.0
        N = 0
        
        for i in range(len(tokens) - 1):
            word_a = tokens[i]
            word_b = tokens[i + 1]
            
            p = self.probability(word_b, given_word_a=word_a)
            log_prob_sum += math.log(p)
            N += 1
        
        if N == 0:
            return float('inf')
        
        avg_neg_log_prob = -log_prob_sum / N
        return math.exp(avg_neg_log_prob)


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: RUN THE EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sample_genesis = """
    In the beginning God created the heaven and the earth.
    And the earth was without form and void and darkness was upon the face of the deep.
    And the Spirit of God moved upon the face of the waters.
    And God said Let there be light and there was light.
    And God saw the light that it was good and God divided the light from the darkness.
    And God called the light Day and the darkness he called Night.
    And the evening and the morning were the first day.
    And God said Let there be a firmament in the midst of the waters.
    And God made the firmament and divided the waters which were under the firmament.
    And God called the firmament Heaven and the evening and the morning were the second day.
    """
    
    # ── Preprocess ──
    print("Preprocessing text...")
    tokens = load_and_preprocess(sample_genesis)
    
    # ── Split into train/test ──
    split = int(0.9 * len(tokens))
    train_tokens = tokens[:split]
    test_tokens = tokens[split:]
    
    # ── Train ──
    print("Training bigram model (no smoothing)...")
    model_no_smooth = BigramLanguageModel(smoothing=0.0)
    model_no_smooth.train(train_tokens)
    
    print("\nTraining bigram model (Laplace smoothing, alpha=1.0)...")
    model_smooth = BigramLanguageModel(smoothing=1.0)
    model_smooth.train(train_tokens)
    
    # ── Generate ──
    random.seed(42)
    print("\n" + "=" * 60)
    print("GENERATED TEXT (no smoothing, seed='god'):")
    print(f"  {model_no_smooth.generate('god', max_tokens=20)}")
    
    print("\nGENERATED TEXT (Laplace smoothing, seed='god'):")
    print(f"  {model_smooth.generate('god', max_tokens=20)}")
    
    # ── Evaluate ──
    print("\n" + "=" * 60)
    ppl_no_smooth = model_no_smooth.perplexity(test_tokens)
    ppl_smooth = model_smooth.perplexity(test_tokens)
    
    print(f"PERPLEXITY (no smoothing):    {ppl_no_smooth:.1f}")
    print(f"PERPLEXITY (Laplace smooth):  {ppl_smooth:.1f}")


if __name__ == "__main__":
    main()
