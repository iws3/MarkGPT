"""
Day 05 Exercise: Trigram Language Model Solution
================================================
Module 01 | Exercise E05.1 & E05.2

Author: Fonyuy-pounds
Date: 2026-04-16
"""

import random
import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(text: str) -> List[str]:
    """
    Convert raw Bible text into a clean list of lowercase word tokens.
    Adds <EOS> at the end of each line to mark sequence boundaries.
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
# PART 2: BUILD THE TRIGRAM MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TrigramLanguageModel:
    """
    A trigram language model that captures P(word_i | word_{i-2}, word_{i-1}).
    """
    
    def __init__(self, smoothing: float = 0.0):
        # counts[ (word_n-2, word_n-1) ][ word_n ]
        self.counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.context_counts: Counter[Tuple[str, str]] = Counter()
        self.smoothing = smoothing
        self.vocabulary: set[str] = set()
    
    def train(self, tokens: List[str]) -> None:
        """
        Count all trigrams and build the transition counts.
        """
        self.vocabulary = set(tokens)
        
        # We need at least 3 tokens to form a trigram
        if len(tokens) < 3:
            print("Warning: Not enough tokens to train a trigram model.")
            return

        # Slide over the tokens to count trigrams
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            context = (w1, w2)
            self.counts[context][w3] += 1
            self.context_counts[context] += 1
            
        print(f"Model trained on {len(tokens)} tokens.")
        print(f"Vocabulary size: {len(self.vocabulary)} unique tokens.")
        print(f"Unique contexts (bigrams): {len(self.context_counts)}")

    def get_probability(self, word: str, context: Tuple[str, str]) -> float:
        """
        Return the probability P(word | context) with optional Laplace smoothing.
        P(w3 | w1, w2) = (count(w1, w2, w3) + alpha) / (count(w1, w2) + alpha * V)
        """
        V = len(self.vocabulary)
        count_w3 = self.counts[context][word]
        count_context = self.context_counts[context]
        
        # Apply Laplace Smoothing
        # If smoothing is 0 and count_context is 0, we return a tiny epsilon
        if self.smoothing == 0:
            if count_context == 0:
                return 1e-12
            return count_w3 / count_context
        else:
            return (count_w3 + self.smoothing) / (count_context + self.smoothing * V)

    def generate(self, seed_phrase: List[str], max_tokens: int = 50) -> str:
        """
        Generate text by sampling from the trigram distribution.
        """
        if len(seed_phrase) < 2:
            return "Error: Seed phrase must contain at least 2 words for a trigram model."
        
        output_tokens = list(seed_phrase)
        current_context = (seed_phrase[-2].lower(), seed_phrase[-1].lower())
        
        for _ in range(max_tokens):
            if current_context not in self.counts and self.smoothing == 0:
                break
            
            # Get possible next words and their probabilities
            if self.smoothing > 0:
                # With smoothing, any word in vocabulary is possible
                choices = list(self.vocabulary)
                weights = [self.get_probability(w, current_context) for w in choices]
            else:
                # Without smoothing, only previously seen follow-ups
                follow_counts = self.counts[current_context]
                choices = list(follow_counts.keys())
                total = sum(follow_counts.values())
                weights = [c / total for c in follow_counts.values()]
            
            if not choices:
                break
                
            next_word = random.choices(choices, weights=weights, k=1)[0]
            
            if next_word == '<EOS>':
                break
                
            output_tokens.append(next_word)
            current_context = (current_context[1], next_word)
            
        return ' '.join(output_tokens)

    def calculate_perplexity(self, tokens: List[str]) -> float:
        """
        Compute the perplexity of the model on a sequence of tokens.
        Perplexity = exp( -1/N * sum(log(P(w_i | w_{i-2}, w_{i-1}))) )
        """
        if len(tokens) < 3:
            return 0.0
            
        log_prob_sum = 0.0
        n = 0
        
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            prob = self.get_probability(w3, (w1, w2))
            
            # Ensure prob is at least a tiny epsilon to avoid log(0)
            prob = max(prob, 1e-12)
            
            log_prob_sum += math.log(prob)
            n += 1
            
        avg_neg_log_prob = -log_prob_sum / n
        return math.exp(avg_neg_log_prob)


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: RUN THE EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load Book of Genesis (Sample from Chapters 1-2)
    genesis_text = """
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
    And God said Let the waters under the heaven be gathered together unto one place and let the dry land appear and it was so.
    And God called the dry land Earth and the gathering together of the waters called he Seas and God saw that it was good.
    And God said Let the earth bring forth grass the herb yielding seed and the fruit tree yielding fruit after his kind whose seed is in itself upon the earth and it was so.
    And the earth brought forth grass and herb yielding seed after his kind and the tree yielding fruit whose seed was in itself after his kind and God saw that it was good.
    And the evening and the morning were the third day.
    """
    
    # Preprocess
    print("Pre-processing Genesis text...")
    tokens = load_and_preprocess(genesis_text)
    
    # Train-test split (90% train, 10% test)
    split_idx = int(0.9 * len(tokens))
    train_tokens = tokens[:split_idx]
    test_tokens = tokens[split_idx:]
    
    # 1. Model without smoothing
    print("\n[Trigram Model: No Smoothing]")
    model_raw = TrigramLanguageModel(smoothing=0.0)
    model_raw.train(train_tokens)
    
    # 2. Model with Laplace smoothing
    print("\n[Trigram Model: Laplace Smoothing (alpha=1.0)]")
    model_smooth = TrigramLanguageModel(smoothing=1.0)
    model_smooth.train(train_tokens)
    
    # Generate Text
    seed = ["and", "god"]
    print("\n" + "="*50)
    print(f"GENERATION (Seed: {' '.join(seed)})")
    print("-" * 50)
    print(f"Raw:    {model_raw.generate(seed, max_tokens=30)}")
    print(f"Smooth: {model_smooth.generate(seed, max_tokens=30)}")
    
    # Perplexity
    print("\n" + "="*50)
    print("EVALUATION (Perplexity)")
    print("-" * 50)
    ppl_raw = model_raw.calculate_perplexity(test_tokens)
    ppl_smooth = model_smooth.calculate_perplexity(test_tokens)
    
    print(f"Perplexity (No Smoothing): {ppl_raw:,.2f}")
    print(f"Perplexity (Smoothing):    {ppl_smooth:,.2f}")
    
    print("\nNote: Perplexity is lower (better) with smoothing on test data because it handles unseen trigrams gracefully.")


if __name__ == "__main__":
    # For reproducible results
    random.seed(42)
    main()
