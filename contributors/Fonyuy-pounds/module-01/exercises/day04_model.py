import random
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

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
