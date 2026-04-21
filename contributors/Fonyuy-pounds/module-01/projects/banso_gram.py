import random
import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import os

def preprocess_text(text: str, is_tsv: bool = False) -> List[str]:
    tokens = []
    lines = text.strip().split('\n')
    
    if is_tsv:
        # Skip header if it exists
        if lines and lines[0].startswith('lamnso'):
            lines = lines[1:]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if is_tsv:
            parts = line.split('\t')
            if not parts:
                continue
            line = parts[0]  # First column is Lamnso
            
        line = line.lower()
        # Remove punctuation except apostrophes (for Banso)
        line = re.sub(r'[^\w\s\']', '', line)
        words = line.split()
        words = [w for w in words if w]
        if words:
            tokens.extend(words)
            tokens.append('<EOS>')
            
    return tokens

class BansoGram:
    def __init__(self, n: int = 2, smoothing: float = 1.0):
        self.n = n
        self.smoothing = smoothing
        self.counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocabulary = set()
        self.V = 0
        
    def train(self, tokens: List[str]):
        self.vocabulary = set(tokens)
        self.V = len(self.vocabulary)
        
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i + self.n - 1])
            next_word = tokens[i + self.n - 1]
            self.counts[context][next_word] += 1
            self.context_counts[context] += 1

    def get_probability(self, word: str, context: tuple) -> float:
        count = self.counts.get(context, {}).get(word, 0)
        context_count = self.context_counts.get(context, 0)
        
        # Laplace smoothing
        return (count + self.smoothing) / (context_count + self.smoothing * self.V)

    def generate(self, seed: List[str], max_len: int = 30) -> str:
        if len(seed) != self.n - 1:
            raise ValueError(f"Seed must be of length {self.n - 1}")
            
        current_context = tuple(seed)
        output = list(seed)
        
        for _ in range(max_len):
            if current_context not in self.counts and self.smoothing == 0:
                break
                
            vocab_list = list(self.vocabulary)
            weights = [self.get_probability(w, current_context) for w in vocab_list]
                
            next_word = random.choices(vocab_list, weights=weights, k=1)[0]
            
            if next_word == '<EOS>':
                break
                
            output.append(next_word)
            current_context = tuple(list(current_context)[1:] + [next_word])
            
        return ' '.join(output)

    def calculate_perplexity(self, tokens: List[str]) -> float:
        log_prob_sum = 0.0
        N = 0
        
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i + self.n - 1])
            word = tokens[i + self.n - 1]
            p = self.get_probability(word, context)
            if p == 0:
                return float('inf')
            log_prob_sum += math.log(p)
            N += 1
            
        if N == 0: 
            return float('inf')
        return math.exp(-log_prob_sum / N)

def main():
    print("Welcome to BansoGram!")
    print("-" * 20)
    
    # Locate data correctly relative to contributor dir or project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    banso_path = os.path.join(base_dir, "data", "banso-vernacular", "proverbs.txt")
    
    sample_genesis = '''
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
    '''
    
    banso_raw = ""
    try:
        with open(banso_path, 'r', encoding='utf-8') as f:
            banso_raw = f.read()
    except Exception as e:
        print(f"Couldn't load Banso data from {banso_path}: {e}")
        
    english_tokens = preprocess_text(sample_genesis)
    banso_tokens = preprocess_text(banso_raw, is_tsv=True)
    
    english_split = int(len(english_tokens) * 0.8)
    e_train, e_test = english_tokens[:english_split], english_tokens[english_split:]
    
    banso_split = int(len(banso_tokens) * 0.8)
    b_train, b_test = banso_tokens[:banso_split], banso_tokens[banso_split:]
    
    print("Training Trigram Models...")
    english_model_tri = BansoGram(n=3, smoothing=1.0)
    english_model_tri.train(e_train)
    
    banso_model_tri = BansoGram(n=3, smoothing=1.0)
    banso_model_tri.train(b_train)
    
    e_ppl = english_model_tri.calculate_perplexity(e_test)
    b_ppl = banso_model_tri.calculate_perplexity(b_test)
    
    print(f"English Trigram Model Perplexity (held-out): {e_ppl:.2f}")
    print(f"Banso Trigram Model Perplexity (held-out): {b_ppl:.2f}")
    
    print("\nGenerating text (Trigram)")
    if len(e_train) >= 2:
        seed_e = e_train[:2]
        print(f"English (seed: {seed_e}):", english_model_tri.generate(seed_e))
        
    if len(b_train) >= 2:
        seed_b = b_train[:2]
        print(f"Banso (seed: {seed_b}):", banso_model_tri.generate(seed_b))

if __name__ == '__main__':
    main()
