import os
import re
import random
from collections import Counter

class TextPipeline:
    def __init__(self, vocab_size=None):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = vocab_size
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        
    def clean_text(self, text):
        """Lowercase and remove basic punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
        
    def tokenize(self, text):
        """Simple whitespace tokenizer after cleaning."""
        return self.clean_text(text).split()
        
    def build_vocab(self, texts):
        """Build vocabulary from a list of texts."""
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))
            
        counter = Counter(all_tokens)
        
        # Start with special tokens
        self.vocab = {self.pad_token: 0, self.unk_token: 1}
        idx = 2
        
        # Add most common words up to vocab_size
        most_common = counter.most_common(self.vocab_size - 2 if self.vocab_size else None)
        for word, _ in most_common:
            self.vocab[word] = idx
            idx += 1
            
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Vocabulary built with {len(self.vocab)} unique tokens.")
        
    def encode(self, text):
        """Convert text to list of token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        
    def decode(self, token_ids):
        """Convert list of token IDs back to text."""
        return " ".join([self.inverse_vocab.get(tid, self.unk_token) for tid in token_ids])

def main():
    print("=== Mini-Project 2: Text Preprocessing Pipeline ===")
    
    # 1. Load Data (Simulated for this exercise based on previous days)
    # Ideally, this would load from data/raw/kjv_sample.csv and data/banso-vernacular/proverbs.txt
    bible_data = [
        "In the beginning God created the heaven and the earth.",
        "And the earth was without form and void.",
        "And God said Let there be light: and there was light.",
        "The LORD is my shepherd; I shall not want.",
        "He maketh me to lie down in green pastures."
    ]
    
    banso_data = [
        "Kpu yi wo mntar nyuy.", # Death is not the end
        "Nyuy ka le yee fonyuy.", # God is good
        "A dzə bika a dzə baa.", # If it's not today, it's tomorrow
    ]
    
    all_data = bible_data + banso_data
    
    # 2. Initialize and train the pipeline
    pipeline = TextPipeline(vocab_size=1000)
    pipeline.build_vocab(all_data)
    
    # 3. Create datasets (Train/Val/Test splits)
    # Shuffle data
    random.seed(42)
    random.shuffle(all_data)
    
    # Split: 60% Train, 20% Val, 20% Test
    n = len(all_data)
    train_split = int(0.6 * n)
    val_split = int(0.8 * n)
    
    train_data = all_data[:train_split]
    val_data = all_data[train_split:val_split]
    test_data = all_data[val_split:]
    
    print(f"\nData split:")
    print(f"Train: {len(train_data)} sentences")
    print(f"Val: {len(val_data)} sentences")
    print(f"Test: {len(test_data)} sentences")
    
    # 4. Encode datasets
    train_encoded = [pipeline.encode(text) for text in train_data]
    
    print("\nSample Encoding (Train[0]):")
    print("Original text:", train_data[0])
    print("Encoded IDs:", train_encoded[0])
    print("Decoded text:", pipeline.decode(train_encoded[0]))
    
    print("\nPipeline is ready for PyTorch Dataset/DataLoader integration!")

if __name__ == "__main__":
    main()
