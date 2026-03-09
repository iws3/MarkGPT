"""
Day 04 Exercise: Build a Bigram Language Model from Scratch
============================================================
Module 01 | Exercise E04.2

This is your first working language model. It is not a neural network.
It is a statistical model that counts how often word B follows word A
and uses those counts as probabilities.

It is intentionally simple — simple enough to understand completely.
Understanding this model deeply is the foundation for understanding why
we need neural networks, attention, and transformers.

What you will implement:
    1. A text preprocessing pipeline (tokenize and clean the Bible text)
    2. A bigram model: count how often each word pair (w_i, w_{i+1}) appears
    3. A text generator: given a seed word, sample the next word using bigram probs
    4. A perplexity calculator: measure how surprised the model is by held-out text

INSTRUCTIONS:
    - Read every comment carefully before writing any code.
    - Fill in each section marked with "### YOUR CODE HERE ###"
    - Run the file after each section to check your work incrementally.
    - Expected final output: ~20 words of generated "Biblical" text + a perplexity score.

LEARNING GOALS:
    After this exercise, you should be able to explain:
    - What a conditional probability table is and how to build one from text
    - Why higher perplexity means the model is more "surprised" by the text
    - Why bigram models fail at long-range coherence (and what this implies we need)
"""

import random
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(text: str) -> List[str]:
    """
    Convert raw Bible text into a clean list of lowercase word tokens.
    
    Preprocessing is not glamorous, but it matters enormously.
    A model trained on inconsistently preprocessed data learns noise,
    not patterns. "God", "GOD", and "god" are the same concept;
    a model should count them as the same token.
    
    Steps to implement:
        1. Convert everything to lowercase (so "God" and "god" are the same)
        2. Remove punctuation (we want word-level bigrams, not "word." bigrams)
        3. Split on whitespace to get individual tokens
        4. Filter out empty strings (splits on multiple spaces leave empty tokens)
        5. Add a special <EOS> (end-of-sentence) token at verse or sentence breaks.
           This helps the model learn that certain words appear at beginnings and
           ends of verses, which is a real pattern in Biblical language.
    
    Args:
        text: Raw text string (e.g., full text of the KJV Bible or Genesis chapter)
    
    Returns:
        List of lowercase word tokens with <EOS> markers at sentence boundaries.
    
    Example:
        Input:  "In the beginning God created the heaven and the earth."
        Output: ['in', 'the', 'beginning', 'god', 'created', 'the', 
                 'heaven', 'and', 'the', 'earth', '<EOS>']
    """
    import re
    
    tokens = []
    
    # Split into lines first — each line in the Bible text is typically a verse
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        ### YOUR CODE HERE ###
        # Step 1: Convert to lowercase
        line = ...  # TODO
        
        # Step 2: Remove punctuation. Hint: use re.sub(r'[^\w\s]', '', line)
        # The pattern [^\w\s] matches anything that is not a word character or whitespace.
        line = ...  # TODO
        
        # Step 3: Split on whitespace to get word tokens
        words = ...  # TODO
        
        # Step 4: Filter out empty strings
        words = ...  # TODO  (hint: list comprehension with `if word`)
        
        # Step 5: Add the tokens to our list, then add <EOS> at the end of the line
        tokens.extend(words)
        tokens.append('<EOS>')  # This one is done for you!
    
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: BUILD THE BIGRAM MODEL
# ─────────────────────────────────────────────────────────────────────────────

class BigramLanguageModel:
    """
    A bigram language model.
    
    This model captures the conditional probability:
        P(word_i | word_{i-1})
    
    In plain English: "given that I just saw word A, how likely is word B next?"
    
    The model is entirely stored in a dictionary of dictionaries:
        self.probs[word_A][word_B] = probability(word_B follows word_A)
    
    This is called a conditional probability table (or CPT).
    """
    
    def __init__(self, smoothing: float = 0.0):
        """
        Args:
            smoothing: Laplace smoothing parameter (alpha).
                      0.0 = no smoothing (pure MLE).
                      1.0 = add-one (Laplace) smoothing.
                      Try different values and observe the effect on perplexity.
        """
        # Stores counts: self.counts[word_A][word_B] = how often B follows A
        self.counts = defaultdict(Counter)
        
        # Stores probabilities: self.probs[word_A][word_B] = P(B | A)
        # We build this from counts after seeing all the training data.
        self.probs = {}
        
        self.smoothing = smoothing
        self.vocabulary = set()   # All unique tokens seen in training
    
    def train(self, tokens: List[str]) -> None:
        """
        Count all bigrams in the token list and build the probability table.
        
        The key insight: we slide a window of size 2 over the entire token list.
        For each consecutive pair (tokens[i], tokens[i+1]), we increment the count
        of "tokens[i+1] follows tokens[i]" by 1.
        
        After counting all pairs, we normalize: divide each count by the total count
        for that context word, to get probabilities that sum to 1.
        
        Args:
            tokens: Preprocessed list of word tokens from load_and_preprocess().
        """
        # Step 1: Build the vocabulary
        self.vocabulary = set(tokens)
        V = len(self.vocabulary)  # Vocabulary size (needed for smoothing)
        
        # Step 2: Count all consecutive pairs
        ### YOUR CODE HERE ###
        # Hint: iterate over range(len(tokens) - 1), accessing tokens[i] and tokens[i+1]
        for i in range(len(tokens) - 1):
            word_a = ...  # TODO
            word_b = ...  # TODO
            self.counts[word_a][word_b] += 1
        
        # Step 3: Normalize counts to probabilities
        # For each context word, divide each count by the total count for that context.
        # With Laplace smoothing: P(B | A) = (count(A,B) + alpha) / (count(A) + alpha * V)
        ### YOUR CODE HERE ###
        for word_a, follow_counts in self.counts.items():
            
            # Total number of times word_a was followed by anything
            total = ...  # TODO: sum(follow_counts.values())
            
            # Apply smoothing and normalize
            self.probs[word_a] = {}
            for word_b, count in follow_counts.items():
                self.probs[word_a][word_b] = ...  # TODO: (count + smoothing) / (total + smoothing * V)
        
        print(f"Model trained on {len(tokens)} tokens.")
        print(f"Vocabulary size: {V} unique tokens.")
        print(f"Unique bigrams: {sum(len(v) for v in self.counts.values())}")
    
    def probability(self, word_b: str, given_word_a: str) -> float:
        """
        Return P(word_b | given_word_a).
        
        If we have never seen given_word_a in training, return a small default probability.
        This is the "unknown context" problem — a fundamental challenge in language modeling.
        """
        if given_word_a not in self.probs:
            return 1e-10  # Very small but not zero (log(0) is undefined!)
        
        if word_b not in self.probs[given_word_a]:
            # Smoothed probability for unseen bigrams
            if self.smoothing > 0:
                total = sum(self.counts[given_word_a].values())
                V = len(self.vocabulary)
                return self.smoothing / (total + self.smoothing * V)
            return 1e-10
        
        return self.probs[given_word_a][word_b]
    
    def generate(self, seed: str, max_tokens: int = 50) -> str:
        """
        Generate text by sampling from the bigram distribution.
        
        Starting from 'seed', repeatedly:
            1. Look up P(next_word | current_word)
            2. Sample a next_word from that distribution
            3. Append it to the output
            4. Use it as the new current_word
        
        Stop when we hit <EOS> or reach max_tokens.
        
        Args:
            seed:       The starting word (e.g., "god", "in", "the")
            max_tokens: Maximum number of tokens to generate
        
        Returns:
            Generated text as a string.
        """
        current_word = seed.lower()
        output_tokens = [current_word]
        
        for _ in range(max_tokens):
            if current_word not in self.probs:
                # Dead end: we've never seen this word in training context
                break
            
            ### YOUR CODE HERE ###
            # Get the dictionary of {next_word: probability} for current_word
            next_word_probs = ...  # TODO: self.probs[current_word]
            
            # Sample a next word from this distribution.
            # Hint: use random.choices(population=..., weights=..., k=1)[0]
            # where population is the list of possible next words
            # and weights is the list of their probabilities.
            words = list(next_word_probs.keys())
            weights = list(next_word_probs.values())
            next_word = random.choices(words, weights=weights, k=1)[0]
            
            if next_word == '<EOS>':
                break
            
            output_tokens.append(next_word)
            current_word = next_word
        
        return ' '.join(output_tokens)
    
    def perplexity(self, tokens: List[str]) -> float:
        """
        Compute the perplexity of this model on a sequence of tokens.
        
        Perplexity measures how "surprised" the model is by the text.
        A model with perplexity 100 on a test set is, on average, as uncertain
        as if it were choosing uniformly from 100 possible next tokens at each step.
        Lower perplexity = better model.
        
        The formula: PPL = exp(- (1/N) * sum(log P(w_i | w_{i-1})))
        
        Why the negative? Because log probabilities are negative (log of a number 
        between 0 and 1 is negative), so we negate to make the sum positive.
        
        Args:
            tokens: Sequence of tokens from the test set (NOT the training set!)
        
        Returns:
            Perplexity score (float). Typically between 50 and 5000 for this model.
        """
        log_prob_sum = 0.0
        N = 0  # Count of bigrams evaluated
        
        ### YOUR CODE HERE ###
        for i in range(len(tokens) - 1):
            word_a = tokens[i]
            word_b = tokens[i + 1]
            
            # Get the log probability of this bigram.
            # We use log probability (not raw probability) to avoid numerical underflow:
            # multiplying many small numbers together gives 0 in floating point.
            # Adding log probabilities is equivalent to multiplying probabilities.
            p = ...  # TODO: self.probability(word_b, given_word_a=word_a)
            log_prob_sum += ...  # TODO: math.log(p)  [note: log of a small probability is very negative]
            N += 1
        
        if N == 0:
            return float('inf')
        
        # Average negative log probability, exponentiated to give perplexity
        avg_neg_log_prob = -log_prob_sum / N
        return math.exp(avg_neg_log_prob)


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: RUN THE EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load the data ─────────────────────────────────────────────────────────
    # For this exercise, we'll use the Book of Genesis (first few chapters).
    # Replace this string with the content of a file you download.
    # In a real run: with open('data/raw/genesis.txt') as f: raw_text = f.read()
    
    # SAMPLE TEXT (just for testing — replace with full Genesis text!)
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
    
    # ── Preprocess ────────────────────────────────────────────────────────────
    print("Preprocessing text...")
    tokens = load_and_preprocess(sample_genesis)
    print(f"Total tokens: {len(tokens)}")
    print(f"First 20 tokens: {tokens[:20]}")
    print()
    
    # ── Split into train/test ─────────────────────────────────────────────────
    # Use 90% for training, 10% for evaluation.
    # IMPORTANT: We evaluate on HELD-OUT data the model hasn't seen during training.
    # Evaluating on training data is meaningless — the model would just memorize everything.
    split = int(0.9 * len(tokens))
    train_tokens = tokens[:split]
    test_tokens = tokens[split:]
    
    # ── Train the model ────────────────────────────────────────────────────────
    print("Training bigram model (no smoothing)...")
    model_no_smooth = BigramLanguageModel(smoothing=0.0)
    model_no_smooth.train(train_tokens)
    print()
    
    print("Training bigram model (Laplace smoothing, alpha=1.0)...")
    model_smooth = BigramLanguageModel(smoothing=1.0)
    model_smooth.train(train_tokens)
    print()
    
    # ── Generate text ──────────────────────────────────────────────────────────
    random.seed(42)  # Fix seed for reproducible generation
    
    print("=" * 60)
    print("GENERATED TEXT (no smoothing, seed='god'):")
    generated = model_no_smooth.generate("god", max_tokens=30)
    print(f"  {generated}")
    print()
    
    print("GENERATED TEXT (Laplace smoothing, seed='god'):")
    generated = model_smooth.generate("god", max_tokens=30)
    print(f"  {generated}")
    print()
    
    # ── Evaluate perplexity ────────────────────────────────────────────────────
    print("=" * 60)
    ppl_no_smooth = model_no_smooth.perplexity(test_tokens)
    ppl_smooth = model_smooth.perplexity(test_tokens)
    
    print(f"PERPLEXITY (no smoothing):    {ppl_no_smooth:.1f}")
    print(f"PERPLEXITY (Laplace smooth):  {ppl_smooth:.1f}")
    print()
    
    print("REFLECTION QUESTIONS (write your answers in your journal):")
    print("  1. Which model has lower perplexity? Why?")
    print("  2. What does the generated text look like? Where does it break down?")
    print("  3. What would a TRIGRAM model do differently? Would it be better?")
    print("  4. What can a bigram model never capture that a GPT can?")


if __name__ == "__main__":
    main()
