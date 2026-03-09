"""
Banso-Aware Tokenizer for MarkGPT
===================================
A Byte Pair Encoding (BPE) tokenizer trained specifically on a mixture
of KJV Bible text and Banso vernacular text (Lamnso').

Why a custom tokenizer matters:
  If we used GPT-2's tokenizer (trained on English web text), Banso words
  would be fragmented badly. A word like "Nfor" (God in Lamnso') might 
  become ["N", "for"] — completely losing its identity.
  
  Our custom tokenizer learns from the actual data distribution and produces
  meaningful subword units for Banso, dramatically reducing "fertility"
  (tokens-per-word) and improving model efficiency.

This tokenizer wraps HuggingFace's tokenizers library and adds:
  - Banso-specific preprocessing (tonal markers, orthographic normalization)
  - Special tokens for MarkGPT
  - Fertility analysis utilities

Relevant reading:
  - Sennrich et al. (2016): Neural MT of Rare Words with Subword Units (BPE)
  - Kudo & Richardson (2018): SentencePiece: A simple and language independent
    subword tokenizer
"""

import os
import re
import json
from pathlib import Path
from typing import List, Optional, Union
from collections import Counter

# We use HuggingFace tokenizers for the heavy lifting,
# but understand BPE deeply from Module 05 before relying on it as a black box.
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel, Whitespace
    from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
    from tokenizers.processors import TemplateProcessing
    HF_TOKENIZERS_AVAILABLE = True
except ImportError:
    HF_TOKENIZERS_AVAILABLE = False
    print("Warning: 'tokenizers' library not installed. Run: pip install tokenizers")


# ─────────────────────────────────────────────────────────────────────────────
# SPECIAL TOKENS
# ─────────────────────────────────────────────────────────────────────────────

SPECIAL_TOKENS = {
    # Standard LM tokens
    "bos_token": "<|bos|>",         # Beginning of sequence
    "eos_token": "<|eos|>",         # End of sequence
    "pad_token": "<|pad|>",         # Padding (for batching variable-length sequences)
    "unk_token": "<|unk|>",         # Unknown token (should be rare with BPE)
    
    # Bible-specific tokens — help the model learn verse structure
    "verse_start": "<|verse|>",     # Start of a Bible verse
    "chapter_start": "<|chapter|>", # Start of a chapter
    "book_start": "<|book|>",       # Start of a book
    
    # Language markers — crucial for mixed-language modeling
    "lang_en": "<|en|>",            # English text follows
    "lang_bn": "<|bn|>",            # Banso/Lamnso' text follows
    
    # Formatting
    "newline": "<|n|>",             # Paragraph break
}


# ─────────────────────────────────────────────────────────────────────────────
# BANSO TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

class BansoPreprocessor:
    """
    Text preprocessing specific to the Banso/Lamnso' language.
    
    Lamnso' is a tonal Grassfields Bantu language spoken in the Nso' Kingdom,
    Northwest Region of Cameroon. It uses a Latin-based orthography developed
    by SIL International, with special characters for tonal marking.
    
    Key orthographic features we handle:
      - Tone marks: high tone (á), low tone (à), falling tone (â)
      - Nasal prefixes: ŋ (velar nasal, as in "ŋgɔm" = king)
      - Open vowels: ɔ (open-mid back) and ɛ (open-mid front)
      - Prenasalized consonants: mb, nd, ng, nj
    """
    
    # Known Banso/Lamnso' word list for vocabulary preservation
    # (expand this with native speaker input)
    BANSO_CORE_VOCAB = [
        "nfor",      # God / chief
        "lamnso",    # the Nso' language
        "nso",       # the Nso' people/kingdom
        "wir",       # we/us
        "kiim",      # death
        "saaki",     # truth
        "shiy",      # peace / shalom
        "mbanyam",   # blessing
        "wvisi",     # prophet
        "taav",      # father
        "maav",      # mother
        "kiv",       # to speak / word
        "ntεm",      # heart
        "ngam",      # great
        "nggay",     # praise
        "vεnlε",     # believe
        "kivri",     # prayer
        "fεntεm",    # spirit / heart
    ]
    
    def __init__(self):
        # Orthographic normalization map: common variant spellings → canonical form
        self.normalization_map = {
            "ŋ": "ng",   # Normalize velar nasal for tokenizer compatibility
            "ɔ": "o",    # Normalize open-mid back vowel (lossy but practical)  
            "ɛ": "e",    # Normalize open-mid front vowel
            "\u2019": "'",  # Curly apostrophe → straight apostrophe
            "\u2018": "'",  # Left single quotation mark → apostrophe
            "–": "-",    # En dash → hyphen
            "—": " - ",  # Em dash → hyphen with spaces
        }
    
    def normalize(self, text: str, preserve_tone: bool = False) -> str:
        """
        Normalize Banso text for consistent tokenization.
        
        Args:
            text: Raw Banso text
            preserve_tone: If True, keep tone diacritics. If False, strip them.
                          Set preserve_tone=True if your training data reliably
                          marks tone. Set False if tone marking is inconsistent.
        
        Returns:
            Normalized text string.
        """
        # Apply character-level normalization
        for old, new in self.normalization_map.items():
            text = text.replace(old, new)
        
        # Strip tone diacritics unless explicitly preserved
        if not preserve_tone:
            # Remove combining diacritical marks (U+0300–U+036F)
            import unicodedata
            text = unicodedata.normalize('NFD', text)
            text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
            text = unicodedata.normalize('NFC', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def detect_language(self, text: str) -> str:
        """
        Heuristic language detection for Banso vs. English text.
        Returns 'banso' or 'english'.
        
        This is a simple heuristic based on known Banso vocabulary.
        For production, use a trained language detector.
        """
        text_lower = text.lower()
        word_count = len(text_lower.split())
        if word_count == 0:
            return 'unknown'
        
        banso_matches = sum(1 for word in self.BANSO_CORE_VOCAB if word in text_lower)
        banso_ratio = banso_matches / max(word_count, 1)
        
        return 'banso' if banso_ratio > 0.05 else 'english'
    
    def add_language_tags(self, text: str) -> str:
        """
        Prepend language tag to a text segment.
        These tags help MarkGPT learn code-switching and stay in the right register.
        """
        lang = self.detect_language(text)
        tag = SPECIAL_TOKENS['lang_bn'] if lang == 'banso' else SPECIAL_TOKENS['lang_en']
        return f"{tag} {text}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TOKENIZER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MarkGPTTokenizer:
    """
    The MarkGPT tokenizer: a trained BPE tokenizer with Banso support.
    
    Usage:
        # Train from scratch on your corpus
        tokenizer = MarkGPTTokenizer()
        tokenizer.train(["data/processed/kjv_clean.txt",
                         "data/banso-vernacular/banso_bible.txt"],
                        vocab_size=8000)
        tokenizer.save("data/banso-vernacular/tokenizer/")
        
        # Load a pre-trained tokenizer
        tokenizer = MarkGPTTokenizer.from_pretrained("data/banso-vernacular/tokenizer/")
        
        # Encode text
        ids = tokenizer.encode("In the beginning, God created the heavens")
        print(ids)  # [34, 12, 908, 4, 231, ...]
        
        # Decode back to text
        text = tokenizer.decode(ids)
        print(text)  # "In the beginning, God created the heavens"
        
        # Check fertility (tokens per word — lower is better for a language)
        fertility = tokenizer.fertility("Nfor a wirnii Lamnso wirɨ") 
        print(f"Fertility: {fertility:.2f} tokens/word")
    """
    
    def __init__(self):
        self.preprocessor = BansoPreprocessor()
        self._tokenizer = None
        self.vocab_size = None
        
        # Token ID lookups (populated after training/loading)
        self.bos_id = None
        self.eos_id = None
        self.pad_id = None
        self.unk_id = None
    
    def train(
        self,
        files: List[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
        output_dir: Optional[str] = None
    ):
        """
        Train a BPE tokenizer on the given text files.
        
        The BPE algorithm (in brief):
          1. Start with a vocabulary of individual characters
          2. Count all adjacent pairs of tokens in the corpus
          3. Merge the most frequent pair into a new token
          4. Repeat until vocab_size is reached
        
        The result is a vocabulary where common words are single tokens,
        and rare words are split into meaningful subwords.
        
        Args:
            files:         List of paths to training text files
            vocab_size:    Target vocabulary size. Typical: 4000–32000.
            min_frequency: Minimum frequency for a pair to be merged.
            output_dir:    If provided, save trained tokenizer here.
        """
        if not HF_TOKENIZERS_AVAILABLE:
            raise ImportError("Install the tokenizers library: pip install tokenizers")
        
        print(f"Training BPE tokenizer on {len(files)} file(s)...")
        print(f"Target vocabulary size: {vocab_size}")
        
        # Initialize a new BPE model
        self._tokenizer = Tokenizer(BPE(unk_token=SPECIAL_TOKENS["unk_token"]))
        
        # Pre-tokenizer: split on whitespace before applying BPE
        # This means BPE will only merge within words, not across word boundaries
        self._tokenizer.pre_tokenizer = Whitespace()
        
        # Define all special tokens so they are never split
        all_special_tokens = list(SPECIAL_TOKENS.values())
        
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=all_special_tokens,
            # Ensure all core Banso vocabulary is included
            initial_alphabet=list(set(''.join(BansoPreprocessor.BANSO_CORE_VOCAB))),
        )
        
        # Train! This reads through all files and runs the BPE merge algorithm.
        self._tokenizer.train(files, trainer)
        
        self.vocab_size = self._tokenizer.get_vocab_size()
        self._update_special_token_ids()
        
        print(f"Training complete. Final vocabulary size: {self.vocab_size}")
        
        if output_dir:
            self.save(output_dir)
    
    def _update_special_token_ids(self):
        """Cache the IDs of special tokens for fast access."""
        vocab = self._tokenizer.get_vocab()
        self.bos_id = vocab.get(SPECIAL_TOKENS["bos_token"], 0)
        self.eos_id = vocab.get(SPECIAL_TOKENS["eos_token"], 1)
        self.pad_id = vocab.get(SPECIAL_TOKENS["pad_token"], 2)
        self.unk_id = vocab.get(SPECIAL_TOKENS["unk_token"], 3)
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        normalize: bool = True
    ) -> List[int]:
        """
        Convert text to a list of token IDs.
        
        Args:
            text:               Input text string
            add_special_tokens: If True, wrap with <|bos|> and <|eos|>
            normalize:          If True, apply Banso normalization
        
        Returns:
            List of integer token IDs
        """
        if normalize:
            text = self.preprocessor.normalize(text)
        
        encoding = self._tokenizer.encode(text)
        ids = encoding.ids
        
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        
        return ids
    
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Convert a list of token IDs back to text.
        
        Args:
            ids:                  List of integer token IDs
            skip_special_tokens:  If True, remove special tokens from output
        
        Returns:
            Decoded text string
        """
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def fertility(self, text: str) -> float:
        """
        Compute the fertility of this tokenizer on the given text.
        Fertility = average number of tokens per word.
        
        A perfect tokenizer for a language would have fertility ~1.0.
        English with GPT-2's tokenizer is ~1.3.
        A poorly-fit tokenizer might give fertility 3+ for unseen languages.
        
        This metric lets us compare tokenizers trained on different data.
        Lower fertility = more efficient representation for this language.
        """
        words = text.split()
        if not words:
            return 0.0
        tokens = self.encode(text, add_special_tokens=False)
        return len(tokens) / len(words)
    
    def analyze_vocabulary(self) -> dict:
        """
        Print a summary of what the vocabulary has learned.
        Useful for understanding what the tokenizer "knows" about Banso.
        """
        vocab = self._tokenizer.get_vocab()
        
        # Find Banso-related tokens (contain known Banso subwords)
        banso_tokens = [
            tok for tok in vocab.keys()
            if any(bw in tok.lower() for bw in BansoPreprocessor.BANSO_CORE_VOCAB)
        ]
        
        # Token length distribution
        lengths = Counter(len(tok) for tok in vocab.keys())
        
        return {
            "total_vocab_size": len(vocab),
            "banso_related_tokens": len(banso_tokens),
            "banso_token_examples": banso_tokens[:20],
            "token_length_distribution": dict(sorted(lengths.items())),
            "special_tokens": {k: vocab.get(v) for k, v in SPECIAL_TOKENS.items()},
        }
    
    def save(self, directory: str):
        """Save tokenizer to directory."""
        os.makedirs(directory, exist_ok=True)
        self._tokenizer.save(os.path.join(directory, "tokenizer.json"))
        # Save special token metadata
        with open(os.path.join(directory, "special_tokens.json"), "w") as f:
            json.dump(SPECIAL_TOKENS, f, indent=2, ensure_ascii=False)
        print(f"Tokenizer saved to {directory}")
    
    @classmethod
    def from_pretrained(cls, directory: str) -> "MarkGPTTokenizer":
        """Load a pre-trained tokenizer from directory."""
        instance = cls()
        instance._tokenizer = Tokenizer.from_file(
            os.path.join(directory, "tokenizer.json")
        )
        instance.vocab_size = instance._tokenizer.get_vocab_size()
        instance._update_special_token_ids()
        return instance


# ─────────────────────────────────────────────────────────────────────────────
# FERTILITY BENCHMARK UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_fertility(tokenizer: MarkGPTTokenizer, test_texts: dict) -> None:
    """
    Module 09 Exercise: Compare tokenizer fertility across languages.
    
    This function is used in the Module 09 exercise to compare fertility
    of tokenizers with different vocabulary sizes and training data.
    
    Args:
        tokenizer:  A trained MarkGPTTokenizer
        test_texts: Dict mapping language name → sample text
    
    Example output:
        Language     | Words | Tokens | Fertility
        English KJV  |   847 |   1023 | 1.21
        Banso Proverb|    42 |     71 | 1.69   ← our tokenizer does much better than GPT-2
    """
    print(f"\n{'Language':<20} | {'Words':>6} | {'Tokens':>7} | {'Fertility':>9}")
    print("-" * 55)
    
    for lang_name, text in test_texts.items():
        words = len(text.split())
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        fertility = tokens / max(words, 1)
        print(f"{lang_name:<20} | {words:>6} | {tokens:>7} | {fertility:>9.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("MarkGPT Tokenizer Demo")
    print("=" * 50)
    print("To train a tokenizer, run:")
    print()
    print("  from src.tokenizer.tokenizer import MarkGPTTokenizer")
    print("  tok = MarkGPTTokenizer()")
    print('  tok.train(["data/processed/kjv_clean.txt",')
    print('             "data/banso-vernacular/banso_bible.txt"],')
    print('             vocab_size=8000,')
    print('             output_dir="data/banso-vernacular/tokenizer/")')
    print()
    print("See modules/module-05/lessons/L25_tokenization.md for the full walkthrough.")
    print("See modules/module-09/lessons/L53_banso_tokenizer.md for Banso-specific training.")
