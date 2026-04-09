#!/usr/bin/env python3
"""
Module-05 enrichment part 1 - 35 commits
Tokenization fundamentals
"""
import subprocess
import os

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-05')

sections = [
    ("## Tokenization Fundamentals\n\n"
     "### What is Tokenization?\n\n"
     "Converting text → tokens (numbers)\n"
     "Token: Smallest unit (word, subword, character)\n"
     "Required step for all NLP models\n"
     "Quality crucial: Affects downstream tasks\n"
     "Trade-off: Granularity vs vocabulary size\n\n",
     "Add tokenization intro"),
    
    ("### Character-level Tokenization\n\n"
     "Simplest: Each character = token\n"
     "Alphabet size: 26 + digits + punct ≈ 100\n"
     "Pros: Handles any text (misspellings, OOV)\n"
     "Cons: Sequences very long, harder to learn\n"
     "Example: \"Hello\" → [H,e,l,l,o]\n"
     "Used in: Character-level language models\n\n",
     "Add character level"),
    
    ("### Word-level Tokenization\n\n"
     "Split on whitespace and punctuation\n"
     "Vocab size: 10K-100K typical\n"
     "Pros: Interpretable, reasonable length\n"
     "Cons: OOV problem (unknown words)\n"
     "Example: \"Hello world!\" → [\"Hello\", \"world\", \"!\"]\n"
     "Problem: \"hello\" vs \"Hello\" = different tokens\n\n",
     "Add word level"),
    
    ("### Subword Tokenization\n\n"
     "Middle ground: Parts of words\n"
     "Vocab size: 1K-100K\n"
     "Examples: Byte-Pair Encoding, WordPiece\n"
     "Balances word and character levels\n"
     "Standard in modern NLP\n"
     "\"Hello\" → [\"He\", \"llo\"] or [\"Hel\", \"lo\"]\n\n",
     "Add subword level"),
    
    ("## Byte-Pair Encoding (BPE)\n\n"
     "### Algorithm\n\n"
     "1. Start with characters + special symbols\n"
     "2. Count all adjacent pairs\n"
     "3. Merge most frequent pair\n"
     "4. Repeat until vocab size reached\n"
     "Simple greedy algorithm\n"
     "Very effective in practice\n\n",
     "Add BPE algorithm"),
    
    ("### BPE Example\n\n"
     "Text: \"hello hello\"\n"
     "Initial: [h,e,l,l,o, ,h,e,l,l,o]\n"
     "Step 1: \"l\" \"l\" frequent → [h,e,ll,o, ,h,e,ll,o]\n"
     "Step 2: \"h\" \"e\" frequent → [he,ll,o, ,he,ll,o]\n"
     "Step 3: \"he\" \"ll\" frequent → [hell,o, ,hell,o]\n"
     "Result: [hell, o, </s>, hell, o]\n\n",
     "Add BPE example"),
    
    ("### BPE Advantages\n\n"
     "Handles misspellings: \"helo\" → [\"he\", \"lo\"]\n"
     "Compression: Frequent words = single token\n"
     "Vocabulary: Finite size (predictable memory)\n"
     "Language independent: Works on any language\n"
     "Reversible: Can decode back\n"
     "Reproducible: Same text → same tokens\n\n",
     "Add BPE advantages"),
    
    ("## WordPiece Tokenization\n\n"
     "### Differences from BPE\n\n"
     "Merge criterion: Likelihood maximization\n"
     "Not just frequency\n"
     "Used in: BERT, RoBERTa\n"
     "Similar results to BPE\n"
     "Slightly different algorithm\n"
     "Both work well in practice\n\n",
     "Add WordPiece"),
    
    ("## SentencePiece\n\n"
     "### Language Agnostic\n\n"
     "Works on raw text (no preprocessing)\n"
     "No language-specific logic\n"
     "Combines BPE and unigram language model\n"
     "Treats space as token\n"
     "Great for non-Latin scripts\n"
     "Used in: T5, mBERT, many recent models\n\n",
     "Add SentencePiece"),
    
    ("## Vocabulary Size Impact\n\n"
     "### Small Vocab (1K)\n\n"
     "Sequence length: Very long\n"
     "Memory per sample: High\n"
     "Training time: Slow\n"
     "Parameter count: Lower (embedding matrix)\n"
     "Typical: Character-level models\n\n",
     "Add small vocab"),
    
    ("### Large Vocab (100K)\n\n"
     "Sequence length: Short\n"
     "Memory per sample: Low\n"
     "Training time: Fast\n"
     "Parameter count: Very high\n"
     "Typical: BERT, GPT\n"
     "Trade-off: Memory vs speed\n\n",
     "Add large vocab"),
    
    ("## Special Tokens\n\n"
     "### Standard Special Tokens\n\n"
     "[CLS]: Classification token (start)\n"
     "[SEP]: Separator (between sentences)\n"
     "[PAD]: Padding (fill short sequences)\n"
     "[UNK]: Unknown (OOV words)\n"
     "[MASK]: Masked token (BERT pre-training)\n"
     "</s>: End of sequence\n"
     "<s>: Start of sequence\n\n",
     "Add special tokens"),
    
    ("### Custom Tokens\n\n"
     "Task-specific: [QUESTION], [ANSWER]\n"
     "Entity types: [PER], [LOC], [ORG]\n"
     "Domain-specific: [CODE], [EQUATION]\n"
     "Improves performance\n"
     "Requires fine-tuning\n"
     "Common in production systems\n\n",
     "Add custom tokens"),
    
    ("## Handling OOV Words\n\n"
     "### Problem\n\n"
     "Word not in vocabulary → [UNK]\n"
     "Loses information\n"
     "Subword tokenization helps\n"
     "\"unrecognizable\" → [\"unrecognizable\"]\n"
     "BPE: [\"unrecogniz\", \"able\"]\n"
     "Preserves information!\n\n",
     "Add OOV problem"),
    
    ("### Solutions\n\n"
     "1. Subword tokenization (BPE, WordPiece)\n"
     "2. Character-level fallback\n"
     "3. Morphological analysis\n"
     "4. Expand vocabulary\n"
     "5. Back-off smoothing (pre-training trick)\n"
     "Best: Combine approaches\n\n",
     "Add OOV solutions"),
    
    ("## Tokenization Quality Metrics\n\n"
     "### Compression Ratio\n\n"
     "Average tokens per word\n"
     "1.0: Perfect (1 token per word)\n"
     "1.3: Good (3 tokens per 10 words)\n"
     "2.0: Poor (half as many words)\n"
     "Impact: Memory and compute\n"
     "Typical: 1.1-1.3 for English\n\n",
     "Add compression ratio"),
    
    ("### Vocabulary Coverage\n\n"
     "% of corpus tokens that are in-vocabulary\n"
     "BERT (30K vocab): 98%+\n"
     "GPT-2 (50K vocab): 99%+\n"
     "Smaller vocab: Lower coverage\n"
     "Affects performance\n"
     "Trade-off: Size vs coverage\n\n",
     "Add coverage"),
    
    ("## Contextual Tokenization\n\n"
     "### Problem: Ambiguity\n\n"
     "\"bank\" = financial vs river bank\n"
     "Single tokenization misses context\n"
     "Morphologically: Same\n"
     "Solution: Same token, different embeddings\n"
     "Transformers learn contextual meaning\n\n",
     "Add contextual"),
    
    ("## Tokenization in Code\n\n"
     "### Using HuggingFace\n\n"
     "```python\n"
     "from transformers import AutoTokenizer\n\n"
     "tokenizer = AutoTokenizer.from_pretrained('bert-base')\n"
     "tokens = tokenizer.encode('Hello world')\n"
     "# [101, 7592, 2088, 102]\n"
     "text = tokenizer.decode(tokens)\n"
     "# 'Hello world'\n"
     "```\n\n",
     "Add tokenization code"),
    
    ("### Custom Tokenizers\n\n"
     "```python\n"
     "from tokenizers import Tokenizer\n"
     "from tokenizers.models import BPE\n\n"
     "tokenizer = Tokenizer(BPE())\n"
     "# Train on your data\n"
     "tokenizer.train_from_iterator(texts, vocab_size=50000)\n"
     "# Use it\n"
     "encoding = tokenizer.encode('Your text')\n"
     "```\n\n",
     "Add custom tokenizer"),
    
    ("## Multilingual Tokenization\n\n"
     "### Challenges\n\n"
     "100+ languages, different scripts\n"
     "Chinese: Words not separated\n"
     "Arabic: Right-to-left\n"
     "Agglutinative: Turkish, Finnish\n"
     "Solution: Universal tokenizers\n"
     "SentencePiece: Language-agnostic\n\n",
     "Add multilingual"),
    
    ("### mBERT Tokenization\n\n"
     "Shared tokenizer across 104 languages\n"
     "110K vocabulary\n"
     "WordPiece trained on multilingual corpus\n"
     "Works reasonably well\n"
     "Trade-off: Per-language quality\n"
     "Enables zero-shot cross-lingual transfer\n\n",
     "Add mBERT tokenizer"),
    
    ("## Tokenization Speed\n\n"
     "### Practical Performance\n\n"
     "BERT tokenizer: 10K tokens/s\n"
     "SentencePiece: 100K tokens/s\n"
     "Character-level: 1M tokens/s\n"
     "Bottleneck: Usually data loading\n"
     "Cache tokens during preprocessing\n"
     "Pre-tokenize for speed\n\n",
     "Add tokenization speed"),
    
    ("## Preprocessing Pipeline\n\n"
     "### Best Practices\n\n"
     "1. Normalize: Lowercasing (task-dependent)\n"
     "2. Remove: Accents, special chars (careful!)\n"
     "3. Tokenize: Use standard tokenizer\n"
     "4. Truncate: Limit length\n"
     "5. Pad: Make same length for batching\n"
     "6. Convert to IDs: Vocabulary lookup\n\n",
     "Add preprocessing"),
    
    ("## Tokenization Errors\n\n"
     "### Common Issues\n\n"
     "1. **Case sensitivity**: \"Hello\" ≠ \"hello\"\n"
     "   Solution: Lowercase before tokenizing\n\n"
     "2. **Punctuation**: Attached vs separate\n"
     "   Solution: Check tokenizer behavior\n\n"
     "3. **Contractions**: \"don't\" → [\n"
     "   Solution: Expand or handle in tokenizer\n\n"
     "4. **Whitespace**: Multiple spaces\n"
     "   Solution: Normalize whitespace\n\n",
     "Add tokenization errors"),
    
    ("## Vocabulary Learning\n\n"
     "### From Unlabeled Data\n\n"
     "Train on large corpus\n"
     "No labels needed\n"
     "Learn frequent patterns\n"
     "Adapt to domain\n"
     "Example: Train on Wikipedia → Reddit\n"
     "Learn Reddit-specific slang\n\n",
     "Add vocab learning"),
]

readme_path = 'README.md'

print(f"Starting module-05 part 1 with {len(sections)} commits...")
print("=" * 60)

for i, (content, msg) in enumerate(sections, 1):
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(content)
    
    try:
        subprocess.run(['git', 'add', 'README.md'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', f'Enrich module-05 part1 {i}: {msg}'], check=True, capture_output=True)
        print(f"[OK] Commit {i:2d}/{len(sections)}: {msg}")
    except subprocess.CalledProcessError:
        print(f"[FAIL] Part 1 {i:2d}: {msg}")

print("=" * 60)
print(f"[DONE] Part 1 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
