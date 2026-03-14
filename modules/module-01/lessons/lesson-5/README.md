# Lesson 5: Characters, Words, Sentences: Units of Language
## Understanding the Building Blocks of Text

## Table of Contents
- Characters and Encoding
- Words and Tokenization
- Sentences and Segmentation
- Text Normalization
- Language-Specific Considerations
- Challenges in Text Processing
- Tools and Libraries
- Practical Examples

---

## Characters and Encoding

Text is fundamentally composed of characters. Understanding character encoding is crucial for text processing.

- ASCII: 7-bit encoding for basic Latin characters

- UTF-8: Variable-length encoding supporting all Unicode characters

- Unicode: Standard for representing text in all writing systems

Proper encoding ensures that text is correctly interpreted across different systems and languages.

---

## Words and Tokenization

Tokenization is the process of breaking text into words or subwords.

- Word-level tokenization: Split on spaces and punctuation

- Subword tokenization: Break words into meaningful units (BPE, WordPiece)

- Character-level: Treat each character as a token

Modern LLMs use subword tokenization to handle out-of-vocabulary words and reduce vocabulary size.