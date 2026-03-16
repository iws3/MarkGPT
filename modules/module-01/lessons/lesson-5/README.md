# Lesson L02.1: Characters, Words, Sentences — Units of Language
## Day 2: Language as Data | Understanding the Building Blocks of Text

### Lesson Overview
How does a computer see language? Not as meaning or poetry, but as data — as sequences of characters, bytes, and symbols. This lesson begins at the foundation: How are letters stored? What is the difference between a character and a byte? How does a neural network know where one word ends and another begins? By the end of this lesson, you will understand the data structures that underlie all text processing: character encoding (ASCII, Unicode), tokenization strategies, and text normalization. These are not glamorous topics, but they are essential. Many bugs in NLP projects trace back to encoding confusion or incorrect tokenization.

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

---

## Sentences and Segmentation

Identifying sentence boundaries is trickier than it seems:

- Periods, question marks, exclamation marks typically mark the end
- But periods appear in abbreviations, decimals, and URLs
- Different languages have different conventions

Sentence segmentation algorithms use heuristics and machine learning to handle these edge cases.

---

## Text Normalization

Before processing, text often needs normalization:

- **Lowercasing**: "The" and "the" become the same token
- **Removing punctuation**: "Hello, world!" becomes "Hello world"
- **Whitespace normalization**: Multiple spaces become one
- **Accent removal**: "café" becomes "cafe"
- **Stop word removal**: Common words like "the", "a", "is"

However, for language models like MarkGPT, normalization must be done carefully. Removing case information loses valuable signal. Removing punctuation might be harmful for stylistic preservation.

---

## Language-Specific Considerations

Different languages require different handling:

- **CJK languages (Chinese, Japanese, Korean)**: No spaces between words; need character-level or morphological tokenization
- **Morphologically rich languages**: Turkish, Finnish — words have many prefixes/suffixes
- **Right-to-left languages**: Arabic, Hebrew — text flows right-to-left
- **Tonal languages**: Banso, Mandarin — tones modify meaning; must be preserved

For MarkGPT, this is crucial. Banso is a tonal language with specific phonological features. ASCII-based tokenization would fail. We will need a tokenizer specifically trained on Banso data (Module 09).

---

## Challenges in Text Processing

Common challenges:

- **Out-of-vocabulary (OOV) words**: Words not in the training vocabulary
- **Misspellings and typos**: Real-world text is messy
- **Multilingual text**: Code-switching between languages
- **Special characters**: Emojis, mathematical symbols, etc.
- **Encoding errors**: Incorrect or mixed encodings

---

## Tools and Libraries

**Python libraries for text processing:**

- `nltk`: Classic NLP toolkit with tokenizers and preprocessing
- `spaCy`: Industrial-strength NLP with modern design
- `tokenizers`: Hugging Face tokenizers library (used by transformers)
- `SentencePiece`: Language-agnostic tokenizer (good for low-resource languages)

**For Banso specifically:**
- We will train custom tokenizers in Module 09
- Current open-source tools are English-focused

---

## Practical Examples

### Example 1: Simple tokenization in Python

```python
text = "The dog barked loudly. What happened?"
tokens = text.split()  # Simple split
print(tokens)  # ['The', 'dog', 'barked', 'loudly.', 'What', 'happened?']

# With punctuation removal
import string
cleaned = text.translate(str.maketrans('', '', string.punctuation))
tokens = cleaned.split()
print(tokens)  # ['The', 'dog', 'barked', 'loudly', 'What', 'happened']
```

### Example 2: Character encoding

```python
text = "Hello"
bytes_repr = text.encode('utf-8')
print(bytes_repr)  # b'Hello'

# For non-ASCII
text_african = "Banso"
bytes_repr = text_african.encode('utf-8')
print(bytes_repr)  # b'Banso'

# With accents
text_accented = "café"
bytes_repr = text_accented.encode('utf-8')
print(bytes_repr)  # b'caf\xc3\xa9'
```

---

## Closing Reflection: Text as Discrete Objects

Coming from the history of AI and the theory of language models, you might expect this lesson to be about *meaning*. Instead, it's about *bytes and spaces*.

This is intentional. Machine learning does not work on meaning directly. It works on discrete, countable objects: characters, tokens, numbers. Before MarkGPT can learn to generate text with meaning (or at least text that *appears* to have meaning), it must first learn how to count and represent text as data.

On Day 2, we go one level deeper: how do machines encode *meaning* itself? That is tokenization and embeddings. But this foundation — understanding that text is data, that encoding matters, that tokenization is a choice — is essential.

When you train MarkGPT on Banso text in Module 09, you will remember this lesson. You will carefully choose how to tokenize Banso words. That choice will ripple through every subsequent layer of the model.

*Next: Lesson L02.2 — Encoding Language: ASCII, Unicode, and Beyond*