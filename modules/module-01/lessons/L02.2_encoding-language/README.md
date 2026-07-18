# Lesson L02.2: Encoding Language — ASCII, Unicode, and Beyond
## Day 2: Language as Data | How Computers Represent Text

### Lesson Overview
In Lesson L02.1, you learned that text is fundamentally a sequence of *characters*. But how do computers actually *store* characters? How does a byte become the letter "A"? This lesson explains the mechanics of character encoding: from ASCII (which only knew English) to UTF-8 (which knows the whole world). You will learn why encoding matters — not as theory, but through concrete examples of mojibake (garbled text), security vulnerabilities, and the challenges of preserving endangered languages in digital form. This foundational knowledge will be crucial when you train a tokenizer for Banso in Module 09.

## Table of Contents
- Introduction to Text Encoding
- ASCII: The Foundation
- Limitations of ASCII
- Unicode: A Universal Solution
- UTF-8 Encoding
- Other Encoding Schemes
- Encoding in Python
- Best Practices for Text Handling

---

## Introduction to Text Encoding

Text encoding is the process of converting human-readable text into a format that computers can store and process. This involves mapping characters to numerical codes.

Without proper encoding, computers would not be able to distinguish between letters, numbers, and symbols. Understanding encoding is essential for working with text in any programming language.

---

## ASCII: The Foundation

ASCII (American Standard Code for Information Interchange) was developed in the 1960s. It uses 7 bits to represent 128 characters, including letters, numbers, punctuation, and control characters.

- A-Z: 65-90
- a-z: 97-122
- 0-9: 48-57
- Space: 32

ASCII was sufficient for English text but couldn't handle accented characters or non-Latin scripts.

---

## Limitations of ASCII

ASCII's main limitation is its scope: only 128 characters. This excludes:

- Accented characters (é, ñ, ü)
- Non-Latin alphabets (Greek, Cyrillic, Arabic)
- Emoji and symbols
- East Asian characters

This led to incompatible encoding systems for different languages, causing data corruption when mixing text from different regions.

---

## Unicode: A Universal Solution

Unicode is a standard that assigns a unique number (code point) to every character in every writing system. It currently supports over 140,000 characters.

Unicode solves the compatibility issues of older encodings by providing a universal character set. However, Unicode itself is not an encoding - it's the mapping. Actual storage requires an encoding like UTF-8.

---

## UTF-8 Encoding

UTF-8 is the most widely used Unicode encoding. It uses 1-4 bytes per character:

- ASCII characters: 1 byte
- Latin characters with accents: 2 bytes
- East Asian characters: 3 bytes
- Emoji and rare symbols: 4 bytes

UTF-8 is backward compatible with ASCII and space-efficient for English text. It's the default encoding for the web and most modern systems.

---

## Other Encoding Schemes

Other Unicode encodings include:

- UTF-16: Uses 2 or 4 bytes, efficient for East Asian text
- UTF-32: Uses 4 bytes per character, simple but wasteful

Legacy encodings like Latin-1, Windows-1252, and Shift-JIS are still encountered but should be avoided for new projects.

---

## Encoding in Python

Python 3 uses UTF-8 by default. Key functions:

- str.encode(): Convert string to bytes
- bytes.decode(): Convert bytes to string
- open(file, encoding='utf-8'): Specify encoding when reading files

Always specify encoding when working with files to avoid platform-dependent behavior.

---

## Best Practices for Text Handling

- Always use UTF-8 for new projects
- Specify encoding explicitly when opening files
- Handle encoding errors gracefully
- Test with non-ASCII characters
- Use Unicode normalization for text comparison
- Be aware of byte order marks (BOM) in UTF-8 files

Proper text encoding prevents data corruption and ensures compatibility across systems.

---

## Practical Examples

### Example 1: Working with UTF-8 in Python

```python
# String to bytes (encoding)
text = "Hello, Banso!"
encoded = text.encode('utf-8')
print(encoded)  # b'Hello, Banso!'

# Bytes to string (decoding)
decoded = encoded.decode('utf-8')
print(decoded)  # "Hello, Banso!"

# Unicode code points
for char in "Banso":
    print(f"{char}: U+{ord(char):04X}")
# B: U+0042
# a: U+0061
# n: U+006E
# s: U+0073
# o: U+006F
```

### Example 2: Handling non-Latin scripts

```python
# Greek
greek = "Ελληνικά"  # Greek
print(greek.encode('utf-8'))  # b'\xce\x95\xce\xbb\xce\xbb\xce\xb7\xce\xbd\xce\xb9\xce\xba\xce\xac'

# Arabic
arabic = "العربية"  # Arabic
print(arabic.encode('utf-8'))  # b'\xd8\xb9\xd8\xb1\xd8\xb8\xd9\x8a\xd8\xa9'

# All successfully encoded as UTF-8
```

### Example 3: The danger of wrong encoding

```python
# If you encode as UTF-8 but decode as ASCII:
text = "café"
encoded = text.encode('utf-8')  # b'caf\xc3\xa9'
try:
    decoded = encoded.decode('ascii')
except UnicodeDecodeError:
    print("Error: Cannot decode UTF-8 as ASCII!")
```

---

## Unicode Normalization

Different Unicode representations can look identical but have different byte sequences:

```python
import unicodedata

# Two ways to write "café"
nfc = "café"  # café (single character)
nfd = unicodedata.normalize('NFD', nfc)  # café (e + combining accent)

print(nfc == nfd)  # False! But they look the same
print(nfc.encode('utf-8'))  # b'caf\xc3\xa9'
print(nfd.encode('utf-8'))  # b'cafe\xcc\x81'
```

For text comparison and search, always normalize to a standard form.

---

## Encoding and File I/O

```python
# Always specify encoding when opening files
with open('text.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Writing with encoding
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write("Hello, Banso! Ελληνικά العربية")
```

---

## Closing Reflection: Language Preservation Through Encoding

Why does this matter for MarkGPT?

Consider this: the Banso language is spoken by approximately 100,000 people. Its written form is not standardized globally. When you collect Banso text for training MarkGPT (Module 09), you will encounter multiple encoding standards, inconsistent diacritics, and mixed orthographies.

A team that doesn't understand UTF-8, Unicode normalization, and encoding errors will accidentally corrupt data. A team that does can carefully standardize, preserve, and version-control the data.

Encoding is not glamorous. But it is foundational to text preservation. Every endangered language that survives in the digital age does so because someone, somewhere, cared about the bytes.

When you build MarkGPT with Banso data, you are not just building a language model. You are creating a digital archive. Encoding matters.

*Next: Lesson L02.3 — The Banso Language: Phonology, Grammar, and Writing Systems*