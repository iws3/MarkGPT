# Lesson 6: Encoding Language: ASCII, Unicode, and Beyond
## How Computers Represent Text

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