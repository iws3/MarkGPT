# Day 02 Reflection Journal

## Exercise E02.1 — Units of Language
**English Sentence Analysis**
*Sentence:* "The quick brown fox jumps over the lazy dog."
* Counts:
  - Characters (including spaces): 44
  - Words: 9
  - Sentences: 1

**Banso Proverb Analysis**
*Proverb (from `data/banso-vernacular/proverbs.txt`):* "A si mbu fo" (A problem shared is half solved)
* Counts:
  - Characters (including spaces): 11
  - Words: 4
  - Sentences: 1

*Observation:* Banso is highly tonal and concise. The English translation of the proverb takes 31 characters and 6 words to convey the same meaning that the Lamnso' proverb conveys in 11 characters and 4 words.

## Exercise E02.2 — Encoding Language
In Python, using the `ord()` function reveals the ASCII (or Unicode) integer representation of individual characters:
- `ord('A')` returns `65`
- `ord('a')` returns `97` (Notice lowercase 'a' is exactly 32 integers offset from uppercase 'A')
- `ord('0')` returns `48`
- `ord(' ')` (space character) returns `32`

This highlights how linguistic symbols map to numerical values that the computer processes beneath the surface.

## Exercise E02.3 — Low-Resource Languages
**3 African Languages Included in GPT-4's Data:**
1. Swahili 
2. Yoruba
3. Zulu
*(These languages have relatively high digital representation. There are dedicated Wikipedia modules, broad localized web communities, governmental digital documents, and existing standardized parallel corpora. This massive volume of text makes them "high-resource" by African linguistic standards.)*

**3 African Languages NOT (Or Severely Underrepresented) in GPT-4's Data:**
1. Lamnso' (Banso)
2. Bafut
3. Duala
*(These Cameroonian languages lack significant machine-readable digital archives. They are primarily oral, or written materials exist only in physical form like localized bibles or pamphlets. There are no expansive Wikipedia pages or massive internet forums leveraging the dialects. This underscores the necessity of building the MarkGPT Banso dataset to transition Lamnso' into a digitally resilient tier.)*
