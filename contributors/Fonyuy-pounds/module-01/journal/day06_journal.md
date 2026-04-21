# Day 06: Module 01 Review & Mini-Project 1: BansoGram
**Date:** 2026-04-19
**Theme:** Consolidation and first hands-on build

## 📖 Module 01 Review Summary
*Briefly list the most important thing you learned in each of the first 5 days.*
- **Day 1 (History):** AI history shows a pendulum between rules-based and neural systems.
- **Day 2 (Encoding/Banso):** Language tokenization defines how an AI understands concepts and words.
- **Day 3 (Math):** Linear algebra handles vectors, probabilities measure surprise in the language sequence.
- **Day 4 (Probability):** Language models are just probability machines estimating P(next_word | previous_words).
- **Day 5 (N-Grams):** Statistical N-grams are strong heuristics but lack wide contextual understanding.

## 🏗️ Mini-Project 1: BansoGram
### 1. Implementation Notes
*What was the hardest part about building BansoGram? (e.g., smoothing, preprocessing, N-size handling)*
The toughest part was ensuring proper context size padding for variable length sequence smoothing using Laplace logic (so zero probabilities would not throw off perplexity). Also, parsing TSV files correctly to extract the Lamnso' phrases independently from their translations.

### 2. Output Comparison
**English (KJV) Sample Output:**
> "And God divided the light from the darkness. And God called the firmament Heaven"

**Banso-Inflected Sample Output:**
> "a si mbu fo'na nsi'na ndaw"

**Observations:** 
*How do the "voices" of the two models differ? Does the Banso model capture the rhythm of the proverbs?*
The Banso model outputs much shorter, highly cohesive clusters of tokens because its training data consists of 10 short proverbs, restricting its vocabulary and leading it to stitch direct sentences. English KJV has repetitive vocabulary (like "And God said..."), allowing the Trigram model to chain realistic sounding biblical syntax smoothly.

### 3. Quantitative Evaluation (Perplexity)
| Model | Perplexity (Test Set) |
|---|---|
| English (Trigram) | 26.54 |
| Banso (Trigram) | 5.32 |

*Why do you think one has higher/lower perplexity than the other? (Hint: Consider dataset size and vocabulary variety).*
The English dataset is slightly more varied, resulting in higher perplexity, whereas the Banso' dataset uses constrained proverbs. The limited Banso text means lower perplexity because there are very few choices for the 'next token' historically seen in context.

---

## 🧠 Reflection
**1. What does it feel like to built your first language model "from scratch"?**
It actually demystifies the black-box magic of Generative AI. Seeing that "intelligence" here stems strictly from probability math over preceding context tuples proves it’s pure statistics.

**2. Based on your BansoGram results, why do you think modern LLMs (like GPT-4) need billions of parameters and neural networks instead of just higher-order N-Grams?**
Because N-Grams hit zero-probability sparsity rapidly as N increases. A 5-Gram would only repeat text entirely memorized in the training set and would not generalize or capture distant contextual sentiment dynamically the way neural embeddings can.

**3. Module 01 Completion Goal Check:**
"What I wanted to understand about language models by Day 60..." (Look back at Day 1). How has your understanding changed in just 6 days?
I now understand how human text inputs map strictly to computational numbers (tokens), how math quantifies learning (Cross-Entropy & Perplexity), and the limits of hardcoded probability tracking over Neural Nets mapping.

---
*End of Module 01. Proceed to Module 02.*
