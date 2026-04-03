# Day 01 Reflection Journal

## Exercise E01.1 — The AI Family Tree
**Reflection:**
What surprised me most about this history is how cyclical it has been, the alternating pattern of funding "summers" followed by "winters" based on over-promising and under-delivering. If I were a researcher in 1956 at the Dartmouth workshop, I would have likely predicted that general human-like reasoning would be achieved much faster through symbolic logic rules before hitting the hard ceiling of combinatorial explosion.

## Exercise E01.2 — Hands-On with a Language Model
**Part A — Baseline English Prompts**
1. `Prompt: "In the beginning, God created the heavens and"` 
   **Output:** "the earth."
2. `Prompt: "The covenant between God and Abraham was a promise that"` 
   **Output:** "God would make him the father of many nations and give his descendants the land of Canaan."
3. `Prompt: "Complete this sentence: The shepherd leaves the ninety-nine to"` 
   **Output:** "find the one lost sheep."
4. `Prompt: "Write a one-sentence proverb about patience."` 
   **Output:** "Patience is a bitter plant that produces sweet fruit."
5. `Prompt: "What does the word 'covenant' mean in the context of the Old Testament?"` 
   **Output:** "A covenant is a solemn, binding agreement or promise between God and His people, establishing a relationship with mutual responsibilities."

**Part B — Language Variation (Pidgin English)**
1. `Prompt: "Complete this phrase in Pidgin: Na so God make heaven and"` 
   **Output:** "earth begin."
2. `Prompt: "Translate 'patience is a virtue' to Cameroonian Pidgin English."` 
   **Output:** "Small small catch monkey."
3. `Prompt: "Translate 'The shepherd leaves the ninety-nine to find the lost sheep' to Pidgin."` 
   **Output:** "The shepherd go leave the ninety-nine sheep make e go find the one wey lost."

**Part C — Observation Journal**
The model adapts remarkably well to the requested register, smoothly switching from theological standard English to West African Pidgin structure. It is very convincing with standard religious texts because those are heavily represented in internet training data. However, the Pidgin translations sometimes default to generic West African (Nigerian) Pidgin or standard English structure layered with slang, revealing its training data limitations regarding specific regional dialects like Cameroonian Pidgin. This tells me the LLM is heavily biased towards High-Resource languages and dominant internet cultures, leading to occasional degradation in low-resource vernaculars.

## Exercise E01.3 — The Reflection Journal: Day 1 Entry
1. **Mental Model:** A language model is fundamentally an advanced statistical pattern-matching engine. It doesn't actually "know" or "think" about what it writes; instead, it has processed billions of examples of human language to mathematically predict the most likely next word or token based on sequence probabilities from the prompt.
2. **Cultural Impact:** Training a model on Banso text isn't just about technical representation; culturally, it means preserving the language digitally and programmatically allowing the linguistic structure and idioms of the culture to interact with modern AI. It asserts visibility and viability, preventing digital extinction for the dialect.
3. **End Goal:** By Day 60, I want to be able to build and fine-tune a small AI application model that can generate an authentically structured text block in Lamnso or Cameroonian Pidgin without sounding unnaturally machine-translated.
