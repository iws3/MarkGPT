# The Banso/Lamnso' Language — Dataset Guide
## Module 09 Reference Document

---

## Who Are the Banso People?

The Banso people — also known as the Nso' — inhabit the highlands of the North West Region of Cameroon, centered on the town of Kumbo (Nso'). Their kingdom, the Nso' Fondom, is one of the largest traditional kingdoms in Cameroon and has maintained an unbroken line of Fons (rulers) for centuries.

Lamnso' (literally "the language of Nso'") is a Grassfields Bantu language in the Niger-Congo family. It is spoken by approximately 400,000–800,000 people across the Bui Division and diaspora communities worldwide. Like all Grassfields languages, it is tonal — meaning the pitch at which a syllable is pronounced changes its meaning. This tonal structure is one of the reasons building a language model for Lamnso' is linguistically fascinating.

Understanding who speaks the language you are modeling is not optional context — it is the foundation of ethical and effective NLP for minority languages.

---

## Linguistic Features of Lamnso'

Several features make Lamnso' structurally different from English in ways that matter for language model design.

**Noun class system.** Lamnso', like most Bantu languages, organizes nouns into classes (called "genders") marked by prefixes. A word's class affects agreement patterns across the sentence — verbs, adjectives, and pronouns all carry class agreement markers. This means that memorizing word-level patterns is not enough; a model must also learn the grammatical agreement patterns.

**Tonal contrasts.** Lamnso' uses three phonemic tones: high (marked with an acute accent: á), low (unmarked or marked with a grave: à), and falling (marked with a circumflex: â). Tonal distinctions can change a word's meaning entirely — for example, distinguishing a noun from a verb, or differentiating two completely unrelated words. Our training corpus has inconsistent tonal marking, which means MarkGPT's current version does not learn tonal contrasts. This is a known limitation.

**Verb-final tendency.** While Lamnso' allows some flexibility, it tends toward Subject-Object-Verb order in certain constructions, unlike English's Subject-Verb-Object order. A language model trained heavily on English may resist this order — which is another reason why a Banso-specific training corpus matters.

**Code-switching.** Most Banso speakers also speak Cameroonian Pidgin English, French, and/or standard English. In natural speech and written text, code-switching between Lamnso' and these other languages is common. Our training data reflects this reality, and the `<|bn|>` and `<|en|>` language tags in the tokenizer help the model learn to track which language is being used.

---

## Available Resources for the Banso Dataset

**Primary Bible sources:**
- SIL International has produced portions of the Bible in Lamnso'. The New Testament (Ndɨ Tev Yiv Nso') is available through the SIL/UBS archives. Contact information and access procedures are documented in `data/banso-vernacular/sources.md`.
- The Lamnso' Gospel of John is available through JAARS/Wycliffe archives.

**Lexical resources:**
- Fresco, M.O. (1978). "Nso' Grammar Outline" — a structural description of Lamnso' including nominal classes and verb morphology.
- Kilham, C. et al. — *Lamnso' Literacy Primer* and word lists (SIL Cameroon)
- The Lamnso' language entries in the Ethnologue database (ethnologue.com, language code: "lns")

**Digital text:**
- The JW.org library contains some Lamnso' publications in open-access format.
- Community-contributed text through this curriculum's data collection initiative.
- Oral literature (proverbs, riddles, narratives) collected by University of Yaoundé I researchers.

**Academic references:**
- Yuka, L. (2011). "Aspects of Nso' Grammar"
- Hombert, J.M. (1976). "Consonant Types, Vowel Height, and Tone in Bantu Languages" — contextualizes the tonal system
- Chumbow, B.S. (1982). "Language and Language Policy in Cameroon" — for understanding the sociolinguistic context

---

## How to Contribute Banso Text

If you are a native Lamnso' speaker or have access to Lamnso' text, your contribution can materially improve MarkGPT's representation of the language. Contributions are especially valuable from:

- Banso diaspora communities (UK, USA, Canada, Germany)
- Students and alumni of Sacred Heart College Mankon and other schools in Kumbo
- Church communities using Lamnso' in worship
- Anyone with digitized personal letters, stories, or proverbs in Lamnso'

Guidelines for contributing text are in `CONTRIBUTING.md`. All contributions will be credited in the dataset documentation and will be made available under a community-agreed license.

---

## A Note on What "Representing a Language" Means

When we train a language model on Lamnso' text, we are encoding patterns of the language — its vocabulary, its grammar, its idioms, its way of connecting ideas. This is both a technical act and a cultural act.

A language model is not a community. It cannot replace the wisdom that lives in fluent speakers, the humor that requires cultural context to appreciate, or the lived experience that gives words their weight. What a language model can do is make the language visible in AI systems, create tools that speakers can use in their own language, and preserve patterns that might otherwise fade as younger generations shift to major languages under economic pressure.

The Banso community is the owner of Lamnso'. This curriculum's role is to provide the technical tools. The community's role is to decide how those tools are used.

---

## Proverbs for Training Data (Sample)

Below are ten Nso' proverbs in Lamnso' with English translations, illustrating the richness of the oral tradition available for the training corpus. These are drawn from publicly documented sources.

*"Wvisi a yi nggikii, yooni a ta yi nggikii wvisi."*  
— "The elder who fails to teach the child will be taught by the child."  
(On the reciprocity of wisdom and the importance of education)

*"Mbii i ta tee sam, sam i ta tee mbii."*  
— "The mouth will betray a person; a person will betray their mouth."  
(On the consequences of speech)

*"Ntεm i yi nggaaŋ, nggaaŋ i yi ntεm."*  
— "The heart is the house; the house is the heart."  
(On the relationship between home, belonging, and spirit)

These proverbs illustrate exactly the kind of parallel, rhythmically balanced structure that makes Lamnso' oral literature so distinctive — and that MarkGPT, at its best, should be able to learn and reproduce.

---

*Module 09 exercises: see `modules/module-09/exercises/`*
*Data download scripts: see `scripts/download_banso_data.py`*
*Datasheet template: see `data/banso-vernacular/DATASHEET_TEMPLATE.md`*
