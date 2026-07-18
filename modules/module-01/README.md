# Module 01 — Foundations of AI and Language
## Days 1–6 | Beginner Level

---

## Module Overview

Welcome to the beginning of your journey. Module 01 asks the oldest questions in artificial intelligence and answers them carefully, from the ground up, without shortcuts.

You will not write a transformer this week. You will not fine-tune a pretrained model. What you will do is far more important: you will build the conceptual architecture that makes everything else in this curriculum legible. Students who rush Module 01 often find themselves confused by Module 06. Students who take Module 01 seriously find that Module 06 almost explains itself.

By the end of Day 6, you will understand what language models actually are (probability machines over sequences), why the history of AI follows the arc it does (the fundamental tension between rule-based and data-driven approaches), how text becomes numbers in a computer (encoding, vocabulary, tokenization basics), and what information theory has to do with learning (entropy, cross-entropy, perplexity). You will also have built your first working language model — BansoGram, a trigram model trained on biblical and Banso text.

---

## Lessons in This Module

All lessons are in the `lessons/` folder of this module. Read them in order; each one builds on the last.

`L01.1_history_of_ai.md` — From Turing to transformers: the story of how AI got here, why the Transformer was a genuine revolution, and where MarkGPT fits in that story. This is the narrative that gives the rest of the course meaning.

`L01.2_what_is_a_language_model.md` — What does it mean to model language? How is next-token prediction related to understanding? What is the difference between a character-level model and a word-level model, and when does each make more sense?

`L01.3_markgpt_in_context.md` — How does MarkGPT compare to GPT-2, GPT-3, and Claude? What can a 10M-parameter model do well, and what should you realistically expect from it? This lesson sets expectations honestly.

`L01.4_setup.md` — Setting up your Python environment step by step, with troubleshooting guidance for common issues on each operating system.

`L02.1_language_as_data.md` — Characters, words, sentences: the units of language and how they map to numerical representations. ASCII, Unicode, and why encoding matters for multilingual models.

`L02.2_banso_language_intro.md` — An introduction to Lamnso' (Banso), its linguistic features, its cultural context, and why modeling it in an AI system is both technically interesting and culturally meaningful.

`L03.1_math_you_already_know.md` — A reassuring tour of the mathematics in this curriculum: vectors, matrices, functions, probability. The emphasis is on building geometric intuition, not on formal proofs.

`L04.1_probability_and_information.md` — Conditional probability, the chain rule, Shannon entropy, cross-entropy loss, and perplexity. These concepts form the mathematical backbone of language model training and evaluation.

`L05.1_ngram_models.md` — Unigram, bigram, and trigram models: how statistical language models work, what they can and cannot do, and why their failures pointed researchers toward neural networks.

---

## Exercises in This Module

All exercises are in the `exercises/` folder. For each day, there is one exercise file. Exercises are a mix of conceptual (paper and pen), programming, and written reflection. All three types matter.

Day 1: AI history drawing exercise, hands-on exploration of a language model, reflection journal
Day 2: Encoding exploration in Python, language representation research
Day 3: Linear algebra by hand and in code, function plotting
Day 4: Entropy calculations, bigram model from scratch on Genesis
Day 5: Trigram model, Laplace smoothing, Chomsky essay exercise
Day 6: Mini-Project 1 — BansoGram

---

## Mini-Project 1: BansoGram

The module culminates in your first real project. BansoGram is a bigram/trigram language model that you will build from scratch in Python, without any machine learning libraries. It will accept a seed word or phrase and generate a continuation of 20–50 words, and it will report its perplexity on held-out text.

The project has a specific cultural dimension: you will train two versions — one on English KJV text, one on available Banso-inflected text — and compare their outputs. The comparison will be your first concrete lesson in how the training data shapes a model's "voice."

Starter template: `exercises/mini_project_1_banso_gram_starter.py`  
Expected output: A file `projects/banso_gram.py` with your completed model and a `reflection_day06.md` with your written analysis.

---

## Resources for This Module

The reference list for the full curriculum is in `docs/REFERENCES_AND_RESOURCES.md`. For Module 01 specifically, the most important starting resources are Jurafsky & Martin's *Speech and Language Processing* (Chapters 1 and 3), 3Blue1Brown's Essence of Linear Algebra video series, and the free online book *Neural Networks and Deep Learning* by Michael Nielsen. All are free; links are in the references document.

---

## A Note on Pacing

Module 01 covers six days at approximately two hours per day. If you have no prior programming experience, the setup on Day 4 (writing a bigram model in Python) may take longer. That is fine. The goal is not to check boxes — it is to build understanding that compounds over the next 54 days.

If you finish early, the bonus challenges in each exercise file will extend your thinking. If you fall behind, the mini-project can be scoped down: a working bigram model on English only is perfectly acceptable. The Banso comparison is an enrichment, not a requirement.

You are ready. Begin with Lesson 1.1.
