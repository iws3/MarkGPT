# Lesson L01.3: How MarkGPT Fits Into the AI Landscape
## Day 1: Welcome & Orientation | Positioning Our Custom LLM in the Broader AI Ecosystem

### Lesson Overview
Now that you understand what a language model is, you need to understand the landscape of models that exist and where MarkGPT sits within it. From billion-parameter powerhouses like GPT-4 to tiny models that can run on a phone, the AI ecosystem is vast and stratified. MarkGPT will be deliberately small (2M-85M parameters) and deliberately multilingual (English + Banso). This lesson maps the terrain, so that when you train MarkGPT, you understand exactly what tradeoffs you are making and why they matter for your specific goal: preserving and demonstrating the power of language technology for underrepresented languages.

## Table of Contents
- Overview of the AI Landscape
- Commercial LLMs: GPT, Claude, Gemini
- Open-Source Alternatives
- Specialized Models
- MarkGPT's Unique Position
- Educational Value
- Technical Specifications
- Future Roadmap
- Conclusion

---

## Overview of the AI Landscape

The AI landscape is diverse, with models ranging from small specialized systems to massive general-purpose LLMs. Commercial models like GPT-4 dominate headlines, but there's a rich ecosystem of open-source and specialized models.

Understanding where MarkGPT fits requires examining the different categories of AI models and their capabilities.

---

## Commercial LLMs: GPT, Claude, Gemini

Commercial LLMs are developed by tech companies and offered as APIs or hosted services. They are trained on massive datasets and have billions of parameters.

- GPT series by OpenAI: Excellent at text generation, reasoning, and coding

- Claude by Anthropic: Strong safety features and long context windows

- Gemini by Google: Multimodal capabilities, integrating text, image, and code

These models set the benchmark for LLM capabilities but are proprietary and expensive to use at scale.

---

## Open-Source Alternatives

Open-source LLMs provide transparency and customization options. Models like Llama, Mistral, and Falcon are freely available for research and commercial use.

These models often lag behind commercial ones in performance but offer flexibility for fine-tuning and deployment. They are crucial for the democratization of AI.

---

## Specialized Models

Specialized models are designed for specific tasks:

- Code generation: GitHub Copilot, CodeLlama

- Medical: BioBERT, ClinicalBERT

- Legal: Legal-BERT

- Scientific: SciBERT

These models are fine-tuned on domain-specific data for better performance in their niches.

---

## MarkGPT's Unique Position

MarkGPT occupies a specific and important niche in the AI ecosystem: it is a **small, bilingual, open-source language model explicitly designed for educational purposes and for preserving underrepresented languages.**

Unlike commercial models, MarkGPT prioritizes transparency over scale. You will understand every line of code that trains it. You will read the papers that inspired its architecture. You will see exactly what data it learned from and what biases might lurk in its outputs.

Unlike many open-source models, MarkGPT is not designed to compete with GPT-4 on benchmarks. Instead, it is designed to do one thing exceptionally well: generate text with *personality* — specifically, the personality of Biblical English fused with the cultural and linguistic patterns of Banso. If you ask GPT-4 to write in the style of Banso proverbs, it will dutifully try. MarkGPT, trained on Banso + English data, will do it with indigenous knowledge, not borrowed patterns.

This is MarkGPT's value: **precision for a specific cultural-linguistic niche, achieved through smallness and specificity.**

---

## Technical Specifications (Overview)

MarkGPT will use:

- **Architecture:** Decoder-only Transformer (similar to GPT)
- **Parameters:** 2M-85M (configurable based on your hardware)
- **Training data:** KJV Bible + Banso-English parallel corpus
- **Tokenizer:** SentencePiece or BPE, trained on combined corpus
- **Framework:** PyTorch
- **Hardware:** CPU-trainable on small configs; GPU-recommended for faster iteration

Full hyperparameter details will be specified in Module 06 and Module 10.

---

## The Educational Value of Building MarkGPT

By building MarkGPT yourself, you gain:

1. **Deep understanding of LLM mechanics** — Not as an abstract concept, but through hands-on implementation.
2. **Practical ML engineering skills** — Data loading, training loops, hyperparameter tuning, debugging.
3. **Ethical framework** — You will confront questions about bias, data collection consent, and language preservation directly, not in theory.
4. **Ownership of the model** — You built it. You understand its strengths and limitations. You can modify it.
5. **A working example for future work** — MarkGPT can be the foundation for fine-tuning other Banso NLP tasks.

---

## Closing Reflection: Why Smallness is Strategic

You might ask: why build a small model when larger models are better? 

The answer is strategic clarity. **Smaller models force you to make every decision consciously.** You cannot hide behind scale. When something goes wrong with MarkGPT, you will debug it yourself and understand why. When it works well, you will see exactly which data, which training procedure, which architectural choice made the difference.

Moreover, smaller models are deployable. They can run on a researcher's laptop, on a phone, on humble hardware in communities where the technology matters most. A 10M-parameter MarkGPT that runs locally is arguably more valuable than access to a 175B-parameter model that lives in a distant cloud.

The AI landscape will continue to be dominated by large generalist models. But the frontier of *impact* — of actually making a difference in overlooked communities — belongs to smaller, specialized models like MarkGPT. 

*Next: Lesson L01.4 — Setting Up Your Learning Environment*