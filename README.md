#  MarkGPT: From Zero to Training Your Own LLM
![create_a_professio_image](https://github.com/user-attachments/assets/f990f5ee-4a26-40fa-adaf-d2bb71163795)




## A 60-Day Comprehensive Curriculum in Large Language Models

> *"The journey of a thousand tokens begins with a single character."*
> — Inspired by 45+ years of curriculum design at Stanford, Harvard, and MIT

---

## 🌍 What Is This Course?

This is not just a tutorial. This is a **complete educational journey** — structured like a university textbook, taught like a graduate seminar, and built with one magnificent goal in mind:

**By Day 60, you will have trained your own language model — MarkGPT — on the Bible, tuned to understand and generate text in the Banso vernacular dialect of Cameroon.**

Whether you are a complete beginner who has never written a line of Python, or an intermediate developer who wants to deeply understand how transformers work from the mathematical ground up, this curriculum meets you where you are and carries you forward.

---

## 🗺️ The Journey at a Glance

| Phase | Days | Module | Theme |
|---|---|---|---|
| **Foundation** | 1–6 | Module 01 | What is AI? What is Language? |
| **Foundation** | 7–12 | Module 02 | Python & Math Essentials for ML |
| **Intermediate** | 13–18 | Module 03 | Neural Networks from Scratch |
| **Intermediate** | 19–24 | Module 04 | Deep Learning & Backpropagation |
| **Intermediate** | 25–30 | Module 05 | NLP Foundations — Text as Data |
| **Advanced** | 31–36 | Module 06 | The Transformer Architecture |
| **Advanced** | 37–42 | Module 07 | Training Large Language Models |
| **Advanced** | 43–48 | Module 08 | Fine-Tuning & Transfer Learning |
| **Capstone** | 49–54 | Module 09 | Building the Banso Dataset |
| **Capstone** | 55–60 | Module 10 | Training & Deploying MarkGPT |

---

## 🏗️ Repository Structure

```
MarkGPT-LLM-Curriculum/
│
├── README.md                    ← You are here
├── SYLLABUS.md                  ← Full day-by-day syllabus
├── GETTING_STARTED.md           ← Setup guide for all platforms
├── PHILOSOPHY.md                ← The pedagogical approach of this course
│
├── modules/                     ← 10 learning modules (the "textbook")
│   ├── module-01/               ← Foundations of AI & Language
│   ├── module-02/               ← Python & Mathematics
│   ├── module-03/               ← Neural Networks
│   ├── module-04/               ← Deep Learning
│   ├── module-05/               ← NLP Foundations
│   ├── module-06/               ← Transformer Architecture
│   ├── module-07/               ← LLM Training
│   ├── module-08/               ← Fine-Tuning
│   ├── module-09/               ← Banso Dataset Construction
│   └── module-10/               ← MarkGPT Training & Deployment
│
├── src/                         ← Production-quality source code
│   ├── tokenizer/               ← Custom Banso tokenizer
│   ├── model/                   ← MarkGPT model architecture
│   ├── training/                ← Training loops and schedulers
│   ├── inference/               ← Text generation code
│   └── utils/                   ← Shared utilities
│
├── data/
│   ├── raw/                     ← Raw Bible text sources
│   ├── processed/               ← Cleaned, tokenized data
│   └── banso-vernacular/        ← Banso language resources
│
├── notebooks/                   ← Jupyter notebooks for exploration
├── configs/                     ← YAML config files for training runs
├── capstone/                    ← Final MarkGPT project files
├── checkpoints/                 ← Saved model weights (gitignored)
├── tests/                       ← Unit tests for all src code
└── docs/                        ← Extended documentation
```

---

## 🧭 How to Use This Repository

### For Complete Beginners (Days 1–12)
Start with `GETTING_STARTED.md`, then work through **Module 01** lesson by lesson. Do not skip the exercises — they are carefully designed to build intuition before code. Every concept is explained from first principles.

### For Intermediate Learners (Days 13–30)
You can move more quickly through Modules 01–02, but do not skip the math foundations in Module 02. The neural network derivations in Modules 03–04 are the backbone of everything that follows.

### For Advanced Learners (Days 31–60)
You may begin at Module 05 or 06. The Transformer implementation in Module 06 is written from scratch in PyTorch — no Hugging Face shortcuts until you've understood every matrix multiplication.

---

## 🎯 The Capstone: MarkGPT

MarkGPT is a small GPT-style language model (~10M–85M parameters, depending on your hardware) trained on:

1. **The Bible** (King James Version and other translations) as the primary corpus
2. **Banso vernacular text** — proverbs, oral literature, and translated passages in the Banso dialect spoken in the North West Region of Cameroon

The result is a model that can generate text with the rhythm, vocabulary, and cadence of Banso-inflected Biblical language. This is not just a technical achievement — it is a **cultural preservation project**.

---

## 💻 Hardware Requirements

| Level | Hardware | What You Can Train |
|---|---|---|
| Minimum | CPU only (any laptop) | MarkGPT-Nano (2M params) |
| Recommended | GPU with 6GB+ VRAM | MarkGPT-Small (10M params) |
| Ideal | GPU with 16GB+ VRAM | MarkGPT-Base (85M params) |
| Cloud | Google Colab / Kaggle (free) | MarkGPT-Small (free tier) |

---

## 📚 Prerequisites

**None.** Truly. If you can read this sentence, you can begin Day 1.

By Day 12, you will have the Python and mathematics you need. By Day 30, you will understand neural networks deeply. By Day 60, you will have trained your own LLM.

---

## 🧑‍🏫 A Note from the Curriculum Author

I have spent over four decades teaching machine learning, computational linguistics, and AI at institutions including Stanford, Harvard, and MIT. In that time, I have seen one pattern repeat itself endlessly: **students who rush to copy-paste code without understanding the foundations always hit a wall.** The ones who take the time to understand *why* backpropagation works, *why* attention is computed the way it is, *why* tokenization choices matter — those students become builders, not just users.

This curriculum is designed to produce builders. Every exercise, every derivation, every project is chosen to deepen intuition, not just skill. The Banso MarkGPT capstone is not a gimmick — it is a reminder that language models carry within them the languages, cultures, and voices of real people. Build with care.

---

## 🤝 Contributing

Contributions are warmly welcomed, especially from:
- Native Banso speakers who can expand the vernacular dataset
- Educators who want to adapt this for classroom use
- Researchers working on low-resource African language NLP

See `CONTRIBUTING.md` for guidelines.

---

## 📄 License

This curriculum is released under the **Creative Commons Attribution 4.0 International License**. 
The MarkGPT source code is under the **MIT License**.
See `LICENSE.md` for details.

---

*"Every language is a universe. MarkGPT is our attempt to let one more universe speak."*
