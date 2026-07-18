# MarkGPT: 60-Day LLM Curriculum Roadmap

## Overview

This roadmap outlines the 9-module, 60-day curriculum for learning deep learning from first principles and building modern language models.

**Time commitment**: 2-3 hours/day for 60 days
**Prerequisite**: Python basics (if-loops, functions, lists)
**Goal**: Build a GPT-style LM trained on King James Bible + Lamnso' language

## Macro Timeline

```
Week 1-2:  Module 1-2 (Foundations)
Week 3-4:  Module 3-4 (Neural Networks)
Week 5-6:  Module 5-6 (Embeddings & Attention)
Week 7-8:  Module 7-8 (Transformers & Transfer Learning)
Week 9:    Module 9 (Language & Culture)
Week 10:   Capstone Project
```

## Detailed Module Breakdown

### Module 1 - Orientation (Days 1-3) [COMPLETED]

**What you'll learn:**
- Course philosophy: "why before how"
- Deep learning landscape
- GPT family (GPT-2, GPT-3, ChatGPT)
- Banso language background
- Setup development environment

**Key deliverable:** Passing `scripts/verify_setup.py`

### Module 2 - Python for ML (Days 4-7)

**What you'll learn:**
- NumPy vectorization (eliminate loops!)
- Pandas for data manipulation
- Matplotlib visualization
- Batch processing

**Curriculum:**
- **Lesson L02.1**: Python for Machine Learning
- **Exercise day02**: NumPy vectorization from scratch
- **Project**: Normalize the King James Bible corpus

**Checkpoint:** Load Bible, compute statistics, plot word frequency

---

### Module 3 - Neural Networks (Days 8-12)

**What you'll learn:**
- Perceptron to deep learning
- Backpropagation from scratch
- Activation functions (ReLU vs Sigmoid)
- Loss functions (MSE, cross-entropy)

**Curriculum:**
- **Lesson L03.1**: Neural Networks from First Principles
- **Lesson L03.2**: Backpropagation Deep Dive
- **Exercise day03**: Implement backprop for 2-layer network
- **Project**: Solve XOR with a simple network

**Checkpoint:** Train network to solve XOR problem (loss < 0.01)

---

### Module 4 - Recurrent Networks (Days 13-16)

**What you'll learn:**
- RNN cells and sequence processing
- Backpropagation through time (BPTT)
- Vanishing gradient problem
- LSTM and GRU gating mechanisms

**Curriculum:**
- **Lesson L04.1**: Recurrent Neural Networks
- **Lesson L04.2**: LSTMs and GRUs
- **Exercise day04**: Implement RNN, understand gradient flow
- **Project**: LSTM language model on Bible (next-character prediction)

**Checkpoint:** RNN generates coherent Bible text (perplexity < 10)

---

### Module 5 - Tokenization & Embeddings (Days 17-21)

**What you'll learn:**
- Byte-Pair Encoding (BPE) algorithm
- Word2Vec and GloVe embeddings
- Tokenization for multilingual text
- Banso-specific preprocessing

**Curriculum:**
- **Lesson L05.1**: Word Embeddings - Meaning in Vectors
- **Lesson L05.2**: Character-Level & Subword Embeddings
- **Exercise day05**: Implement BPE tokenizer
- **Project**: Train Word2Vec on Bible, analyze semantic space

**Checkpoint:** Tokenizer achieves 1.5-2.0 tokens/word on Bible

---

### Module 6 - Attention & Multi-Head Attention (Days 22-26)

**What you'll learn:**
- Scaled dot-product attention
- Multi-head attention mechanism
- Causal masking for language models
- Attention visualization

**Curriculum:**
- **Lesson L06.1**: The Attention Mechanism
- **Lesson L06.2**: Attention in Transformers
- **Exercise day06**: Implement attention from scratch
- **Project**: Visualize attention patterns, understand which words it attends to

**Checkpoint:** Attention correctly attends only to past tokens (causal mask working)

---

### Module 7 - Transformers (Days 27-33)

**What you'll learn:**
- Full Transformer architecture
- Pre-norm vs post-norm
- Positional encodings (learned vs sinusoidal)
- Scaling laws for LLMs

**Curriculum:**
- **Lesson L07.1**: Transformer Architecture
- **Lesson L07.2**: Training Transformers at Scale
- **Exercise day07**: Train transformer LM on synthetic data
- **Project**: Train MarkGPT-Nano on Bible, measure scaling laws

**Checkpoint:** MarkGPT-Nano trains without NaN loss, validation < 4.5

---

### Module 8 - Transfer Learning & LoRA (Days 34-38)

**What you'll learn:**
- Pre-training vs fine-tuning
- LoRA (Low-Rank Adaptation)
- Parameter-efficient adaptation
- Multi-task learning

**Curriculum:**
- **Lesson L08.1**: Transfer Learning and LoRA
- **Lesson L08.2**: Multilingual LLMs
- **Exercise day08**: Implement LoRA adapter
- **Project**: Fine-tune MarkGPT on Banso using LoRA (1% of parameters!)

**Checkpoint:** LoRA fine-tuning 10x faster than full fine-tuning, <5% quality loss

---

### Module 9 - Language, Culture, Data (Days 39-43)

**What you'll learn:**
- Building datasets for under-resourced languages
- Parallel corpus creation
- Language preservation vs. replacement
- Ethics of AI for minority communities

**Curriculum:**
- **Lesson L09.1**: Language, Culture, and Data
- **Lesson L09.2**: Dataset Ethics & Community Partnership
- **Reading**: Nettle & Romaine (2000), *Vanishing Voices*
- **Project**: Create Banso-English parallel corpus, document linguistic features

**Checkpoint:** Contribute Banso data to public archive, write documentation

---

### Capstone Project (Days 44-60)

**Objective:** Apply everything to build end-to-end MarkGPT

**Phases:**
1. **Setup** (Days 44-45): Environment, data download
2. **Data prep** (Days 46-47): Preprocess Bible + Banso, create tokenizer
3. **Train** (Days 48-54): Train MarkGPT-Nano, then Small
4. **Fine-tune** (Days 55-57): LoRA fine-tuning on Banso
5. **Deploy** (Days 58-60): API, web UI, model card, submission

**Deliverables:**
- [ ] Trained models (checkpoints saved)
- [ ] Code on GitHub with README & examples
- [ ] Training report (loss curves, metrics, hyperparameters)
- [ ] Generated samples (English & Banso)
- [ ] Model card (Hugging Face format)
- [ ] Reflection essay (2-3 pages): What did you learn? How will you use this?

**Stretch goals:**
- Deploy model on Hugging Face Model Hub
- Build web app (Gradio/Streamlit)
- Compare multiple architectures (RNN vs Transformer)
- Add language detection

---

## Tools & Environments

### Required

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (or CPU, but slower)

### Optional

- Jupyter Notebook (for exploration)
- Weights & Biases (experiment tracking)
- Tensorboard (visualization)
- VS Code + Python extension

### Hardware

**Minimum:** CPU (slow but works)  
**Recommended:** GPU with 4GB+ VRAM (RTX 2060, RTX 3060, M1/M2)  
**Ideal:** GPU with 8GB+ VRAM (RTX 3070, RTX 4080)

---

## Learning Path Flowchart

```
START
  ↓
Module 1: Orientation
  ↓
Module 2: Python Fundamentals
  ↓
Module 3-4: Neural Networks & RNNs
  ↓
Module 5: Embeddings & Tokenization ← KEY BOTTLENECK
  ↓
Module 6: Attention (pre-req for transformers)
  ↓
Module 7: Transformers (heart of modern LLMs)
  ↓
Module 8: Transfer Learning & LoRA
  ↓
Module 9: Ethics & Language Considerations
  ↓
Capstone: Build End-to-End System
  ↓
Reflection & Celebration 🎉
```

---

## Prerequisites & Co-requisites

### Required Before Starting

- [ ] Python basics (loops, functions, lists, dictionaries)
- [ ] Linear algebra basics (matrices, dot product)
- [ ] Calculus basics (derivatives, chain rule)
- [ ] 2-3 hours/day for 60 days

### Optional, But Helpful

- [ ] Some ML/deep learning exposure (Andrew Ng's course)
- [ ] Git version control (learn as you go)
- [ ] Command line tools (learn as you go)

---

## Success Criteria

### By End of Course

- [ ] Understand how LLMs work from first principles
- [ ] Can implement all components (attention, positives, etc.)
- [ ] Train and fine-tune models independently
- [ ] Deploy a working language model
- [ ] Contribute meaningfully to ai4africa/multilingual-AI efforts

### Metrics

- **Knowledge**: Pass all module quizzes (80%+)
- **Implementation**: All exercises run without errors
- **Project**: Capstone project complete and deployed
- **Reflection**: Articulate what you learned and how it matters

---

## Community & Support

- **Slack channel**: #markgpt-curriculum (coming soon)
- **GitHub discussions**: Post questions, share projects
- **Office hours**: Weekly online sessions (UTC/EST timezones)
- **Peer learning**: Find study buddy from course

---

## What Comes After?

After this 60-day curriculum, you can:

1. **Research**: Implement cutting-edge papers (flash attention, rotary embeddings, etc.)
2. **Industry**: Join ML teams at startups/tech companies
3. **Open Source**: Contribute to PyTorch, Hugging Face, etc.
4. **Teaching**: Teach this curriculum to others
5. **Specialization**: 
   - Multilingual LLMs
   - Low-resource language NLP
   - LLM safety/alignment
   - Efficient inference

---

## Frequently Asked Questions

**Q: Can I skip modules?**  
A: Not recommended, but if you have ML background, can skip Module 1-2 and skim 3-4.

**Q: What if I'm slower than 60 days?**  
A: That's fine! This is self-paced. 90-120 days is also great.

**Q: Can I do this without a GPU?**  
A: Yes, but expect 10-20x slower training. Use smaller models (Nano).

**Q: What if I get stuck?**  
A: See FAQ.md and troubleshooting guide. Post in GitHub discussions.

---

## Resources & References

### Papers (in order of appearance)

1. Goodfellow et al. (2016). *Deep Learning*. MIT Press.
2. Hochreiter & Schmidhuber (1997). "Long Short-Term Memory." *Neural Computation*.
3. Mikolov et al. (2013). "Efficient Estimation of Word Representations." *ICLR*.
4. Vaswani et al. (2017). "Attention is All You Need." *NeurIPS*.
5. Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.

### Online Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Hugging Face Transformers Library](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Anthropic AI Safety Alignment](https://www.anthropic.com/research)

---

## Contributing to MarkGPT

See CONTRIBUTING.md for guidelines on:
- Submitting bug reports
- Proposing new modules
- Improving lessons/exercises
- Translating curriculum

---

**Start your journey today!** 🚀

Next step: Check out [GETTING_STARTED.md](../GETTING_STARTED.md)
