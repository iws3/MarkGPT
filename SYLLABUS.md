# 📅 SYLLABUS — MarkGPT 60-Day LLM Curriculum
## Complete Day-by-Day Learning Schedule

---

## MODULE 01 — Foundations: What Is AI? What Is Language?
### Days 1–6 | Beginner Level

---

### DAY 1 — Welcome & Orientation
**Theme:** What is artificial intelligence, really?

**Lessons:**
- `L01.1` — The History of AI: From Turing to Transformers (20-min read)
- `L01.2` — What Is a Language Model? (15-min read)
- `L01.3` — How MarkGPT Fits Into the AI Landscape (10-min read)
- `L01.4` — Setting Up Your Learning Environment (hands-on setup)

**Exercises:**
- `E01.1` — Draw the AI family tree: write out the lineage from symbolic AI → connectionism → deep learning → LLMs in your own words
- `E01.2` — Play with GPT-2 on Hugging Face Spaces. Type 5 prompts in English, then try 5 in any language you know. Document what you observe.
- `E01.3` — Reflection journal: "What do I want to understand about language models by Day 60?"

**Resources:**
- Turing, A. (1950). *Computing Machinery and Intelligence*
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444
- Wolfram, S. — *What Is ChatGPT Doing and Why Does It Work?* (2023, free online)

---

### DAY 2 — Language as Data
**Theme:** How do computers see text?

**Lessons:**
- `L02.1` — Characters, Words, Sentences: Units of Language
- `L02.2` — Encoding Language: ASCII, Unicode, and Beyond
- `L02.3` — The Banso Language: Phonology, Grammar, and Writing Systems
- `L02.4` — Why Low-Resource Languages Matter in AI

**Exercises:**
- `E02.1` — Open a text editor and type a sentence in English. Now count: how many characters? Words? Sentences? Now try to do the same analysis on a Banso proverb (examples provided in `data/banso-vernacular/proverbs.txt`)
- `E02.2` — Run `ord('A')` in Python. Explore the ASCII table. What number is 'a'? '0'? Space? 
- `E02.3` — Research task: Find 3 African languages that have been included in GPT-4's training data and 3 that have not. What are the differences in written resources available?

**Resources:**
- Abney, S. (1996). Statistical Methods and Linguistics. *The Balancing Act*
- Joshi, P. et al. (2020). The State and Fate of Linguistic Diversity and Inclusion in the NLP World. *ACL 2020*
- UNESCO Atlas of the World's Languages in Danger

---

### DAY 3 — Mathematics You Already Know
**Theme:** The math behind ML is not as scary as you think

**Lessons:**
- `L03.1` — Vectors and What They Really Mean (with visual intuition)
- `L03.2` — Matrices as Transformations
- `L03.3` — Functions, Derivatives, and the Idea of Learning
- `L03.4` — Probability Basics: Events, Distributions, and Surprise

**Exercises:**
- `E03.1` — Without a computer: multiply two 2×2 matrices by hand. Verify with numpy.
- `E03.2` — Plot the functions f(x) = x², f(x) = sin(x), and f(x) = 1/(1+e^-x) by hand. Then plot them in Python using matplotlib.
- `E03.3` — The probability puzzle: if a language model assigns probability 0.3 to "the", 0.1 to "dog", and 0.05 to "barked" — what is the probability of the sentence "the dog barked" under a naive independence assumption? What's wrong with that assumption?

**Resources:**
- 3Blue1Brown — *Essence of Linear Algebra* (YouTube series, free)
- 3Blue1Brown — *Essence of Calculus* (YouTube series, free)
- Gilbert Strang — *Introduction to Linear Algebra*, 5th ed. (MIT OpenCourseWare)

---

### DAY 4 — Probability and Information Theory
**Theme:** Language models are probability machines

**Lessons:**
- `L04.1` — Conditional Probability: P(next word | previous words)
- `L04.2` — The Chain Rule of Probability
- `L04.3` — Shannon Entropy: Measuring Surprise
- `L04.4` — Cross-Entropy Loss: How LLMs Learn
- `L04.5` — Perplexity: Measuring How Well a Model Knows Language

**Exercises:**
- `E04.1` — Calculate the entropy of a fair coin, a loaded coin (0.9/0.1), and a two-word "language" with equal probabilities. Which is most surprising?
- `E04.2` — Implement a toy bigram language model in Python from scratch (starter code in `exercises/`). Train it on Genesis Chapter 1.
- `E04.3` — Compute the perplexity of your bigram model. What does the number mean intuitively?

**Resources:**
- Shannon, C.E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*
- Cover, T.M. & Thomas, J.A. — *Elements of Information Theory*, 2nd ed.

---

### DAY 5 — N-Gram Models: The Pre-Neural Era
**Theme:** Language modeling before neural networks

**Lessons:**
- `L05.1` — Unigrams, Bigrams, Trigrams: Statistical Language Models
- `L05.2` — Smoothing Techniques: Laplace, Kneser-Ney
- `L05.3` — The Limits of N-Gram Models
- `L05.4` — Why These Limits Led to Neural Networks

**Exercises:**
- `E05.1` — Build a trigram model from scratch in Python on the Book of Genesis. Generate 50 words of "Bible-like" text. How does it compare to real Genesis?
- `E05.2` — Experiment with Laplace smoothing. What happens to your model's perplexity when you add smoothing? Why?
- `E05.3` — Written essay (200 words): "Why is the sentence 'Colorless green ideas sleep furiously' grammatical but nonsensical, and what does this teach us about n-gram models?"

**Resources:**
- Jurafsky, D. & Martin, J.H. — *Speech and Language Processing*, 3rd ed. Ch. 3 (free online)
- Chen, S. & Goodman, J. (1999). An Empirical Study of Smoothing Techniques for Language Modeling

---

### DAY 6 — Module 01 Review & Mini-Project
**Theme:** Consolidation and first hands-on build

**Lessons:**
- `L06.1` — Review: The Big Picture So Far
- `L06.2` — What Makes a Good Language Model? Criteria and Benchmarks

**Project (Mini-Project 1):**
Build `BansoGram` — a bigram/trigram language model trained on Banso proverbs and Bible excerpts. It should:
1. Accept a seed word or phrase in Banso/English
2. Generate a continuation of 20–50 words
3. Report its perplexity on a held-out test set
4. Compare outputs: model trained on English only vs. model trained on Banso-inflected text

**Submit:** `modules/module-01/projects/banso_gram.py` with a written reflection in `reflection_day06.md`

---

## MODULE 02 — Python & Mathematics Essentials
### Days 7–12 | Beginner–Intermediate

---

### DAY 7 — Python for Machine Learning
**Lessons:** Python data types for ML, NumPy arrays, vectorized operations, broadcasting
**Exercises:** Implement matrix multiply without numpy, then benchmark numpy's version. Implement softmax from scratch.
**Resources:** fast.ai — Practical Deep Learning for Coders (Lesson 1); Python Data Science Handbook (VanderPlas, free online)

---

### DAY 8 — Data Manipulation with Pandas & Visualization
**Lessons:** DataFrames for NLP data, text cleaning pipelines, matplotlib & seaborn for loss curves
**Exercises:** Load the KJV Bible CSV, compute word frequency distributions, plot a Zipf's Law curve, clean and tokenize the text
**Resources:** Pandas documentation; Wes McKinney — *Python for Data Analysis*

---

### DAY 9 — Linear Algebra Deep Dive
**Lessons:** Matrix operations, eigenvalues (conceptual), SVD and word embeddings, dot products as similarity measures
**Exercises:** Implement PCA from scratch using numpy. Visualize word co-occurrence vectors in 2D. What clusters naturally?
**Resources:** Gilbert Strang lectures (MIT OCW 18.06); *Mathematics for Machine Learning* (Deisenroth et al., free PDF)

---

### DAY 10 — Calculus for Optimization
**Lessons:** Derivatives, the chain rule (crucial for backprop), gradient descent intuition, local vs. global minima
**Exercises:** Minimize f(x) = x⁴ - 4x² + x using gradient descent. Plot the trajectory. Try different learning rates. Which diverges? Which converges slowly?
**Resources:** Karpathy, A. — *Yes you should understand backprop* (Medium post); 3Blue1Brown Calculus series

---

### DAY 11 — Statistics & Probability for NLP
**Lessons:** Bayes' theorem in NLP, maximum likelihood estimation, the Gaussian distribution, sampling methods
**Exercises:** Implement a Naive Bayes text classifier from scratch. Train on KJV vs. non-biblical text. Evaluate accuracy.
**Resources:** Bishop, C.M. — *Pattern Recognition and Machine Learning*, Ch. 1–2; *Bayesian Reasoning and Machine Learning* (Barber, free online)

---

### DAY 12 — Module 02 Review & Environment Setup
**Lessons:** Setting up GPU environments (Colab, local), PyTorch basics, the tensor abstraction
**Exercises:** Install PyTorch, create tensors, perform gradient computation with autograd. Train a single neuron.
**Project (Mini-Project 2):** A complete text preprocessing pipeline for the Bible + Banso data, producing clean, tokenized, split datasets ready for training.

---

## MODULE 03 — Neural Networks from Scratch
### Days 13–18 | Intermediate

---

### DAY 13 — The Neuron: Biology to Mathematics
**Lessons:** The biological neuron, the McCulloch-Pitts model, activation functions (sigmoid, ReLU, tanh, GELU), forward pass
**Exercises:** Implement a single neuron from scratch. Train it to learn the AND gate, then OR gate, then XOR (observe the failure). Why does XOR fail?
**Resources:** McCulloch, W. & Pitts, W. (1943). A Logical Calculus of Ideas Immanent in Nervous Activity; Rosenblatt, F. (1958). The Perceptron

---

### DAY 14 — Multi-Layer Perceptrons
**Lessons:** Hidden layers, universal approximation theorem, forward pass through multiple layers, the representational power of depth
**Exercises:** Implement a 2-layer MLP. Solve XOR. Visualize the learned decision boundary. Add a third layer — what changes?
**Resources:** Cybenko, G. (1989). Approximation by Superpositions of a Sigmoidal Function; Nielsen, M. — *Neural Networks and Deep Learning* (free online)

---

### DAY 15 — Backpropagation: The Heart of Learning
**Lessons:** The computation graph, automatic differentiation, the chain rule applied to networks, gradient flow
**Exercises:** Derive the gradients for a 2-layer MLP by hand on paper. Verify your derivation against PyTorch autograd. Implement backprop from scratch without autograd.
**Resources:** Rumelhart, D., Hinton, G. & Williams, R. (1986). Learning representations by back-propagating errors. *Nature*; Karpathy, A. — *micrograd* GitHub

---

### DAY 16 — Loss Functions & Optimization
**Lessons:** MSE, cross-entropy, negative log-likelihood, SGD, momentum, Adam optimizer
**Exercises:** Compare SGD, SGD+momentum, and Adam on the same problem. Plot loss curves. Implement Adam from scratch.
**Resources:** Kingma, D. & Ba, J. (2014). Adam: A Method for Stochastic Optimization; Goodfellow et al. — *Deep Learning* (free online), Ch. 8

---

### DAY 17 — Regularization & Generalization
**Lessons:** Overfitting and underfitting, dropout, weight decay, batch normalization, the bias-variance tradeoff
**Exercises:** Intentionally overfit a model on 100 Bible verses. Then apply dropout and weight decay. Measure train vs. validation perplexity before and after.
**Resources:** Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting; Ioffe & Szegedy (2015). Batch Normalization

---

### DAY 18 — Module 03 Review & Mini-Project
**Project (Mini-Project 3):** Build a character-level language model using an MLP trained on Psalm 23 in both English and Banso translation. The model should generate plausible continuations. Compare the character-level outputs between the two languages.

---

## MODULE 04 — Recurrent Networks & Sequence Modeling
### Days 19–24 | Intermediate

---

### DAY 19 — Sequences and the Problem of Memory
**Lessons:** Why feedforward networks can't handle sequences natively, the concept of hidden state, unrolling through time
**Exercises:** Implement a "manual" sequence processor: feed characters one at a time into an MLP, concatenating hidden state. Observe how fast information is lost.

---

### DAY 20 — Recurrent Neural Networks (RNNs)
**Lessons:** The RNN cell, BPTT (backprop through time), the vanishing gradient problem, gradient clipping
**Exercises:** Train an RNN from scratch on the Book of John. Generate text. Observe the repetition and incoherence. Why does this happen?
**Resources:** Elman, J.L. (1990). Finding Structure in Time; Hochreiter, S. (1991). Untersuchungen zu dynamischen neuronalen Netzen (the vanishing gradient thesis)

---

### DAY 21 — LSTMs and GRUs
**Lessons:** The LSTM cell gates (forget, input, output), the cell state as long-term memory, GRU as a simplified LSTM
**Exercises:** Replace your RNN with an LSTM. Compare loss curves and generated text quality. Implement the LSTM cell equations from scratch.
**Resources:** Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*; Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder

---

### DAY 22 — Seq2Seq and the Encoder-Decoder Architecture
**Lessons:** Translation as a sequence-to-sequence problem, the encoder-decoder framework, information bottleneck
**Exercises:** Build a toy seq2seq model for reversing sentences. Why does long-sequence reversal degrade? This is your intuition for why attention was invented.

---

### DAY 23 — Attention: The Breakthrough
**Lessons:** Bahdanau attention, the alignment matrix, soft vs. hard attention, dot-product attention
**Exercises:** Add attention to your seq2seq model. Visualize the attention weights as a heatmap. What does the model "look at" when generating each output word?
**Resources:** Bahdanau, D., Cho, K. & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate; Luong et al. (2015). Effective Approaches to Attention-based NMT

---

### DAY 24 — Module 04 Review & Mini-Project
**Project (Mini-Project 4):** Build a character-level LSTM with attention trained on the Gospel of Mark. Generate 200-word passages that continue a given first verse. Compare quality with your Day 18 MLP model. This is your baseline before Transformers.

---

## MODULE 05 — NLP Foundations: Text as Data
### Days 25–30 | Intermediate

---

### DAY 25 — Tokenization in Depth
**Lessons:** Word tokenization, subword tokenization (BPE, WordPiece, SentencePiece), the vocabulary size tradeoff, special tokens ([CLS], [SEP], [PAD], [UNK])
**Exercises:** Implement Byte Pair Encoding from scratch. Train it on the KJV Bible. What is the most common subword unit? Train it separately on Banso text — how does the vocabulary differ?
**Resources:** Sennrich, R. et al. (2016). Neural Machine Translation of Rare Words with Subword Units; Kudo, T. (2018). SentencePiece: A simple and language independent subword tokenizer

---

### DAY 26 — Word Embeddings
**Lessons:** One-hot encoding limitations, distributed representations, Word2Vec (CBOW and Skip-gram), GloVe, fastText
**Exercises:** Train Word2Vec on KJV Bible. Find the nearest neighbors of "grace", "covenant", "shepherd". Now try "nfor" (Banso for God). Visualize with t-SNE.
**Resources:** Mikolov, T. et al. (2013). Efficient Estimation of Word Representations in Vector Space; Pennington, J. et al. (2014). GloVe: Global Vectors for Word Representation

---

### DAY 27 — Text Classification & Sentiment
**Lessons:** Feature engineering for text, TF-IDF, classification with logistic regression and MLP, evaluation metrics (precision, recall, F1)
**Exercises:** Build a classifier to distinguish Psalms of praise vs. Psalms of lament. Can you get above 80% accuracy? What features matter most?

---

### DAY 28 — Named Entity Recognition & Sequence Labeling
**Lessons:** IOB tagging, CRF models, evaluation with span-level metrics
**Exercises:** Annotate 50 verses from Genesis with entity tags (PERSON, PLACE, DEITY, TRIBE). Train a simple NER model. What entities does the model learn most easily?

---

### DAY 29 — Pre-Transformer Language Models: ELMo and GPT-1
**Lessons:** Contextual embeddings vs. static embeddings, ELMo's bidirectional LM, OpenAI GPT-1 and the "pretraining" paradigm
**Exercises:** Load ELMo embeddings. Compare the representation of "right" in "the right hand of God" vs. "that is right and just" — show how context changes the embedding.
**Resources:** Peters, M. et al. (2018). Deep contextualized word representations (ELMo); Radford, A. et al. (2018). Improving Language Understanding by Generative Pre-Training (GPT-1)

---

### DAY 30 — Module 05 Review & Mini-Project
**Project (Mini-Project 5):** A complete NLP pipeline: tokenize the KJV Bible, train BPE, generate embeddings, build a simple semantic search engine that finds verses similar to a query. Try queries like "God speaks in the storm" and "the bread of life" and evaluate result quality.

---

## MODULE 06 — The Transformer Architecture
### Days 31–36 | Advanced

---

### DAY 31 — "Attention Is All You Need"
**Lessons:** The 2017 breakthrough paper, limitations of RNNs that Transformers solve, the full Transformer architecture overview
**Exercises:** Read the original Vaswani et al. (2017) paper carefully. Annotate every equation in your own words. Draw the full architecture on paper.
**Resources:** Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017* — THIS IS THE MOST IMPORTANT PAPER IN THIS CURRICULUM

---

### DAY 32 — Scaled Dot-Product Attention
**Lessons:** Queries, keys, and values — the information retrieval analogy, the scaling factor √dₖ and why it matters, the attention weight matrix
**Exercises:** Implement scaled dot-product attention from scratch in PyTorch (no nn.MultiheadAttention). Verify your output matches PyTorch's implementation on identical inputs.

---

### DAY 33 — Multi-Head Attention & Positional Encoding
**Lessons:** Why multiple attention heads? What does each head specialize in? Sinusoidal positional encoding, learned positional embeddings, RoPE
**Exercises:** Visualize the attention heads of a pre-trained model (use BertViz). What do different heads attend to? Implement sinusoidal positional encoding and verify the geometric properties.
**Resources:** Voita et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting; Su et al. (2021). RoFormer: Rotary Position Embedding

---

### DAY 34 — The Transformer Block
**Lessons:** Layer normalization (pre-norm vs. post-norm), feed-forward sublayers, residual connections, the role of each component
**Exercises:** Implement a complete Transformer block from scratch. Verify forward pass dimensions. Count the parameters. Where do most parameters live?

---

### DAY 35 — The Decoder-Only GPT Architecture
**Lessons:** Encoder-only (BERT) vs. encoder-decoder (T5) vs. decoder-only (GPT), causal masking, the autoregressive generation process
**Exercises:** Implement the causal mask. Verify that position i cannot attend to position j > i. Implement a complete GPT-style model with configurable depth and width.
**Resources:** Radford et al. (2018) GPT-1; Brown et al. (2020) GPT-3 paper; Karpathy, A. — *nanoGPT* GitHub (study the code carefully)

---

### DAY 36 — Module 06 Review & Mini-Project
**Project (Mini-Project 6):** Implement `MarkGPT-Nano` — a minimal GPT-style Transformer (~2M parameters) trained on a single book of the Bible (start with John). It should generate coherent text completions. This is the architectural seed of your final MarkGPT.

---

## MODULE 07 — Training Large Language Models
### Days 37–42 | Advanced

---

### DAY 37 — The Pre-Training Objective
**Lessons:** Next-token prediction as self-supervised learning, the dataset scale required, what models learn during pretraining
**Exercises:** Train MarkGPT-Nano on 3 different books (Genesis, Psalms, John). Compare loss curves. What affects convergence speed most?

---

### DAY 38 — Data Pipelines at Scale
**Lessons:** DataLoader design, tokenization pipelines, data shuffling and streaming, the importance of data quality vs. quantity
**Exercises:** Build an efficient streaming DataLoader for the full KJV Bible. Profile your training speed. Find the bottleneck.

---

### DAY 39 — Learning Rate Schedules & Training Stability
**Lessons:** Warmup, cosine annealing, linear decay, gradient clipping, mixed-precision training (fp16/bf16)
**Exercises:** Ablation study: train MarkGPT-Nano with 5 different LR schedules. Plot all loss curves on one graph. Write a 1-page analysis of what you observe.
**Resources:** Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts

---

### DAY 40 — Evaluation: Beyond Loss
**Lessons:** Perplexity revisited, BLEU/ROUGE for generation, BERTScore, human evaluation, calibration
**Exercises:** Evaluate your MarkGPT-Nano on held-out Bible chapters. Compute perplexity. Sample 20 generated passages and rate them qualitatively on fluency, coherence, and "Biblical register."

---

### DAY 41 — Scaling Laws
**Lessons:** The Chinchilla scaling laws, compute-optimal training, the relationship between model size, data, and compute
**Exercises:** Given a fixed compute budget of 10¹⁸ FLOPs, what is the optimal model size and dataset size according to Chinchilla? Apply this to MarkGPT.
**Resources:** Kaplan, J. et al. (2020). Scaling Laws for Neural Language Models; Hoffmann, J. et al. (2022). Training Compute-Optimal Large Language Models (Chinchilla)

---

### DAY 42 — Module 07 Review & Mini-Project
**Project (Mini-Project 7):** Train MarkGPT-Small (10M parameters) on the full KJV Bible. Document your complete training run: configuration, loss curve, sample outputs at every 1000 steps, final evaluation. This is the model you will fine-tune in Module 08.

---

## MODULE 08 — Fine-Tuning & Transfer Learning
### Days 43–48 | Advanced

---

### DAY 43 — The Transfer Learning Paradigm
**Lessons:** What is transfer learning? Feature extraction vs. fine-tuning, why pretrained weights are valuable, catastrophic forgetting
**Exercises:** Fine-tune MarkGPT-Small on the Gospel of Luke only (after pretraining on full Bible). Does Lukan style emerge? How quickly?

---

### DAY 44 — Instruction Fine-Tuning
**Lessons:** RLHF overview, SFT (supervised fine-tuning), the instruction-following dataset format, prompt templates
**Exercises:** Create 50 instruction-response pairs from Biblical text (e.g., "Summarize Genesis 1 in the Banso style" → [example response]). Fine-tune MarkGPT-Small on these.
**Resources:** Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback (InstructGPT); Wei et al. (2022). Finetuned Language Models Are Zero-Shot Learners

---

### DAY 45 — Parameter-Efficient Fine-Tuning (PEFT)
**Lessons:** LoRA (Low-Rank Adaptation), adapters, prefix tuning, prompt tuning — how to fine-tune large models with minimal compute
**Exercises:** Implement LoRA from scratch. Apply it to fine-tune MarkGPT on Banso-style text. Compare parameter count: full fine-tune vs. LoRA.
**Resources:** Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models

---

### DAY 46 — Tokenizer Adaptation for New Languages
**Lessons:** Why base tokenizers fail on low-resource languages, vocabulary extension, embedding initialization for new tokens
**Exercises:** Train a BPE tokenizer on your Banso dataset. Measure fertility (tokens per word) compared to GPT-2's tokenizer on the same text. Extend MarkGPT's vocabulary with Banso-specific tokens.

---

### DAY 47 — Evaluation for Culturally-Specific Language
**Lessons:** Cultural appropriateness in generation, fluency vs. faithfulness, human evaluation protocols for minority languages, community-centered evaluation
**Exercises:** Design a 10-item human evaluation rubric for MarkGPT's Banso-style output. If possible, share 10 generated passages with a native Banso speaker and collect ratings.

---

### DAY 48 — Module 08 Review & Mini-Project
**Project (Mini-Project 8):** Fine-tune MarkGPT-Small on Banso-English parallel text using LoRA. The model should now generate completions that blend Biblical cadence with Banso vocabulary and idiom.

---

## MODULE 09 — Building the Banso Dataset
### Days 49–54 | Advanced + Cultural

---

### DAY 49 — The Banso Language and People
**Lessons:** The Banso people of the Nso Kingdom, Northwest Cameroon, the linguistic features of Banso (also called Lamnso'), tonal features, vocabulary roots, writing conventions
**Exercises:** Compile a glossary of 50 Banso words with English translations. Identify 10 words with no direct English equivalent. What do these tell us about Banso culture?

---

### DAY 50 — Sourcing Biblical Text in Banso/Related Languages
**Lessons:** Available Bible translations in Cameroonian languages, the SIL/Wycliffe archives, using OpenBible and parallel corpora
**Exercises:** Download available Bible text in Lamnso' (or closest available dialect). Align it with KJV at the verse level. Build a parallel corpus of at least 500 verse pairs.

---

### DAY 51 — Data Cleaning and Augmentation
**Lessons:** Removing artifacts, normalizing orthography, data augmentation strategies for low-resource languages (back-translation, paraphrasing)
**Exercises:** Clean and normalize your parallel corpus. Implement a simple back-translation augmenter. Document every cleaning decision in `data/banso-vernacular/CLEANING_LOG.md`

---

### DAY 52 — Dataset Versioning and Documentation
**Lessons:** Datasheets for datasets, data cards, ethical considerations in minority language data collection, consent and community involvement
**Exercises:** Write a full datasheet for your Banso Bible corpus following the Gebru et al. (2021) format. This document is as important as the data itself.
**Resources:** Gebru, T. et al. (2021). Datasheets for Datasets. *Communications of the ACM*

---

### DAY 53 — Tokenizer Training for Banso
**Lessons:** SentencePiece for tonal languages, vocabulary size selection, the fertility metric, handling diacritics and tonal markers
**Exercises:** Train three Banso tokenizers with vocabularies of 1000, 4000, and 8000 tokens. Compare fertility scores and perplexity on held-out text. Choose the optimal vocabulary size.

---

### DAY 54 — Module 09 Review & Dataset Freeze
**Deliverable:** A complete, documented, versioned dataset package:
- `data/banso-vernacular/banso_bible_v1.0.txt` — cleaned text corpus
- `data/banso-vernacular/parallel_corpus.csv` — aligned KJV ↔ Banso verse pairs  
- `data/banso-vernacular/datasheet.md` — full dataset documentation
- `data/banso-vernacular/tokenizer/` — trained BPE tokenizer
- `data/banso-vernacular/vocab_stats.json` — vocabulary analysis

---

## MODULE 10 — Training & Deploying MarkGPT
### Days 55–60 | Capstone

---

### DAY 55 — Final Architecture Design
**Lessons:** Choosing MarkGPT's final configuration based on your hardware, the tradeoffs at each scale, configuration management with YAML
**Exercises:** Complete `configs/markgpt_final.yaml`. Run a 100-step smoke test. Verify the model can fit in memory and reaches expected MFU (model FLOP utilization).

---

### DAY 56 — Pretraining MarkGPT on Biblical Text
**Exercises:** Launch the full pretraining run on the KJV Bible. Log with WandB or TensorBoard. Save checkpoints every 1000 steps. Monitor: loss, gradient norm, learning rate, perplexity.
**Milestone:** MarkGPT should generate recognizable Biblical prose by the end of this day.

---

### DAY 57 — Fine-Tuning MarkGPT on Banso Data
**Exercises:** Fine-tune using LoRA on the Banso corpus. Mix ratio: 70% Banso, 30% KJV. Evaluate after every 200 steps. Sample generations and check for Banso vocabulary emergence.

---

### DAY 58 — Evaluation & Analysis
**Exercises:** Run the full evaluation suite: perplexity, human evaluation rubric, linguistic feature analysis (does the model use Banso tonal patterns?). Write a 2-page evaluation report.

---

### DAY 59 — Inference API & Demo
**Lessons:** Text generation strategies: greedy, top-k, top-p (nucleus sampling), temperature scaling
**Exercises:** Build a simple command-line interface for MarkGPT. Then build a minimal Gradio web demo. Deploy to Hugging Face Spaces.

---

### DAY 60 — Graduation: Reflection, Documentation & Sharing
**Final Deliverables:**
1. `capstone/markgpt/model_card.md` — A complete model card
2. `capstone/markgpt/training_report.md` — Full training narrative
3. `capstone/markgpt/sample_outputs.md` — 50 curated outputs showing MarkGPT's personality
4. A public GitHub release of your MarkGPT weights and code
5. Personal reflection: "What I learned. What surprised me. What I would do differently."

**Celebration:** You have built an LLM. You have preserved a piece of Banso linguistic heritage in silicon. That matters.

---

## 📊 Assessment Overview

| Assessment | Weight | When |
|---|---|---|
| Daily exercises (attendance & completion) | 20% | Daily |
| Mini-Projects 1–8 | 40% | End of each module |
| MarkGPT Capstone | 30% | Days 55–60 |
| Final Reflection Essay | 10% | Day 60 |

---

## 🔖 Recommended Reading Order (Core Texts)

1. Jurafsky & Martin — *Speech and Language Processing* (free, read alongside modules 1–5)
2. Goodfellow, Bengio & Courville — *Deep Learning* (free, read alongside modules 3–4)
3. Vaswani et al. (2017) — *Attention Is All You Need* (read on Day 31, re-read on Day 60)
4. Karpathy, A. — *nanoGPT* codebase (read alongside module 6)
5. Brown et al. (2020) — *Language Models are Few-Shot Learners* (GPT-3 paper)

---

*Last updated: 2024 | This syllabus is a living document — check `CHANGELOG.md` for updates*
