# 📚 Complete Resources & References Guide
## MarkGPT 60-Day LLM Curriculum

> "Stand on the shoulders of giants — but know whose shoulders you're standing on."

This document is your master reference library. Every resource listed here has been chosen because it deepens understanding, not just skill. For each resource, a brief annotation explains why it matters and when to read it.

---

## PART I: FOUNDATIONAL TEXTBOOKS

These are the texts a graduate student in NLP/ML would be expected to know. They are listed in the order you should encounter them in this curriculum.

---

### Speech and Language Processing
**Authors:** Daniel Jurafsky & James H. Martin  
**Access:** Free PDF at https://web.stanford.edu/~jurafsky/slp3/  
**Read during:** Modules 01–05  

This is the bible (pun intended) of NLP education. Jurafsky and Martin cover everything from n-gram models and part-of-speech tagging through to neural networks and transformers, always explaining *why* before *how*. If you read no other textbook in this curriculum, read this one. It is honest about mathematical prerequisites, patient with notation, and generous with examples. The chapter on language models (Chapter 3) should be read on Days 4–5; the chapters on neural networks (Chapters 7–9) alongside Module 03.

---

### Deep Learning
**Authors:** Ian Goodfellow, Yoshua Bengio, Aaron Courville  
**Access:** Free online at https://www.deeplearningbook.org/  
**Read during:** Modules 03–04  

The graduate textbook for deep learning. It is mathematically rigorous — more so than most learners need — but Parts I and II (Applied Math & Machine Learning Basics, and Modern Practical Deep Networks) are invaluable. Pay special attention to Chapter 6 (Deep Feedforward Networks) and Chapter 8 (Optimization for Training Deep Models). The backpropagation chapter (6.5) is one of the clearest derivations available.

---

### Mathematics for Machine Learning
**Authors:** Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong  
**Access:** Free PDF at https://mml-book.github.io/  
**Read during:** Module 02  

The linear algebra and calculus you need, explained specifically for machine learning. Chapter 2 (Linear Algebra) and Chapter 5 (Vector Calculus) are the most important for this curriculum. Unlike a pure mathematics textbook, every concept is immediately motivated by its ML application. Read this instead of going back to a pure linear algebra course.

---

### Neural Networks and Deep Learning (Online Book)
**Author:** Michael Nielsen  
**Access:** Free at http://neuralnetworksanddeeplearning.com/  
**Read during:** Module 03  

Nielsen's online book is shorter and more intuitive than Goodfellow et al. Its strength is its visual approach to backpropagation and its emphasis on developing intuition alongside calculation. The four chapters can be read in a weekend. If the Goodfellow textbook feels overwhelming, start here.

---

## PART II: SEMINAL RESEARCH PAPERS

These papers mark major turning points in the history of language models. Reading them in order is like watching the field evolve in real time.

---

### The Historical Foundation

**Turing, A.M. (1950). "Computing Machinery and Intelligence." *Mind*, 59(236), 433–460.**  
Read on Day 1. The question this paper asks — "Can machines think?" — motivates everything in this curriculum. It is written in elegant English requiring no mathematical background.

**McCulloch, W.S. & Pitts, W. (1943). "A Logical Calculus of Ideas Immanent in Nervous Activity." *Bulletin of Mathematical Biophysics*, 5(4), 115–133.**  
The original mathematical neuron paper. Read on Day 13, just before implementing your own.

**Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain." *Psychological Review*, 65(6), 386–408.**  
The perceptron — the ancestor of every neural network you will build.

**Rumelhart, D.E., Hinton, G.E. & Williams, R.J. (1986). "Learning Representations by Back-propagating Errors." *Nature*, 323, 533–536.**  
Four pages that changed everything. Read before Day 15 (backpropagation from scratch).

---

### The Modern NLP Foundation

**Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735–1780.**  
The LSTM paper. Read before Day 21. It introduces gating mechanisms that solve the vanishing gradient problem for sequences.

**Bengio, Y., Ducharme, R., Vincent, P. & Jauvin, C. (2003). "A Neural Probabilistic Language Model." *Journal of Machine Learning Research*, 3, 1137–1155.**  
The first neural language model paper. It introduces learned word embeddings — the foundation of all modern NLP.

**Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. (2013). "Distributed Representations of Words and Phrases and their Compositionality." *NeurIPS 2013*.**  
The Word2Vec paper. Read before Day 26. After this paper, "words are vectors" became standard practice.

**Bahdanau, D., Cho, K. & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR 2015*.**  
The paper that introduced attention for sequence models. Read before Day 23. Understanding this paper deeply is essential for appreciating what the Transformer's "attention is all you need" claim really means.

---

### The Transformer Era (The Most Important Papers in This Curriculum)

**Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. & Polosukhin, I. (2017). "Attention Is All You Need." *NeurIPS 2017*.**  
Read on Day 31, in full, with a pen. The founding document of the modern AI era. Every equation, figure, and experiment matters.

**Radford, A., Narasimhan, K., Salimans, T. & Sutskever, I. (2018). "Improving Language Understanding by Generative Pre-Training." OpenAI.**  
GPT-1. The paper that established the pretraining → fine-tuning paradigm for language. Read before Module 06.

**Devlin, J., Chang, M.W., Lee, K. & Toutanova, K. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL 2019*.**  
The encoder-only counterpart to GPT. Understanding both BERT and GPT shows you the full design space for Transformer language models.

**Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. & Sutskever, I. (2019). "Language Models are Unsupervised Multitask Learners." OpenAI.**  
GPT-2. The paper that showed language models are implicit multi-task learners — generating summaries, answering questions, translating text without explicit training on those tasks.

**Brown, T., Mann, B., Ryder, N., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS 2020*.**  
GPT-3 (175B parameters). The paper that demonstrated emergence: capabilities that appear at scale but not at smaller sizes. Read before Module 07 (scaling laws).

---

### Training & Optimization

**Kingma, D.P. & Ba, J. (2014). "Adam: A Method for Stochastic Optimization." *ICLR 2015*.**  
Read before Day 16. You will implement Adam from scratch on Day 16, and understanding the paper makes the code intuitive.

**Loshchilov, I. & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR 2019*.**  
The AdamW paper. MarkGPT uses AdamW. Read before Day 37.

**Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). "Scaling Laws for Neural Language Models." OpenAI.**  
Read before Day 41. The paper that established quantitative relationships between model size, dataset size, compute, and loss. Critical for understanding why MarkGPT's size is chosen.

**Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). "Training Compute-Optimal Large Language Models." *NeurIPS 2022*.**  
The Chinchilla paper. Showed that GPT-3 was significantly under-trained on data relative to its parameter count. The compute-optimal scaling laws you use on Day 41.

---

### Fine-Tuning and Efficiency

**Hu, E.J., Shen, Y., Wallis, P., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.**  
Read before Day 45. LoRA is the method you'll use to fine-tune MarkGPT on Banso text efficiently.

**Ouyang, L., Wu, J., Jiang, X., et al. (2022). "Training Language Models to Follow Instructions with Human Feedback." *NeurIPS 2022*.**  
The InstructGPT paper. Read before Day 44 (instruction fine-tuning). Explains RLHF — the method behind ChatGPT's alignment.

---

## PART III: LANGUAGE & CULTURAL RESOURCES

These resources are as important as the technical papers for building MarkGPT responsibly and authentically.

---

### Low-Resource NLP and African Languages

**Joshi, P., Santy, S., Budhiraja, A., Bali, K. & Choudhury, M. (2020). "The State and Fate of Linguistic Diversity and Inclusion in the NLP World." *ACL 2020*.**  
An essential read before beginning Module 09. This paper maps which languages are represented in NLP research and which are ignored, and why this matters.

**Adelani, D.I., et al. (2021). "MasakhaNER: Named Entity Recognition for African Languages." *EMNLP 2021*.**  
A landmark paper on African language NLP. Demonstrates how community-driven data collection can produce high-quality datasets for low-resource languages.

**Bender, E.M., Gebru, T., McMillan-Major, A. & Shmitchell, S. (2021). "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?" *FAccT 2021*.**  
The "stochastic parrots" paper. Essential reading before Module 09 for understanding the ethical dimensions of large language model training, especially regarding underrepresented languages.

---

### Dataset Ethics

**Gebru, T., Morgenstern, J., Vecchione, B., et al. (2021). "Datasheets for Datasets." *Communications of the ACM*.**  
Read before Day 52. The standard format for documenting datasets. You will fill in a datasheet for your Banso corpus following this format.

**Mitchell, M., Wu, S., Zaldivar, A., et al. (2019). "Model Cards for Model Reporting." *FAccT 2019*.**  
The standard format for model documentation. The MarkGPT model card in `capstone/markgpt/model_card.md` follows this format.

---

### Banso / Lamnso' Language Resources

**SIL International — Lamnso' Language Profile** (https://www.sil.org)  
SIL has worked with the Nso' community on Bible translation and literacy programs. Their archived materials are the primary source for Lamnso' written text.

**Ethnologue — Lamnso' language entry** (https://www.ethnologue.com/language/lns/)  
Basic demographic and typological data on Lamnso'.

**Yuka, Leonard (2011). "Aspects of Nso' Grammar."**  
The most accessible grammatical description of Lamnso' in English.

**Chumbow, Beban Sammy (1982). "Language and Language Policy in Cameroon."**  
Essential context for understanding Cameroon's multilingual environment and the position of Lamnso' within it.

---

## PART IV: FREE ONLINE COURSES AND TUTORIALS

For learners who prefer video and interactive content alongside the textbooks:

**fast.ai — Practical Deep Learning for Coders** (https://fast.ai)  
Jeremy Howard's bottom-up approach — code first, theory second. Use this as a supplement to Modules 03–04. Lesson 1 provides a fast practical overview before you build from scratch.

**3Blue1Brown — Neural Networks (YouTube series)** (https://www.3blue1brown.com/topics/neural-networks)  
The best visual introduction to neural networks, backpropagation, and gradient descent. Watch before Day 13 and before Day 15. Grant Sanderson's visual intuitions are unmatched.

**Andrej Karpathy — "Zero to Hero" (YouTube series)** (https://karpathy.ai/zero-to-hero.html)  
The companion course to nanoGPT. Karpathy builds language models from scratch in a series of videos that are rigorous, entertaining, and deeply educational. Watch the makemore series alongside Modules 03–05, and the nanoGPT lecture alongside Module 06.

**Hugging Face NLP Course** (https://huggingface.co/learn/nlp-course)  
The practical guide to using Hugging Face tools: tokenizers, datasets, Transformers, and deployment. Useful from Module 08 onward.

**Stanford CS224N — Natural Language Processing with Deep Learning** (https://web.stanford.edu/class/cs224n/)  
Full lecture videos and assignments freely available. The assignments are excellent for learners who want additional structured exercises beyond this curriculum's modules.

**MIT 6.S191 — Introduction to Deep Learning** (http://introtodeeplearning.com/)  
MIT's introductory deep learning course, fully available online. Good parallel resource for Modules 03–04.

---

## PART V: CODE REPOSITORIES TO STUDY

**nanoGPT by Andrej Karpathy** (https://github.com/karpathy/nanoGPT)  
The reference implementation that MarkGPT is inspired by. Study this code carefully alongside Module 06. Every design choice is documented.

**minGPT by Andrej Karpathy** (https://github.com/karpathy/minGPT)  
An earlier, more pedagogically oriented version of nanoGPT. Slightly easier to follow as a first read.

**micrograd by Andrej Karpathy** (https://github.com/karpathy/micrograd)  
A tiny autograd engine in ~150 lines of Python. Study this to understand what PyTorch's autograd is doing under the hood. Read before Day 15.

**Transformers by Hugging Face** (https://github.com/huggingface/transformers)  
The production implementation of hundreds of transformer models. After you have built MarkGPT from scratch, study how the same ideas are implemented at scale. In particular, the GPT-2 implementation is a useful comparison.

**llm.c by Andrej Karpathy** (https://github.com/karpathy/llm.c)  
GPT-2 training in pure C. Not required, but fascinating for understanding what is happening at the hardware level below PyTorch.

---

## PART VI: COMMUNITIES AND STAYING CURRENT

**Papers with Code** (https://paperswithcode.com/)  
Tracks the state of the art in machine learning with links to code implementations. Essential for knowing which papers are worth reading as the field evolves.

**Hugging Face Hub** (https://huggingface.co/)  
The central repository for pretrained models, datasets, and papers in NLP.

**Arxiv cs.CL (Computation and Language)** (https://arxiv.org/list/cs.CL/recent)  
Where new NLP research papers appear daily. After completing this curriculum, building a habit of scanning arXiv will keep you current.

**ML Subreddit** (https://www.reddit.com/r/MachineLearning/)  
Community discussion of papers, experiments, and news.

**AI Twitter/X community** — Follow researchers like Andrej Karpathy, Yann LeCun, Christopher Manning, and Abeba Birhane for diverse perspectives on the field's direction.

---

*This reference list is a living document. New papers will be added as the curriculum is updated. Suggestions are welcome via pull request.*
