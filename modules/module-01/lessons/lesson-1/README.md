# Lesson 1: The History of AI: From Turing to Transformers
## A Complete Narrative for the Curious Beginner

## Table of Contents
- Before We Begin: A Thought Experiment
- 1. The Beginning: Alan Turing and the Imitation Game (1950)
- 2. The First Wave: Symbolic AI (1950s–1980s)
- 3. The Second Wave: Connectionism and Neural Networks (1980s–2000s)
- 4. The Deep Learning Revolution (2006–2015)
- 5. Language's Turn: From Word2Vec to BERT to GPT (2013–2019)
- 6. The Transformer Epoch (2017–Present)
- 7. Where MarkGPT Fits In
- Key Concepts from This Lesson
- Exercises for Day 1

---

## Before We Begin: A Thought Experiment

Imagine you are six years old and you ask your grandmother a question: *"What does 'covenant' mean?"*

She doesn't open a dictionary. She doesn't recite a definition. She tells you a story — maybe about a promise your grandfather made, maybe about a handshake that meant everything in a village. Through that story, through context and warmth and memory, the word *covenant* takes on meaning for you. You now understand it not as a definition but as a living thing.

Now: can a machine do that?

This question — whether a machine can understand, generate, or engage meaningfully with language — is the question that has driven artificial intelligence research for more than seventy years. And the answer, as of the 2020s, is: *sort of, and in ways that continue to astonish even the researchers who build these systems.*

This lesson tells the story of how we got here.

---

## 1. The Beginning: Alan Turing and the Imitation Game (1950)

The story of machine intelligence begins seriously in 1950 with a British mathematician named Alan Turing, who published a paper titled *"Computing Machinery and Intelligence."* It opens with a question that still echoes today: *"Can machines think?"*

Turing was too careful a thinker to answer that question directly. Instead, he proposed a test — now called the *Turing Test* or the *Imitation Game* — in which a human judge conducts text-based conversations with both a human and a machine. If the judge cannot reliably distinguish the machine from the human, then for all practical purposes, the machine is demonstrating something that looks like intelligence.

The Turing Test was less a solution than a provocation. It shifted the question from the philosophical ("Can machines think?") to the operational ("Can a machine fool a person?"). This shift — from asking what intelligence *is* to asking what it *does* — would shape AI research for generations.

**Takeaway for your journey:** When you eventually test MarkGPT and read its outputs, you will have a direct, personal sense of what this question means. Does it sound like something that understands? The answer is more complicated than a simple yes or no.

---

## 2. The First Wave: Symbolic AI (1950s–1980s)

Early AI researchers were wildly optimistic. In 1956, a group of mathematicians and scientists met at Dartmouth College for a summer workshop that is considered the founding moment of AI as a field. They believed that within a generation, machines would be able to do everything a human mind could do.

Their approach was called *symbolic AI* or *Good Old-Fashioned AI (GOFAI)*. The idea was elegant and logical: intelligence is fundamentally about manipulating symbols according to rules. To give a machine intelligence, you write down the rules explicitly. For language, this meant writing grammars — formal descriptions of which sentences are valid and what they mean.

This produced genuinely impressive systems. Programs like ELIZA (1966) could carry on simple conversations. Expert systems in the 1970s and 80s could diagnose diseases with accuracy comparable to specialists, by encoding the rules doctors use into formal logic.

But symbolic AI had a deep problem: **the world is too complicated for complete rule specification.** Language especially. Every rule has exceptions. Every grammar leaves out idioms, slang, poetry, sarcasm, and the endless creativity of human expression. The more rules researchers wrote, the more exceptions they discovered. The dream of capturing language in a rulebook slowly collapsed.

---

## 3. The Second Wave: Connectionism and Neural Networks (1980s–2000s)

If the first wave tried to program intelligence from the top down (rules first, then behavior), the second wave tried to grow it from the bottom up (experience first, then behavior). This approach was inspired by the brain, not by logic.

The key insight: rather than telling a machine what the rules are, show it thousands of examples and let it discover the patterns itself. This is called *learning*, and the structures that do it are called *artificial neural networks* — loosely inspired by biological neurons.

The building blocks had been around since the 1950s, but two moments catalyzed the modern era. In 1986, David Rumelhart, Geoffrey Hinton, and Ronald Williams published the backpropagation algorithm — a method for efficiently training multi-layer neural networks. This was the key that had been missing.

But even backpropagation wasn't enough at first. Networks were shallow and slow, data was limited, and computers weren't powerful enough. The 1990s were a "second winter" for neural networks — funding dried up, interest faded, and statistical methods like Support Vector Machines dominated.

---

## 4. The Deep Learning Revolution (2006–2015)

The revival came from a small team led by Geoffrey Hinton at the University of Toronto, who in 2006 showed that deep networks (networks with many layers) could be pre-trained in a clever way before fine-tuning on specific tasks. Suddenly, deep was not just harder — it was *better*.

The decisive moment was 2012. A neural network called AlexNet, trained on ImageNet (a massive dataset of labeled photographs), cut the image classification error rate almost in half compared to any previous method. The AI community took notice.

Within a few years, deep neural networks were breaking records in image recognition, speech recognition, and eventually language. The ingredients were in place: better algorithms, massive datasets from the internet, and the GPU — a chip designed for rendering video games that turned out to be extraordinarily good at the matrix multiplications that neural networks require.

---

## 5. Language's Turn: From Word2Vec to BERT to GPT (2013–2019)

For language specifically, the first major deep learning breakthrough came in 2013 with Word2Vec — a neural method for learning word representations (called *embeddings*) from text. The surprising discovery: words with similar meanings ended up close together in the embedding space. Mathematics could be done on meanings. *king − man + woman ≈ queen*.

But word embeddings were static — the word "bank" had the same representation whether you were talking about a river bank or a financial institution. The next step was *contextual* embeddings: representations that change depending on surrounding words.

In 2018, two landmark models appeared. First, **ELMo** (from AllenNLP) showed that deep bidirectional language models produce powerful contextual representations. Second, and more transformatively, Google released **BERT** — a model built on a new architecture called the *Transformer* — that achieved state-of-the-art performance on nearly every language benchmark in a single paper.

At the same time, OpenAI released the first **GPT** (Generative Pre-trained Transformer). Where BERT was built for understanding, GPT was built for generation: given some text, predict what comes next. This would become the paradigm for MarkGPT.

---

## 6. The Transformer Epoch (2017–Present)

The Transformer architecture — introduced by Vaswani et al. in the 2017 paper "Attention Is All You Need" — is the most important architectural innovation in AI in decades. We will study it in extraordinary detail in Module 06, but here is the core idea:

Instead of processing text word by word in sequence (as RNNs do), Transformers process all words simultaneously and let every word directly "attend to" every other word. This parallel processing is why Transformers can be scaled to enormous size and why they can be trained efficiently on modern hardware.

GPT-2 (2019, 1.5 billion parameters), GPT-3 (2020, 175 billion parameters), and then ChatGPT (2022), Claude, Gemini, and Llama demonstrated that scaling up Transformer-based language models produces something that feels qualitatively different from anything before — systems that can write, reason, code, translate, summarize, and converse in ways that regularly surprise their creators.

---

## 7. Where MarkGPT Fits In

MarkGPT is not GPT-4. It is not even GPT-2. It will have somewhere between 2 million and 85 million parameters, depending on your hardware — orders of magnitude smaller than production LLMs.

But here is what matters: **MarkGPT uses exactly the same architecture, the same training algorithm, and the same fundamental ideas as the largest models in the world.** The difference is scale, not structure. By building MarkGPT, you will understand — from the inside — how every modern LLM works.

And MarkGPT has something the big models don't: it will know Banso. It will have absorbed the cadences of Lamnso' proverbs and the rhythms of Biblical language in a dialect that has almost no representation in any commercial AI system. That is something genuinely new.