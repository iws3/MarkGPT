# Lesson 2: What Is a Language Model?
## Understanding the Core of Modern AI

## Table of Contents
- Introduction to Language Models
- The Mathematical Foundation
- Types of Language Models
- How Language Models Learn
- Evaluating Language Models
- Language Models in Practice
- Challenges and Limitations
- The Future of Language Modeling

---

## Introduction to Language Models

A language model is a system that learns the probability distribution of sequences of words or tokens in a language. It predicts what word is likely to come next given the previous words. Language models are the foundation of many natural language processing tasks, from machine translation to text generation.

The concept dates back to the 1950s, but modern language models use deep learning to achieve remarkable performance. They are trained on vast amounts of text data and can generate coherent, contextually appropriate text.

In this lesson, we'll explore what makes language models tick, how they work, and why they are so powerful.

---

## The Mathematical Foundation

At its core, a language model computes the probability of a sequence of words: P(w1, w2, ..., wn). Using the chain rule of probability, this can be decomposed into: P(w1) * P(w2|w1) * P(w3|w1,w2) * ... * P(wn|w1,...,w{n-1}).

In practice, we use the Markov assumption to limit the context window, looking only at the previous k words. For bigram models, k=1; for trigram models, k=2.

Modern neural language models use attention mechanisms to consider all previous words, making them much more powerful.

---

## Types of Language Models

Language models can be categorized in several ways:

1. **Statistical vs Neural**: Statistical models use n-grams and smoothing techniques. Neural models use deep learning.

2. **Autoregressive vs Autoencoding**: Autoregressive models (like GPT) predict the next token. Autoencoding models (like BERT) predict masked tokens.

3. **Unidirectional vs Bidirectional**: Unidirectional models only look at previous context. Bidirectional models consider both directions.

4. **Causal vs Masked**: Causal models are for generation, masked for understanding.

Each type has its strengths and use cases.

---

## How Language Models Learn

Language models learn by maximizing the likelihood of the training data. For a sequence, the model predicts each token given the previous ones, and the loss is the negative log probability of the correct token.

Training involves:

1. Tokenizing the text into subwords or words

2. Converting tokens to embeddings

3. Passing through the model (RNN, Transformer, etc.)

4. Computing the loss

5. Backpropagating to update parameters

This process is repeated on millions of examples until the model converges.

---

## Evaluating Language Models

Language models are evaluated using metrics like perplexity, which measures how well the model predicts the test data. Lower perplexity indicates better performance.

Other metrics include BLEU for translation, ROUGE for summarization, and human evaluation for quality.

Cross-entropy loss is also used during training to measure how well the model fits the data.

---

## Language Models in Practice

Language models are used in many applications:

- Text generation and completion

- Machine translation

- Sentiment analysis

- Chatbots and virtual assistants

- Code generation

- Summarization

They power systems like GPT, BERT, and are integrated into search engines, email clients, and more.