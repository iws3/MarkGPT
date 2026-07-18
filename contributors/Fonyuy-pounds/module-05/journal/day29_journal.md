# Day 29 Learning Journal
## Module 05: NLP Foundations — Pre-Transformer Language Models (ELMo & GPT-1)
**Contributor:** Fonyuy-pounds  
**Date:** Day 29, MarkGPT 60-Day Curriculum  
**Branch:** `fonyuy-pounds-day29`

---

## 1. Today's Objective

The goal today was to understand the **paradigm shift from static to contextual embeddings** and the **pretraining→finetuning** revolution initiated by ELMo (2018) and GPT-1 (2018). These models represent a critical bridge between classical NLP and modern deep learning.

---

## 2. Static Embeddings: The Foundation (Word2Vec, GloVe)

### What Are Static Embeddings?

A static embedding is a **fixed vector assigned to each word type** in a vocabulary, regardless of context:

$$
\text{embedding}(\text{"bank"}) = \mathbf{v}_{\text{bank}} \in \mathbb{R}^d
$$

Every occurrence of "bank" — whether referring to a river bank or a financial bank — gets the **same vector**.

### How They Work

**Word2Vec (Mikolov et al., 2013):**
- Skip-gram: predict context words from target word
- CBOW: predict target word from context words
- Loss: maximize dot product of similar word pairs

$$
\mathcal{L} = -\log P(\mathbf{w}_{\text{context}} | \mathbf{w}_{\text{target}})
$$

**GloVe (Pennington et al., 2014):**
- Matrix factorization approach
- Combines global statistics (cooccurrence matrix) with local context windows
- Loss combines factorization and context prediction:

$$
\mathcal{L} = \sum_{i,j} f(X_{ij}) \left( \mathbf{w}_i^T \mathbf{w}_j + b_i + b_j - \log X_{ij} \right)^2
$$

Where $f(X_{ij})$ is a weighting function (rare cooccurrences weighted less).

### Limitations of Static Embeddings

1. **Polysemy Problem:** Same word, multiple meanings
   - "bank" (river) ≠ "bank" (financial)
   - "right" (correct) ≠ "right" (direction)
   - Solution: use context, but static embedding ignores it!

2. **Out-of-Vocabulary (OOV) Problem:**
   - Words unseen during training have no embedding
   - Mitigations: subword units (FastText), <UNK> token

3. **Grammatical Ambiguity:**
   - "book" (noun: "I read a book") vs. "book" (verb: "I will book a flight")
   - Single embedding cannot distinguish these roles

---

## 3. Contextual Embeddings: The Paradigm Shift

### The Key Insight

Instead of computing embedding once per word **type**, compute it once per word **token** in context:

$$
\text{embedding}(\text{"bank"} \mid \text{context}) = f_{\theta}(\text{context}, \text{pos})
$$

Where $f_{\theta}$ is a learned function (e.g., bidirectional LSTM) and context is the full or windowed sequence.

### ELMo: Embeddings from Language Models (Peters et al., 2018)

**Architecture:**
- Bidirectional LSTM (2 layers)
- Character CNN for subword representations
- Output: concatenation of character CNN, forward LSTM, backward LSTM

$$
\text{ELMo}(\text{token}_i) = \mathbf{c}_i \oplus \mathbf{h}^{\text{fwd}}_i \oplus \mathbf{h}^{\text{bwd}}_i
$$

Where $\oplus$ denotes concatenation.

**Training Objective:**
Pre-train on a large corpus using the language modeling objective:

$$
\mathcal{L} = -\sum_i \left[ \log P(w_i | w_1, \ldots, w_{i-1}) + \log P(w_i | w_{i+1}, \ldots, w_n) \right]
$$

Bidirectional LM means predicting both forward and backward.

**Key Advantage:**
- Fine-tune for downstream tasks (NER, sentiment, SQuAD)
- Embeddings automatically capture task-relevant context

### Polysemy Example: "right"

Consider these three sentences:

1. "the **right** hand of God is glorious"
2. "that is **right** and just"
3. "on the **right** and left"

**With static embeddings:**
$$
\mathbf{v}_{\text{right}, \text{sent1}} \approx \mathbf{v}_{\text{right}, \text{sent2}} \approx \mathbf{v}_{\text{right}, \text{sent3}}
$$

All three embeddings are nearly identical, losing semantic distinction.

**With contextual embeddings (ELMo):**
$$
\text{ELMo}(\text{right} \mid \text{sent1}) \neq \text{ELMo}(\text{right} \mid \text{sent2}) \neq \text{ELMo}(\text{right} \mid \text{sent3})
$$

The LSTM reads surrounding tokens and adjusts the representation:
- Sent1: context includes "hand" (direction) → emphasis on spatial meaning
- Sent2: context includes "and just" (moral judgment) → emphasis on correctness
- Sent3: context explicitly pairs "right and left" → direction sense activated

---

## 4. GPT-1 & The Pretraining Revolution (Radford et al., 2018)

### Generative Pre-Training

GPT-1 introduced a radically simple yet effective paradigm:

1. **Pretrain** on a huge corpus (BookCorpus, ~1.7GB text)
2. **Fine-tune** on task-specific labeled data (small datasets)

**Key insight:** With enough unlabeled data, a language model learns representations useful for **any** downstream task.

### The GPT-1 Architecture

- Transformer decoder (12 layers)
- Causal masking: token at position $i$ can only attend to positions $< i$
- Trained on next-token prediction

$$
\mathcal{L}_{\text{pretrain}} = -\sum_i \log P(w_i | w_1, \ldots, w_{i-1}; \theta)
$$

### Autoregressive Language Modeling

The **core operation** of GPT-1 (and all large language models):

$$
P(w_{1:n}) = \prod_{i=1}^n P(w_i | w_{1:i-1})
$$

Decompose the joint probability into a product of conditionals.

**Generation Process (Inference):**

```
Given context [w_1, ..., w_k]:
    Repeat until stop:
        Compute P(w_{k+1} | w_1, ..., w_k) via transformer
        Sample w_{k+1} from distribution
        Append w_{k+1} to sequence
```

### Temperature-Scaled Sampling

To control randomness during generation:

$$
P_{\text{temp}}(w_i | \text{context}) = \frac{\exp(\log p_i / T)}{\sum_j \exp(\log p_j / T)}
$$

Where $T$ is temperature:
- $T \to 0$: greedy (always pick argmax)
- $T = 1$: normal distribution
- $T \to \infty$: uniform distribution

**Interpretation:** $1/T$ scales the logits before softmax:
- High $T$ → softer probabilities (more diversity)
- Low $T$ → sharper probabilities (more deterministic)

---

## 5. Pretraining vs. Fine-tuning

### The Paradigm Shift

**Old Approach (2017 and earlier):**
- Task-specific architectures
- Train from random initialization on labeled data
- Limited by dataset size and annotation cost

**New Approach (GPT, BERT era):**
- Single **task-agnostic** pretraining phase
- Transfer to downstream tasks via light fine-tuning
- Scales with unlabeled data (not labeled data)

### Fine-tuning Recipe

1. **Initialize** with pretrained weights
2. **Add task head** (e.g., classification layer)
3. **Fine-tune** on labeled data with learning rate $\alpha_{\text{finetune}}$ (small, e.g., 5e-5)
4. **Evaluate** on test set

Mathematical view:

$$
\theta^* = \arg\min_\theta \mathcal{L}_{\text{task}}(\theta, \mathcal{D}_{\text{labeled}})
$$

Where $\theta$ are the pretrained weights, and $\mathcal{D}_{\text{labeled}}$ is the task-specific labeled set (small).

### Why It Works

1. **Feature transfer:** Lower layers learn general linguistic features (syntax, morphology)
2. **Task transfer:** Higher layers adapt to task-specific patterns
3. **Data efficiency:** Pretrain on billions of tokens; fine-tune on thousands
4. **Regularization:** Staying close to pretrained weights prevents overfitting

---

## 6. ELMo vs. GPT-1: Key Differences

| Aspect | ELMo | GPT-1 |
|--------|------|-------|
| **Direction** | Bidirectional LM (forward + backward LSTM) | Causal/unidirectional (left-to-right) |
| **Architecture** | 2-layer BiLSTM + char CNN | 12-layer Transformer |
| **Scale** | ~94M parameters | ~117M parameters |
| **Output** | Contextual token embeddings | Full model for language modeling |
| **Fine-tuning** | Concatenate to task-specific models | Replace model body + add task head |
| **Downstream** | Best as feature extractor | Best as transfer-learning base |

**Conceptual win for GPT:**
- Simpler fine-tuning: no need to design task-specific architectures
- Unified framework: same model for all tasks
- Causal masking enables autoregressive generation

---

## 7. Banso Theological Vocabulary in Context

In Banso linguistic space, consider sacred terms:

**Nfor (Deity):**
- "Nfor is the supreme creator" → theological sense
- "We honor Nfor's wisdom" → reverence sense
- "Nfor's judgment is just" → justice/authority sense

Contextual embeddings would distinguish these senses, whereas static embeddings would conflate them.

**Kibor (Praise/Celebration):**
- "The kibor ceremony celebrates harvest" → ceremonial sense
- "We express kibor for blessings" → emotional sense
- "Kibor unites the community" → social sense

This is critical for **low-resource language NLP** — Banso, like many African languages, has limited labeled data. Pretraining on available text (Bible, oral traditions) allows transfer to downstream tasks.

---

## 8. Key Takeaways

1. **Static embeddings** (Word2Vec, GloVe) are fast and simple but cannot handle polysemy or grammatical ambiguity.

2. **Contextual embeddings** (ELMo, GPT) compute representations that depend on surrounding context, capturing nuance and word sense.

3. **ELMo** uses bidirectional LSTMs and can be plugged into any downstream model.

4. **GPT-1** introduces the **pretraining→fine-tuning** paradigm that becomes dominant in modern NLP.

5. **Autoregressive language modeling** — predicting the next token given previous tokens — is the pretraining objective that enables generative AI.

6. **Temperature-scaled sampling** controls the diversity of generated text.

7. The paradigm shift from task-specific training to task-agnostic pretraining is foundational to modern large language models (GPT-2, GPT-3, ChatGPT).

8. For low-resource languages like Banso, pretraining on available text enables downstream tasks via transfer learning.

---

## 9. Resources & Further Reading

- Peters, M., Neumann, M., Iyyer, M., et al. (2018). Deep contextualized word representations. *EMNLP*.
- Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving Language Understanding by Generative Pre-Training. *OpenAI Blog*.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *EMNLP*.
- Karpukhin, V., Oguz, B., Min, S., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP*.

---

**End of Day 29 Journal.**
