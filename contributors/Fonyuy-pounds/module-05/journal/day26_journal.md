# Day 26 Journal: Word Embeddings & Representation Learning
## Daily Setup
- **Date**: May 18, 2026
- **Module**: 05 - NLP Foundations
- **Topic**: Distributed Representations & Word2Vec
- **Time Spent**: 6 hours

## Goals for Today
- [x] Analyze the limitations of one-hot word encodings.
- [x] Compare Word2Vec (Skip-gram vs. CBOW), GloVe, and fastText architectures.
- [x] Implement the Skip-gram architecture with Negative Sampling from scratch in PyTorch.
- [x] Integrate the Banso cultural concept of God ("nfor") into our training corpora.
- [x] Train embeddings, find the nearest neighbors for "grace", "covenant", "shepherd", "nfor", and visualize the results using t-SNE.

## Notes and Learnings

### 1. One-Hot Encoding Limitations
One-hot encodings represent words as sparse vectors of size $|V|$, where all entries are zero except the index of the word.
- **Sparse and High-Dimensional**: As $|V|$ grows, the size of each vector explodes, leading to the curse of dimensionality.
- **Orthogonality**: For any two distinct words $w_1$ and $w_2$, their dot product is:
  $$v_{w_1}^\top v_{w_2} = 0$$
  This means that "shepherd" is mathematically as unrelated to "flock" as it is to "microchip". One-hot encodings cannot capture semantic similarity.

### 2. Distributed Representations (Embeddings)
Distributed representations map words to dense vectors in $\mathbb{R}^d$ ($d \ll |V|$). The core idea is **distributional semantics**:
> *"You shall know a word by the company it keeps."* — John Rupert Firth (1957)

- Semantic similarity is translated into geometric proximity (e.g., Cosine Similarity).
- Semantic relationships are captured as vector offsets (e.g., $v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$).

### 3. Word2Vec Architectures
Word2Vec, introduced by Mikolov et al. in 2013, uses a shallow neural network with a single hidden layer to learn word representations.

#### Continuous Bag of Words (CBOW)
- **Objective**: Predicts a target word $w_t$ given its surrounding context words $w_{t-k}, \dots, w_{t+k}$.
- **Formula**:
  $$P(w_t \mid w_{t-k}, \dots, w_{t+k})$$
- **Properties**: Faster to train, works well for frequent words.

#### Skip-gram
- **Objective**: Predicts surrounding context words given a target word $w_t$.
- **Formula**:
  $$P(w_{t+j} \mid w_t) \quad \text{for } -k \le j \le k, j \neq 0$$
- **Properties**: Captures rare words better since each word creates multiple positive training instances.

#### Negative Sampling (SGNS)
Using the full Softmax over a vocabulary $|V|$ is computationally prohibitive:
$$P(c \mid t) = \frac{\exp(v_c^\top v_t)}{\sum_{w \in V} \exp(v_w^\top v_t)}$$

Negative Sampling solves this by converting the multiclass classification problem into a set of binary logistic regressions. For each true $(target, context)$ pair, we sample $K$ negative words $n_1, \dots, n_K$ from a noise distribution $U(w)$ (typically the unigram distribution raised to the $0.75$ power):
$$L_{\text{SGNS}} = - \log \sigma(v_c^\top v_t) - \sum_{i=1}^K \log \sigma(-v_{n_i}^\top v_t)$$

### 4. Alternative Embeddings: GloVe & fastText
- **GloVe (Global Vectors for Word Representation)**: A matrix factorization approach. Instead of a sliding window, it fits embeddings directly to the global word co-occurrence matrix by minimizing a weighted least-squares objective.
- **fastText**: An extension of Word2Vec developed by Facebook. It represents words as bags of character n-grams (e.g., "where" represented by `<wh`, `whe`, `her`, `ere`, `re>`). This allows it to learn embeddings for out-of-vocabulary (OOV) words and handle highly inflected morphologically rich languages beautifully (highly applicable to Bantu languages like Lamnso').

---

## Exercises: Training Word2Vec

I implemented a clean Skip-gram with Negative Sampling model in PyTorch under `day26_exercises.py` and trained it on a combined corpus containing:
1. A 150KB chunk of the KJV Bible text.
2. Banso proverbs from `data/banso-vernacular/proverbs.txt`.
3. Synthetic parallel sentences that align "nfor" (Banso for God) with its Biblical contexts ("lord", "shepherd", "grace", "covenant").

### Cosine Similarity Results
After training, we calculated the nearest neighbors (top-5) using Cosine Similarity. The learned relationships are highly intuitive:

- **GRACE**: Clusters with words like **mercy**, **truth**, **faith**, and **saved**.
- **COVENANT**: Highly associated with **commandment**, **testament**, **promise**, and **lord**.
- **SHEPHERD**: Closest to **flock**, **feed**, **guide**, and **lord**.
- **NFOR** (Banso for God): Incredibly, because of our contextual priming and parallel Banso proverb training, `nfor` successfully clustered in the vector space right alongside **god**, **lord**, **creator**, and **grace**!

### Visualization
The t-SNE 2D projection plots the relative locations of these words, proving that the model mapped the Banso concept of deity (`nfor`) directly into the cluster of core Biblical theology terms!

---

## Reflection
Writing Word2Vec from scratch in PyTorch was a brilliant exercise. Standard libraries like Gensim hide the underlying vector multiplications and backpropagation, but building it manually with `nn.Embedding` and coding the negative sampling loss function explicitly makes the mechanics crystal clear. I now understand exactly how neural weights are updated to reflect semantic company. This is the exact vector representation that will feed into our deep Transformer models in Module 06!
