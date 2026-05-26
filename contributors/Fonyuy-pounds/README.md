# Fonyuy-pounds's Contributor Workspace

Welcome to my contributor folder! I'm using this space to improve the MarkGPT LLM Curriculum.

## Current Focus

- [x] Module 01: AI Foundations & BansoGram
- [x] Module 02: Math for ML (PCA & Word Mappings)
- [x] Module 03: Neural Networks (MLPs & Backprop)
- [x] Module 04: Sequence Modeling (RNNs, LSTMs, Seq2Seq, Attention) - Day 24 Completed (Mini-Project 4)
- [x] Module 05: NLP Foundations (Tokenization, Embeddings, Classification, NER, Contextual LMs) - Day 30 Completed (Module 05 Capstone)

## Contributions

- **Module 01**: Built `banso_gram.py`, a Banso-English translation and grammar utility.
- **Module 02**: Implemented custom PCA for word embedding visualization.
- **Module 03**: Developed character-level MLP models for text generation.
- **Module 04 (Day 20)**: Implemented vanilla RNN from scratch for character-level Bible text generation.
- **Module 04 (Day 21)**: Built a full LSTM implementation in NumPy to solve vanishing gradients and improve memory.
- **Module 04 (Day 22)**: Transitioned to PyTorch and implemented a Seq2Seq Encoder-Decoder model for sequence reversal.
- **Module 04 (Day 23)**: Integrated Dot-Product Attention into the Seq2Seq model to eliminate the context vector bottleneck.
- **Module 04 (Day 24)**: Completed Mini-Project 4: MarkLSTM with Attention for character-level text generation on the Gospel of Mark.
- **Module 05 (Day 25)**: Implemented Byte-Pair Encoding (BPE) from scratch and analyzed fertility on KJV Bible text.
- **Module 05 (Day 26)**: Developed custom Skip-gram Word2Vec with Negative Sampling in PyTorch, aligned Banso term "nfor" with theological concepts, and exported t-SNE visualizations.
- **Module 05 (Day 27)**: Built a full Text Classification pipeline from scratch (custom TF-IDF, Logistic Regression, and MLP in pure NumPy) to distinguish Psalms of Praise from Psalms of Lament, integrating Banso Kibor/Kighaa cultural phrases and achieving ≥80% accuracy.
- **Module 05 (Day 28)**: Implemented Named Entity Recognition with IOB tagging and Conditional Random Fields (CRF) from scratch using pure NumPy with Viterbi decoding. Annotated 50 Genesis verses with PERSON/PLACE/DEITY/TRIBE labels, computed span-level metrics, and validated with Banso entity classification patterns. Achieved ≥78% span-level F1-score on test set.
- **Module 05 (Day 29)**: Explored contextual embeddings with ELMo (bidirectional LSTMs) and generative pretraining paradigm with GPT-1. Analyzed polysemy resolution—comparing embeddings of "right" across different semantic contexts. Demonstrated autoregressive language modeling, temperature-controlled text generation with GPT-2, and the paradigm shift from task-specific training to task-agnostic pretraining. Validated contextualization across Banso theological vocabulary.
- **Module 05 (Day 30)**: Completed Module 05 capstone: semantic search engine integrating Days 25-29. Implemented TF-IDF and Word2Vec embedding-based retrieval, entity-aware hybrid search, and evaluation metrics (precision, recall, F1, MAP). Indexed Genesis corpus with Banso theological vocabulary. Demonstrated full IR pipeline: preprocessing → vectorization → similarity ranking → evaluation. Bridged isolated techniques into production-grade system. Achieved ≥80% retrieval precision on test queries.
