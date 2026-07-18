# Glossary

## Machine Learning & NLP Terms

**Accuracy**
Percentage of correct predictions out of total predictions. Formula: correct / total.

**Attention**
Mechanism that allows models to focus on relevant parts of input. Multi-head attention uses multiple independent attention mechanisms.

**Autoregressive**
Model that predicts next token based on previous tokens. MarkGPT uses causal autoregressive generation.

**Batch**
Group of training examples processed together. Batch size = number of examples per batch.

**BERT**
Bidirectional Encoder Representations from Transformers. Pre-trained encoder model by Devlin et al. (2019).

**BPE (Byte-Pair Encoding)**
Tokenization algorithm that iteratively merges most frequent token pairs. Used in MarkGPT tokenizer.

**Causal Masking**
Attention mask that prevents attending to future tokens (only sees past). Essential for language modeling.

**Cross-attention**
Attention between two different sequences (encoder-decoder models). MarkGPT uses only self-attention (decoder-only).

**Decoder**
Model component that generates or transforms sequences. MarkGPT is decoder-only (like GPT).

**Embedding**
Learned vector representation of a discrete entity (word, token, position). d_model=512 means 512-dimensional embeddings.

**Encoder**
Model component that encodes input into representations. BERT is encoder-only; MarkGPT is decoder-only.

**Epoch**
One complete pass through the entire training dataset.

**Fine-tuning**
Training a pre-trained model on a new task/dataset with smaller learning rate.

**Gradient**
Derivative of loss with respect to model parameters. Used to update weights during training.

**Hugging Face**
Popular NLP library and model hub. Source of many pre-trained models.

**Inference**
Running model to make predictions (as opposed to training).

**Language Model**
Model trained to predict next token given previous tokens. MarkGPT is a language model.

**LoRA (Low-Rank Adaptation)**
Parameter-efficient fine-tuning technique. Freezes model, trains small rank-r matrices. Reduces parameters by 250x.

**Loss**
Numerical measure of model error. Typically cross-entropy for language modeling.

**Multilingual**
Supporting multiple languages (English + Banso in MarkGPT).

**Perplexity**
Exponent of average negative log probability. Lower = better. Perplexity = exp(loss).

**Pre-training**
Training on large unsupervised dataset (e.g., Bible corpus). Before fine-tuning on downstream tasks.

**RoPE (Rotary Position Embeddings)**
Positional encoding method using rotation matrices. Better length extrapolation than absolute positions.

**Scaling Laws**
Empirical relationships between model size, data size, and performance. Chinchilla laws: optimal param-to-data ratio.

**Softmax**
Activation function that normalizes logits to probability distribution. Sum of probabilities = 1.

**Tokenization**
Process of converting text to tokens (subword units). BPE is MarkGPT's tokenization method.

**Transformer**
Architecture with self-attention and feed-forward layers. Foundation of modern LLMs (Vaswani et al., 2017).

**Vocabulary**
Set of all unique tokens. MarkGPT: 10,000 tokens.

## MarkGPT-Specific Terms

**MarkGPT Nano**
10M parameter model variant. Runs on mobile/edge devices.

**MarkGPT Small**
50M parameter model variant. Default deployment model.

**MarkGPT Base**
125M parameter model variant. Recommended for GPU servers.

**MarkGPT Medium**
350M parameter model variant. High accuracy, compute-intensive.

**Model Card**
Documentation describing model details, capabilities, limitations, and ethical considerations.

**Mixed Precision Training**
Training with both FP32 and FP16 operations. Faster and more memory-efficient.

**Flash Attention**
Optimized attention implementation. 2-4x faster on modern GPUs (A100, V100).

**Curriculum Learning**
Training strategy starting with easier examples, progressing to harder. Used in MarkGPT multilingual setup.

**Banso (Lamnso')**
Niger-Congo language spoken in Cameroon's Northwest Region. Community partner language for MarkGPT.

## Dataset & Data Terms

**Corpus**
Large collection of text data. Bible corpus (4M tokens), Banso corpus (300k tokens).

**Domain**
Topic area of data. MarkGPT primarily trained on religious (Bible) domain.

**Pre-processing**
Preparing raw text for training (cleaning, tokenization, normalization).

**Sequence Length**
Number of tokens in a single training example. MarkGPT: 1024 by default.

**Train/Val/Test Split**
Partition of data: training (80%), validation (10%), test (10%).

## Hardware Terms

**GPU (Graphics Processing Unit)**
Specialized processor optimized for parallel computations. Much faster than CPU for deep learning.

**CUDA**
NVIDIA's parallel computing platform. Required for training on NVIDIA GPUs.

**Memory (VRAM)**
GPU memory. Limits batch size and model size. Larger = faster training.

**Throughput**
Processing speed, usually measured in tokens/second.

**Latency**
Time for single prediction. Important for real-time applications.

## Ethics & Community Terms

**Bias**
Systematic unfairness in model outputs. E.g., gender bias (doctor = male).

**Fairness**
Equitable treatment across different groups. MarkGPT targets balanced multilingual fairness.

**Community Benefit**
Reciprocal advantages for communities contributing data/feedback. 5% revenue share with Banso community.

**Minority Language**
Language with fewer speakers and less training data. Banso is minority language (< 10k speakers).

**Language Preservation**
Effort to sustain and revitalize endangered languages. MarkGPT supports through training and tooling.

## Tools & Frameworks

**PyTorch**
Deep learning framework used by MarkGPT. Alternative: TensorFlow.

**Weights & Biases (W&B)**
Experiment tracking platform integrated into MarkGPT training.

**Docker**
Containerization platform for reproducible deployment.

**Git**
Version control system used for MarkGPT development and collaboration.

**Hugging Face Transformers**
Popular open-source NLP library. MarkGPT compatible with Hugging Face format.

## Abbreviations

| Abbr | Full Name |
|------|-----------|
| LLM | Large Language Model |
| NLP | Natural Language Processing |
| ML | Machine Learning |
| AI | Artificial Intelligence |
| BPE | Byte-Pair Encoding |
| RoPE | Rotary Position Embeddings |
| LoRA | Low-Rank Adaptation |
| W&B | Weights & Biases |
| CUDA | Compute Unified Device Architecture |
| VRAM | Video Random Access Memory |
| OOM | Out of Memory |
| GPU | Graphics Processing Unit |
| CPU | Central Processing Unit |
| FFNN | Feed-Forward Neural Network |
| MLP | Multi-Layer Perceptron |

---

**Glossary Version**: 1.0
**Last Updated**: 2024
**Maintained by**: MarkGPT Docs Team
