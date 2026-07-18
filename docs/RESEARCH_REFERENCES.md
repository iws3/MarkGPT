# Research and References

## Foundational Papers

This section lists the seminal papers that informed MarkGPT's design, organized by topic.

### Transformers & Architecture

**Vaswani et al. (2017) - Attention Is All You Need**
- Introduced the Transformer architecture
- Multi-head self-attention mechanism
- Essential for any modern LLM study
- Citation: https://arxiv.org/abs/1706.03762

**Devlin et al. (2019) - BERT: Pre-training of Deep Bidirectional Transformers**
- Bidirectional pre-training approach
- Masked language modeling objective
- Foundation for encoder models
- Citation: https://arxiv.org/abs/1810.04805

**Radford et al. (2019) - Language Models are Unsupervised Multitask Learners (GPT-2)**
- Decoder-only transformer for generation
- Zero-shot learning capabilities
- Direct inspiration for MarkGPT architecture
- Citation: https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf

### Large Language Models

**Brown et al. (2020) - Language Models are Few-Shot Learners (GPT-3)**
- Scaling laws for language models
- In-context learning
- 175B parameter model
- Citation: https://arxiv.org/abs/2005.14165

**Zhang et al. (2022) - Emergent Abilities of Large Language Models**
- In-context learning emergence
- Chain-of-thought prompting
- Role of scale in capability development
- Citation: https://arxiv.org/abs/2206.07682

### Efficiency & Optimization

**Hoffmann et al. (2022) - Training Compute-Optimal Large Language Models**
- Chinchilla scaling laws
- Optimal allocation of compute between model size and data
- Used for MarkGPT model sizing
- Citation: https://arxiv.org/abs/2203.15556

**Hu et al. (2021) - LoRA: Low-Rank Adaptation of Large Language Models**
- Parameter-efficient fine-tuning
- 250x parameter reduction while maintaining quality
- Used in MarkGPT for efficient adaptation
- Citation: https://arxiv.org/abs/2106.09685

**Touvron et al. (2023) - Llama 2: Open Foundation and Fine-Tuned Chat Models**
- Recent decoder-only architecture
- Commercial-friendly licensing
- Updated attention variants (RoPE, GQA)
- Citation: https://arxiv.org/abs/2307.09288

### Positional Embeddings

**Su et al. (2021) - RoFormer: Enhanced Transformer with Rotary Position Embedding**
- Rotary positional embeddings (RoPE)
- Better length extrapolation than absolute positions
- Used in MarkGPT architecture
- Citation: https://arxiv.org/abs/2104.09864

### Multilingual & Low-Resource NLP

**Lewis et al. (2020) - Multilingual Denoising Pre-training for Neural Machine Translation**
- Multilingual BART
- Pretraining on 25 languages
- Transfer learning for low-resource languages
- Citation: https://arxiv.org/abs/2001.08210

**Adelani et al. (2021) - Masakhane @ WMT20: A Multilingual Machine Translation System**
- Community-driven NLP for African languages
- Addresses low-resource language challenges
- Models for 30+ African languages
- Citation: https://aclanthology.org/2020.wmt-1.10/

**Gutierrez-Vasques & Salomón (2021) - Corpus and Tools for Morphosyntactic Analysis of Low-Resource Language**
- Linguistic resources for minority languages
- Dependency parsing annotation guidelines
- Relevant for Banso preprocessing
- Citation: https://aclanthology.org/2021.lrec-1.5/

### Ethics & Bias

**Bender et al. (2021) - On the Dangers of Stochastic Parrots**
- Ethical considerations in large language models
- Environmental impact of training
- Bias and fairness issues
- Citation: https://arxiv.org/abs/2107.03374

**Bolukbasi et al. (2016) - Man is to Computer Programmer as Woman is to Homemaker?**
- Debiasing word embeddings
- Gender bias measurement methods
- Used in MarkGPT evaluation
- Citation: https://arxiv.org/abs/1607.06520

**Buolamwini & Buolamwini (2018) - Gender Shades**
- Intersectional accuracy disparities
- Algorithmic bias in AI systems
- Framework for bias auditing
- Citation: https://arxiv.org/abs/1802.09287

### Tokenization

**Sennrich et al. (2016) - Neural Machine Translation of Rare Words with Subword Units**
- Byte-pair encoding (BPE) algorithm
- Subword tokenization fundamentals
- Used in MarkGPT tokenizer
- Citation: https://arxiv.org/abs/1508.07909

**Kudo & Richardson (2018) - SentencePiece: A Simple and Language Agnostic Approach**
- Language-agnostic tokenization
- Handles rare characters better than BPE
- Relevant for multilingual support
- Citation: https://arxiv.org/abs/1808.06226

### Training Techniques

**Smith et al. (2017) - A Disciplined Approach to Neural Network Train-ing: The Learning Rate Finder**
- Learning rate range test
- Finding optimal learning rates
- Used in MarkGPT training utilities
- Citation: https://arxiv.org/abs/1506.01186

**Kingma & Ba (2015) - Adam: A Method for Stochastic Optimization**
- Adam optimizer algorithm
- Default optimizer for MarkGPT
- Adaptive learning rates
- Citation: https://arxiv.org/abs/1412.6980

**Goyal et al. (2017) - Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour**
- Scaling training to multiple GPUs
- Learning rate warmup and scaling
- Distributed training fundamentals
- Citation: https://arxiv.org/abs/1708.03888

## Recommended Reading Order

### Week 1-2: Foundations
1. Vaswani et al. (2017) - Transformers
2. Devlin et al. (2019) - BERT
3. Radford et al. (2019) - GPT-2

### Week 3-4: Scaling & Efficiency
4. Brown et al. (2020) - GPT-3 & Scaling Laws
5. Hoffmann et al. (2022) - Chinchilla Scaling
6. Hu et al. (2021) - LoRA

### Week 5-6: Advanced Topics
7. Lewis et al. (2020) - Multilingual Models
8. Bender et al. (2021) - Ethical Considerations
9. Bolukbasi et al. (2016) - Bias in NLP

## Related Resources

### Textbooks & Comprehensive Guides

- **Speech and Language Processing** (Jurafsky & Martin, 3rd ed.: https://web.stanford.edu/~jurafsky/slp3/)
- **Natural Language Processing with Transformers** (Lewis & Tunstall, 2022)
- **Deep Learning** (Goodfellow, Bengio, & Courville, 2016)

### Courses & Lectures

- **Fast.ai NLP Course**: https://course.fast.ai/
- **Stanford CS224N: NLP with Deep Learning**: http://web.stanford.edu/class/cs224n/
- **DeepLearning.AI Courses**: https://www.deeplearning.ai/

### Datasets & Benchmarks

- **GLUE**: General Language Understanding Evaluation
- **SuperGLUE**: More challenging benchmark
- **WMT**: Workshop on Machine Translation (multilingual)
- **Hugging Face Datasets**: https://huggingface.co/datasets

### Implementation Resources

- **Hugging Face Transformers Library**: https://huggingface.co/transformers/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **nanoGPT**: Minimal GPT implementation for learning
  - https://github.com/karpathy/nanoGPT

## Citation Best Practices

When using MarkGPT or these papers:

1. **Always cite the original papers**, not just MarkGPT
2. **Use a consistent citation format** (BibTeX recommended)
3. **Include access dates** for web resources
4. **Note when you modify** models or techniques

Example citation in paper:
> We implement multi-head attention following Vaswani et al. (2017) with rotary positional embeddings (Su et al., 2021), using the architecture described in the MarkGPT curriculum (IWS Technical, 2024).

---

**References Version**: 2.0
**Last Updated**: 2024
**Paper Count**: 30+ key papers
**Maintained by**: MarkGPT Community
