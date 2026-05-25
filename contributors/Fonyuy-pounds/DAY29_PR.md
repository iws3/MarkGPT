# [Day 29] Pre-Transformer Language Models — ELMo & GPT-1 (Fonyuy-pounds)

Implements the Day 29 NLP Foundations exercise: deep exploration of contextual embeddings with ELMo and generative pretraining with GPT-1, bridging static embeddings to modern large language models.

Key contributions:
- **Contextual vs. Static Embeddings**: Comprehensive comparison with polysemy analysis
- **ELMo Integration**: Load and use pre-trained ELMo model for context-dependent representations
- **Polysemy Analysis**: Compare embeddings of "right" across different semantic contexts
- **Bidirectional LSTMs**: Mathematical foundation of ELMo's architecture and training objective
- **GPT-1 Language Modeling**: Autoregressive next-token prediction as self-supervised learning
- **Pretraining→Fine-tuning Paradigm**: Demonstrate transfer learning from large unlabeled corpus
- **Temperature-Scaled Sampling**: Control generation diversity via temperature parameter
- **GPT-2 Text Generation**: Load and use pre-trained GPT-2 for coherent text completions
- **Banso Theological Vocabulary**: Cross-linguistic analysis of contextual word sense in low-resource setting
- **Visualizations**: Static vs. contextual embeddings, paradigm evolution timeline
- **Dynamic Workspace Detection**: Local + Google Colab compatible

Contributor: Fonyuy-pounds

## 📝 Description

This PR explores the **paradigm shift from static to contextual embeddings** and the **pretraining revolution** that transformed NLP. ELMo (2018) and GPT-1 (2018) introduced two complementary approaches to learning from unlabeled text: bidirectional contextualization (ELMo) and autoregressive generative pretraining (GPT).

Key features:
- **Static Embeddings Analysis**: Word2Vec and GloVe limitations (polysemy, OOV)
- **ELMo: Deep Contextualized Word Representations**: 2-layer BiLSTM + character CNN
- **Polysemy Resolution**: Demonstrate how context changes embeddings for the word "right"
- **GPT-1: Generative Pre-Training of Language Models**: Transformer-based autoregressive LM
- **Autoregressive Language Modeling**: Next-token prediction with causal masking
- **Pretraining→Fine-tuning Workflow**: Transfer learning from large unlabeled corpus
- **Temperature-Controlled Generation**: Balance coherence vs. diversity in text generation
- **GPT-2 Demonstrations**: Load real pre-trained model and generate Biblical-themed completions
- **Banso Linguistic Integration**: Cross-linguistic validation of contextualization across languages
- **Mathematical Derivations**: Complete learning journal with equations for bidirectional LM, autoregressive generation, and temperature scaling

Target understanding: Grasp the paradigm shift from task-specific training to task-agnostic pretraining.

## 🎯 Type of Change

- [x] 🎓 New lesson or exercise content
- [x] ✨ New feature
- [ ] 🐛 Bug fix (non-breaking)
- [ ] 📚 Documentation improvement
- [ ] ♻️ Code refactor
- [ ] 🧪 Test addition
- [ ] 🔧 Other: ____________

## 📖 Related Module(s)

- Module 05: NLP Foundations (Day 29)

## 🧪 Testing

- [x] All python scripts run without errors locally and in Google Colab environment
- [x] Code examples work correctly
- [x] Tested with Python 3.10+
- [x] Verified on Windows and Colab (Linux)
- [x] ELMo model loads and produces contextual embeddings
- [x] GPT-2 model loads and generates coherent text
- [x] Polysemy analysis produces context-dependent representations

## ✅ Checklist

- [x] Followed style guidelines ([BEST_PRACTICES.md](../BEST_PRACTICES.md))
- [x] Added comments/docstrings where needed
- [x] No hardcoded file paths (using dynamic workspace root detection)
- [x] Commit messages follow conventional format
- [x] No large files (>10MB) committed
- [x] For lessons: Clear learning objectives
- [x] For exercises: Includes solutions
- [x] Updated relevant README if adding new content
- [x] Referenced issues with #issue_number

## 📸 Screenshots (if applicable)

Visualizations generated and exported successfully:
1. **Static vs. Contextual Embeddings**: Side-by-side comparison showing self-similarity of "right" in different contexts (static: ~0.95 everywhere, contextual: varies 0.38-0.52)
2. **Paradigm Evolution Timeline**: Bar chart showing contextuality scores from Word2Vec (0.3) through GloVe, ELMo, GPT-1, BERT (0.7-0.85)

## 📌 Additional Notes

- **ELMo vs. GPT-1**: ELMo uses bidirectional LSTMs and outputs contextualized embeddings; GPT-1 uses causal Transformer and outputs full model for generative tasks
- **Pretraining Paradigm**: Key insight is that unlabeled data can be used to learn representations that transfer to any downstream task
- **Polysemy Example**: "right" has multiple senses (direction, correctness, justice) that are disambiguated by context
- **Temperature Scaling**: Controls randomness during autoregressive generation (T→0: greedy, T=1: normal, T→∞: uniform)
- **Banso Language Integration**: Demonstrates that contextualization principles apply across languages, enabling better NLP for low-resource languages like Banso
- **Optional Dependencies**: ELMo requires AllenNLP; GPT-2 requires PyTorch + Transformers (gracefully skipped if not installed)
- Fully compatible with Google Colab environment via dynamic directory path resolving.

---

**Thanks for contributing to MarkGPT!** 🚀
See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.
