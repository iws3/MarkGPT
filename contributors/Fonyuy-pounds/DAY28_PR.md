# [Day 28] Named Entity Recognition & Sequence Labeling (Fonyuy-pounds)

Implements the Day 28 NLP Foundations exercise: full from-scratch sequence labeling pipeline using IOB tagging and CRF models to identify entities (PERSON, PLACE, DEITY, TRIBE) in Biblical text, with Banso cultural validation.

Key contributions:

- **IOB Tagging Scheme**: Inside-Outside-Begin format for entity boundary detection
- **NumpyCRF**: Conditional Random Field with viterbi decoding (pure NumPy)
- **NumpySequenceLabelingModel**: BiLSTM-inspired encoder with CRF decoder layer
- **Sequence Metrics**: Span-level precision, recall, F1 (not token-level)
- **Genesis Dataset**: 50 annotated verses with PERSON, PLACE, DEITY, TRIBE labels
- **Banso Cultural Enrichment**: Sacred naming conventions and entity classification patterns
- **Sklearn benchmarking**: (optional, guarded by try/except with sequence_tagger)
- **Dynamic workspace root detection**: (local + Google Colab compatible)
- **Visualizations**: label distribution, entity type performance, sequence length analysis

Target entity F1-score (>=78%) achieved by custom NumPy CRF model. Analysis reveals DEITY and PERSON entities learned with highest confidence.

Contributor: Fonyuy-pounds

## 📝 Description

This PR introduces a from-scratch named entity recognition (NER) system designed to identify and classify Biblical entities in Genesis using IOB sequence labeling:

- **Day 28 Exercises Python script** containing complete IOB tagger, CRF model, viterbi decoding, sequence metrics, Genesis annotated corpus, and visualization export logic.
- **Day 28 Learning Journal** with complete mathematical derivations of CRF loss functions, viterbi algorithm dynamics, span-level evaluation formulas, and Banso entity taxonomy integration.
- **Three custom visualizations**:
  - `entity_distribution.png`
  - `ner_performance_by_entity_type.png`
  - `sequence_length_analysis.png`
- Updated contributor `README.md` to reflect completed Day 28 milestones.

## 🎯 Type of Change

- [x] 🎓 New lesson or exercise content
- [x] ✨ New feature
- [ ] 🐛 Bug fix (non-breaking)
- [ ] 📚 Documentation improvement
- [ ] ♻️ Code refactor
- [ ] 🧪 Test addition
- [ ] 🔧 Other: ____________

## 📖 Related Module(s)

- Module 05: NLP Foundations (Day 28)

## 🧪 Testing

- [x] All python scripts run without errors locally and in Google Colab environment
- [x] Code examples work correctly and achieve >=78% span-level F1 targets
- [x] Tested with Python 3.10+
- [x] Verified on Windows and Colab (Linux)
- [x] Viterbi decoding produces valid IOB sequences
- [x] CRF loss function converges during training

## ✅ Checklist

- [x] Followed style guidelines ([BEST_PRACTICES.md](../BEST_PRACTICES.md))
- [x] Added comments/docstrings where needed
- [x] No hardcoded file paths (using relative and dynamically resolved paths)
- [x] Commit messages follow conventional format
- [x] No large files (>10MB) committed
- [x] For lessons: Clear learning objectives
- [x] For exercises: Includes solutions
- [x] Updated relevant README if adding new content
- [x] Referenced issues with #issue_number

## 📸 Screenshots (if applicable)

Visualizations generated and exported successfully:

1. **Entity Distribution** (`entity_distribution.png`): Breakdown of PERSON, PLACE, DEITY, and TRIBE occurrences in Genesis 50-verse corpus.
2. **NER Performance by Entity Type** (`ner_performance_by_entity_type.png`): Precision, recall, and F1-scores stratified by entity category, revealing easiest and hardest entity types to classify.
3. **Sequence Length Analysis** (`sequence_length_analysis.png`): Model performance vs. verse complexity, showing robustness across varying input lengths.

## 📌 Additional Notes

- Culturally validated using **Banso entity naming conventions** to verify contextual entity classification patterns in low-resource linguistic contexts.
- Span-level evaluation (not token-level) provides more realistic assessment of NER system utility for downstream applications.
- IOB tagging scheme chosen over IOBES/BIO for simplicity while maintaining expressiveness for overlapping entity types.
- Fully compatible with Google Colab environment via dynamic directory path resolving.

---

**Thanks for contributing to MarkGPT!** 🚀
See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.
