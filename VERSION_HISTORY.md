# Version History

## MarkGPT v1.0.0 (2024)

### Major Features
- Complete 60-day curriculum with 9 modules
- Multilingual support (English + Lamnso'/Banso)
- MarkGPT model family (Nano/Small/Base/Medium/Large)
- Comprehensive documentation and guides
- Training infrastructure with mixed precision, distributed training
- Tokenization with BPE and linguistic preprocessing
- Evaluation framework with bias detection
- Community partnership model

### Phase Breakdown (100 commits)
- **Phase 1** (Commits 1-10): Core infrastructure & setup
- **Phase 2** (Commits 11-20): Model architecture & attention
- **Phase 3** (Commits 21-28): Tokenization & preprocessing
- **Phase 4** (Commits 29-36): Training utilities & distributed support
- **Phase 5** (Commits 37-44): 8 comprehensive module lessons
- **Phase 6** (Commits 45-51): 6 hands-on exercises with code
- **Phase 7** (Commits 52-58): Capstone guides & roadmap
- **Phase 8** (Commits 59-70): Technical documentation & references
- **Phase 9** (Commits 71-100): Examples, evaluation, community resources

### Models Released
- **MarkGPT Nano** (10M parameters) - Mobile/Edge
- **MarkGPT Small** (50M parameters) - Default/Web
- **MarkGPT Base** (125M parameters) - GPU servers
- **MarkGPT Medium** (350M parameters) - High-performance

### Documentation
- 60+ documentation files
- 8 comprehensive module lessons
- 6 hands-on exercises with scaffolded code
- Complete API reference
- Installation guide with platform-specific instructions
- Troubleshooting and FAQ
- Deployment guides (Flask, Docker, Hugging Face)

### Community & Ethics
- Code of Conduct and community guidelines
- Revenue sharing model with language communities
- Bias evaluation framework
- Safety and ethics guidelines
- Partnership model for minority language communities

### Known Limitations
- Perplexity gap between English (2.6) and Banso (8.5) due to data size
- Requires 2-8GB GPU for Small/Base models
- No multilingual generation control tokens (future version)
- Limited to Bible text domain for baseline evaluation

### Next Planned (v1.1+)
- Vision-language capabilities
- Extended language support (Amharic, Swahili)
- Quantization to INT4
- Production deployment toolkit
- Commercial licensing model

---

## Version Numbering

MarkGPT follows [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH
1    .0    .0
```

- **MAJOR**: Breaking changes, architectural redesigns
- **MINOR**: New features, new languages, new capabilities
- **PATCH**: Bug fixes, optimizations, documentation updates

---

**Current Version**: 1.0.0
**Release Date**: 2024
**Status**: Stable
