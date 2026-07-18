# Safety, Bias, and Ethical Considerations

## Overview
This document outlines ethical principles, known risks, and mitigation strategies for developing and deploying MarkGPT models, especially in multilingual and minority language contexts.

## Ethical Principles

### 1. Community Benefit
- ✅ All Banso data collected with informed consent and community agreements
- ✅ Revenue sharing: 5% of commercial use to language preservation
- ✅ Open source: Code and models freely available under permissive licenses
- ✅ Community advisory board: Decision-making power for language communities

### 2. Fair Representation
- ✅ Diverse curriculum: Balances English and minority language perspectives
- ✅ Equitable training: Same compute allocated to all languages during multilingual training
- ✅ Accessibility: Free resources for underrepresented communities

### 3. Transparency
- ✅ Model cards: Detailed documentation of what models do and don't do
- ✅ Data provenance: Clear sourcing and licensing for all datasets
- ✅ Limitations: Explicit discussion of model failure modes

## Known Risks and Biases

### Language Model Risks

#### 1. **English-Centric Bias**
- **Problem**: English has 100x more pre-training data, creating language imbalance
- **Symptom**: Lamnso' model generates English tokens when uncertain
- **Mitigation**:
  - Use `language_prefix` tokens (`<banso>` vs `<en>`)
  - Monitor per-language perplexity during training
  - Fine-tune with Banso-only data to anchor language representations
  - Use curriculum learning to prioritize underrepresented language early

#### 2. **Stereotyping in Bible Data**
- **Problem**: Bible contains gender stereotypes, cultural assumptions, outdated language
- **Symptom**: Model perpetuates gender role associations (doctor=male, nurse=female)
- **Mitigation**:
  - Filter for gender-stereotyping language patterns
  - Add balanced counter-examples during fine-tuning
  - Evaluate on gender bias benchmarks (e.g., Bolukbasi et al. 2016)
  - Include discussion of stereotypes in curriculum

#### 3. **Religious Content Bias**
- **Problem**: Bible data is religious; models may favor theological interpretations
- **Symptom**: Model less capable with secular or non-Christian perspectives
- **Mitigation**:
  - Supplement with secular data (Wikipedia, news) if diversity needed
  - Acknowledge bias in model card
  - Guidance to end users: "This model is trained on Bible data; expect religious perspectives"

#### 4. **Minority Language Underrepresentation**
- **Problem**: Lamnso' data is small (300k vs 4M English tokens)
- **Symptom**: Lower quality generations, less reliable factuality
- **Mitigation**:
  - Use data augmentation: Backtranslation, paraphrasing, synthetic data
  - Transfer learning from English improves Lamnso' perplexity by ~12%
  - Explicit curriculum: Train on Banso data after English pre-training convergence

### Downstream Harm Risks

#### 1. **Misinformation**
- **Risk**: Model generates plausible but false information
- **Example**: False genealogies, invented Bible verses
- **Mitigation**:
  - Prompt: "Provide citations from the source text"
  - Guardrails: Block generation of verifiable claims without training data support
  - Evaluation: Fact-checking benchmarks before deployment

#### 2. **Cultural Appropriation**
- **Risk**: Model used to generate "AI-art" mocking minority cultures
- **Mitigation**:
  - License: CC-BY-SA requires attribution and sharing alike
  - Usage terms: Disallow use in commercial products without community approval
  - Monitoring: Track deployed instances for misuse

#### 3. **Language Homogenization**
- **Risk**: Creating one "standard" Lamnso' variant suppresses natural variation
- **Mitigation**:
  - Train on dialectal variation from different speakers
  - Curriculum lesson on linguistic diversity (L09.1)
  - Community feedback: Collect feedback from Banso speakers quarterly

#### 4. **Replacement of Speakers**
- **Risk**: "AI speaker" replaces actual community members (e.g., for church services)
- **Mitigation**:
  - Clear messaging: "AI is supplement, not replacement"
  - Marketing: Feature Banso community members in case studies
  - Policy: Licensing terms prohibit use for voice synthesis without human consent

## Bias Evaluation Framework

### Quantitative Metrics

#### 1. **Perplexity by Demographic**
```python
# Measure if model performs better on English than Banso
banso_ppl = evaluate(model, banso_test_set)
english_ppl = evaluate(model, english_test_set)
disparity_ratio = banso_ppl / english_ppl

# Target: < 1.3x (ideal: < 1.1x)
assert disparity_ratio < 1.3, f"Language disparity: {disparity_ratio}"
```

#### 2. **Gender Bias in Next-Token Prediction**
```python
# Contrastive evaluation (Bolukbasi et al. 2016)
sentence_a = "The doctor is [male/female]"
# Measure P(male | context) vs baseline

# Target: < 5% absolute difference between gendered pairs
```

#### 3. **Topic Representation**
```python
# Measure if model generates equal depth across topics
topics = ["agriculture", "spirituality", "family", "governance"]
for topic in topics:
    prompt = f"Tell me about {topic}"
    length = generate_until_eos(prompt)
    disparity = max(lengths) - min(lengths)
    assert disparity < 50, "Topic representation imbalance"
```

#### 4. **Language Representation in Multilingual Model**
```python
# For every 100 generated tokens, measure language distribution
multilingual_generations = sample_from_model(n=1000)
language_dist = measure_language_distribution(multilingual_generations)

# Target: ±10% deviation from intended mix (e.g., if 30% Banso in training, expect 20-40% in outputs)
assert 0.2 < language_dist["banso"] < 0.4
```

### Qualitative Review

- [ ] Manual review of 100 random generations by native speakers
- [ ] Assessment of stereotypical content (religious, gender, cultural)
- [ ] Evaluation by community advisory board (quarterly)
- [ ] Feedback from downstream applications

## Deployment Safeguards

### Before Release
- [ ] Bias evaluation completed and documented
- [ ] Community stakeholders approved release
- [ ] Model card prepared with limitations section
- [ ] Content filtering in place for sensitive generations

### During Use
- [ ] Monitor for common failure modes (hallucinated facts, stereotyping)
- [ ] Log model outputs for audit (with user consent)
- [ ] Version control: Deprecate and fix biased models quickly

### Post-Deployment
- [ ] Quarterly community feedback collection
- [ ] User reports of harms addressed within 7 days
- [ ] Model updates released with bias fixes
- [ ] Annual external audit by independent researchers

## Community Benefit Model

### Revenue Sharing
```
Commercial Use License Fee: $50,000/year
├─ Banso Language Preservation Society: $2,500 (5%)
├─ University Research Fund: $2,500 (5%)
├─ Community Education: $2,000 (4%)
└─ Company Profit: $43,000 (86%)
```

### Non-Profit & Academic
- ✅ Free for research and non-profit use
- ✅ Attribution only requirement
- ✅ Community benefit projects prioritized for compute grants

## References

- Bolukbasi et al. (2016): "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings"
- Buolamwini & Buolamwini (2018): "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification"
- Mitchell et al. (2019): "Model Cards for Model Reporting"
- Bender & Friedberg (2021): "On the Dangers of Stochastic Parrots"
- Sutawika et al. (2022): "The IndoNLG Benchmark: Benchmarking Natural Language Generation for the Indonesian Language"

---

**Policy Version**: 1.0
**Last Updated**: 2024
**Next Review**: Q2 2024
