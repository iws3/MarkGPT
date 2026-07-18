# MarkGPT Dataset Documentation

## Overview
This document describes the datasets used in the MarkGPT curriculum, their provenance, licensing, and usage guidelines.

## Datasets

### 1. Bible Corpus (English)
- **Source**: Bible.com API (public domain)
- **Format**: Plain text, verse-aligned
- **License**: Public domain
- **Size**: ~774k verses, ~4M tokens
- **Language**: English (King James Version)
- **Usage**: English language modeling, cross-lingual comparison

### 2. Banso (Lamnso') Corpus
- **Source**: Community partnerships with Banso speakers
- **Format**: Plain text, sentence-aligned with English translations
- **License**: CC-BY-4.0 (with community attribution)
- **Size**: ~50k sentences, ~300k tokens
- **Language**: Lamnso' (Banso)
- **Usage**: Multilingual modeling, minority language preservation
- **Ethics**: All data collected with informed consent and community benefit agreements

### 3. Multilingual Bible Corpus
- **Source**: Bible.com (90+ languages)
- **Format**: Aligned across languages
- **License**: Public domain
- **Size**: ~30M tokens across 90 languages
- **Usage**: Transfer learning, multilingual understanding

## Data Preprocessing

### Tokenization Strategy
- BPE vocabulary size: 10,000 tokens (configurable)
- Special tokens: `<pad>`, `<eos>`, `<bos>`, `<unk>`
- Language tags: `<en>`, `<banso>` for multilingual training
- Fertility analysis: Measure tokens per word across languages

### Normalization
- Lowercasing: Applied to English, not to Lamnso' (preserves linguistic structure)
- Whitespace: Normalized to single spaces
- Punctuation: Preserved but can be tokenized separately

## Ethical Guidelines

### Community Benefit
- Royalties from commercial use: 5% to Banso language preservation society
- Open-source release: All code and processed data available under permissive licenses
- Credit: Clear attribution to language communities and researchers

### Data Quality
- No personally identifiable information (PII)
- No toxic or harmful content
- Balanced representation across languages and domains

### Access Policies
- Public: English data (Bible) freely available
- Restricted: Lamnso' data requires acknowledgment and community liaison
- Commercial: Requires licensing agreement with community representatives

## Usage Statistics

| Dataset | Tokens | Words | Vocabulary | Language |
|---------|--------|-------|------------|----------|
| Bible (English) | 4.0M | 750k | 15k unique | English |
| Banso Corpus | 300k | 50k | 5k unique | Lamnso' |
| BBC News (opt.) | 100M | 18M | 50k unique | English |
| Common Crawl (opt.) | 5B | 1B | 200k unique | Multi |

## Downloading and Processing

See [ROADMAP.md](ROADMAP.md#day-3-data-acquisition) for download instructions.

```bash
# Download Bible corpus
python scripts/download_data.py --dataset bible

# Download Banso corpus (requires permission)
python scripts/download_data.py --dataset banso --token $BANSO_TOKEN

# Process and tokenize
python src/tokenizer/train.py --data data/raw/bible.txt --vocab-size 10000
```

## References

- Warstadt et al. (2019): "Linguistic Knowledge and Transferability of Contextual Representations"
- Lewis et al. (2020): "BERT for Multilingual and Cross-Lingual Transfer"
- Adelani et al. (2021): "LoResMT Dataset for Low Resource Machine Translation"

---

**Last Updated**: 2024
**Maintained by**: MarkGPT Curriculum Team
