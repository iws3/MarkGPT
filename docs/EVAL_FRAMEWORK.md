# Model Evaluation Framework

## Overview

Guide for evaluating MarkGPT models across multiple dimensions.

## Evaluation Dimensions

### 1. Language Modeling Quality

**Perplexity**
```python
from src.utils.eval import evaluate_model
import math

metrics = evaluate_model(model, test_loader)
perplexity = math.exp(metrics['loss'])  # Target: 2.6 for English
```

**Targets by Language**:
- English (Bible): Perplexity < 3.0
- Banso (multilingual): Perplexity < 10.0
- Multilingual (balanced): English < 3.0, Banso < 10.0

### 2. Downstream Tasks

**Text Classification**
- Sentiment analysis accuracy
- Topic classification
- Bias detection

**Text Generation**
- Human evaluation (coherence, factuality, grammar)
- Automatic metrics (BLEU, ROUGE)
- Toxicity filtering

### 3. Fairness & Bias

**Gender Bias Measurement** (Bolukbasi et al. 2016)
```python
male_bias = analyze_gender_bias(model, test_pairs)
# Target: < 5% absolute difference
```

**Language Bias**
```python
lang_disparity = banso_perplexity / english_perplexity
# Target: < 3.0x ratio
```

### 4. Robustness

**Out-of-distribution Performance**
- Test on unseen domains
- Different text styles
- Code-switching between languages

**Adversarial Examples**
- Typo tolerance
- Synonym substitution
- Grammatical errors

## Evaluation Scripts

### Full Evaluation Pipeline

```bash
# Evaluate all dimensions
python scripts/eval_full.py \\
    --model-path checkpoints/markgpt-small.pt \\
    --device cuda

# Output: evaluation_report_2024-01-15.json
```

### Specific Evaluations

```bash
# Perplexity on held-out test set
python scripts/eval_perplexity.py \\
    --model markgpt-small \\
    --data-path data/test.txt

# Benchmark inference speed
python scripts/eval_benchmark.py \\
    --model markgpt-small \\
    --batch-sizes 1,8,32,64

# Bias evaluation
python scripts/eval_bias.py \\
    --model markgpt-small \\
    --framework bolukbasi
```

## Evaluation Metrics Reference

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| Perplexity | exp(-1/N Σ log P(x)) | Lower is better (target: 3.0) |
| Accuracy | # correct / # total | % correct predictions (0-100%) |
| Precision | TP / (TP + FP) | % of positive predictions correct |
| Recall | TP / (TP + FN) | % of actual positives detected |
| F1 Score | 2×(P×R)/(P+R) | Harmonic mean of precision & recall |
| BLEU | Cumulative n-gram overlap | For sequence-to-sequence tasks |
| ROUGE | Recall of n-gram overlap | For summarization evaluation |

## Human Evaluation

### Setup
1. Generate 100 samples from model
2. Recruit 3 evaluators (native speakers preferred)
3. Use rubric below
4. Calculate inter-annotator agreement (Cohen's kappa)

### Evaluation Rubric

**Coherence (1-5)**
- 1: Nonsensical, incoherent
- 3: Mostly coherent but some confusion
- 5: Fully coherent, natural flow

**Factuality (1-5)**
- 1: Mostly false or hallucinated facts
- 3: Mix of truth and fabrication
- 5: Factually accurate (verifiable with source data)

**Grammar (1-5)**
- 1: Many grammatical errors
- 3: Few grammatical errors
- 5: Grammatically perfect

**Example Annotation**
```json
{
  "sample_id": "001",
  "prompt": "John 3:16",
  "generation": "For God so loved the world that he gave his only begotten son...",
  "coherence": 5,
  "factuality": 5,
  "grammar": 5,
  "notes": "Perfect match to source text"
}
```

## Benchmark Results

### MarkGPT Performance

**English (Bible test set, 500 examples)**
```
Nano (10M):     Perplexity = 5.2  │ Human Eval = 2.8/5
Small (50M):    Perplexity = 3.8  │ Human Eval = 3.6/5
Base (125M):    Perplexity = 2.6  │ Human Eval = 4.1/5
Medium (350M):  Perplexity = 2.1  │ Human Eval = 4.3/5
```

**Banso (100 examples from held-out corpus)**
```
Small (50M):    Perplexity = 18.4 │ Human Eval = 2.4/5
Base (125M):    Perplexity = 12.1 │ Human Eval = 3.1/5
```

**Multilingual (70% EN / 30% Banso)**
```
English Regression:  +0.2 perplexity (7.7% increase)
Banso Capability:    Perplexity reduced from 21.2 to 8.5 (60% improvement)
```

## Continuous Evaluation

### Automated Checks

After each training run:
```bash
# Check for regressions
python scripts/check_regressions.py \\
    --baseline baseline_metrics.json \\
    --current current_metrics.json \\
    --threshold 0.05  # Fail if >5% regression
```

### W&B Logging

```python
# Log metrics during training
wandb.log({
    "train/loss": loss,
    "val/perplexity": perplexity,
    "val/gen_coherence": coherence_score,
    "metrics/bias_disparity": gender_bias
})
```

## Performance Targets

Model acceptance requires:

- ✅ Perplexity within 5% of baseline
- ✅ No increase in bias metrics
- ✅ All inference tests pass
- ✅ Throughput maintained
- ✅ Memory footprint within spec

---

**Framework Version**: 1.0
**Last Updated**: 2024
**Maintained by**: MarkGPT ML Team
