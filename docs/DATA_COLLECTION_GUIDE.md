# Data Collection & Annotation Guide

## Overview

Guide for collecting, preparing, and annotating data for MarkGPT fine-tuning or evaluation.

## Data Collection Methods

### Web Scraping

```python
import requests
from bs4 import BeautifulSoup

def scrape_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(['script', 'style']):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Clean up
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text
```

### APIs

```python
# Bible API example
import requests

response = requests.get('https://www.bible.com/api/2/verses?passage=John%203:16')
verse = response.json()['passages'][0]
text = verse['text']
```

### Community Contributions

For minority language data (e.g., Banso):
1. **Partner with language community**: Establish benefit agreement
2. **Collect from institutions**: Churches, schools, cultural centers
3. **Crowdsource**: Via platform like Amazon Mechanical Turk, Appen
4. **Incentivize**: Pay contributors fairly for their time

**Sample Agreement**:
```
- Community receives 5% of commercial usage revenue
- Open source release with CC-BY attribution
- Community advisory board on deployment decisions
- Annual reporting on usage and impact
```

## Data Preparation

### Formatting

**Standard format** (one sentence per line):
```
This is the first sentence.
This is the second sentence.
```

**Bilingual format** (with language tags):
```
<en> And God said, Let there be light
<banso> Ɨ Chíí tìŋ, Bâ kɨ̀ kɔ́n
```

### Cleaning

```python
import re

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove control characters
    text = ''.join(ch for ch in text if ch.isprintable() or ch.isspace())
    
    return text
```

### Deduplication

```python
def deduplicate(texts):
    seen = set()
    unique = []
    
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    
    return unique
```

### Privacy & Ethics

**Before collecting**, audit for:
- [ ] PII (names, addresses, phone numbers) → Remove or anonymize
- [ ] Sensitive content (medical, financial) → Consider exclusion
- [ ] Outdated/offensive content → Review carefully
- [ ] Copyright issues → Verify permissions

## Annotation

### Task Types

#### 1. Quality Rating (1-5)

```
Text: "And God said, Let there be light."
Quality: [5]
 1 = Nonsensical/corrupted
 2 = Poor quality with errors
 3 = Acceptable
 4 = Good quality
 5 = Excellent/canonical
```

#### 2. Topic Labeling

```
Text: "Jesus walked along the shore."
Topics: [religion, narrative, nature]
```

#### 3. Language Identification

```
Text: "Ɨ Chíí tìŋ"
Language: [banso]
Confidence: [high/medium/low]
```

#### 4. Toxicity Detection

```
Text: "I hate strangers."
Toxicity: [no/mild/severe]
Category: [hate_speech/violence/offensive/none]
```

### Annotation Workflow

#### Step 1: Define Spec
```markdown
# Annotation Spec: Quality Rating

## Task
Rate text quality from 1-5

## Guidelines
- Rate only the text quality
- Ignore domain-specific correctness
- Examples:
  - "And God said" → 5 (well-formed)
  - "dn gd sd" → 1 (corrupted)

## Time per item
~10-30 seconds
```

#### Step 2: Create Instructions
- Clear task description
- 3-5 examples with explanations
- Edge cases and how to handle
- Quality control checks

#### Step 3: Inter-Annotator Agreement
```python
from sklearn.metrics import cohen_kappa_score

annotator_1 = [5, 4, 3, 5, 2]
annotator_2 = [5, 4, 3, 4, 2]

kappa = cohen_kappa_score(annotator_1, annotator_2)
# Score: 0.8 = Good agreement
# Target: > 0.7
```

#### Step 4: Quality Assurance
- [ ] At least 2 annotators per item
- [ ] Cohen's kappa > 0.7
- [ ] Majority vote decides if conflict
- [ ] 10% gold standard checks

### Annotation Platforms

| Platform | Cost | Use Case |
|----------|------|----------|
| Prodigy | $0.06/item avg | In-house; flexible |
| Mechanical Turk | $0.01-0.50/item | Cheap volume |
| Appen | $0.10-1.00/item | Quality priority |
| Label Studio (self-hosted) | Free | Open source |
| Doccano (self-hosted) | Free | Open source |

### Handling Disagreements

```python
# If annotators disagree:
if annotator_1_score != annotator_2_score:
    # Option 1: Majority vote (3+ annotators)
    final_score = majority_vote([a1, a2, a3])
    
    # Option 2: Average (for continuous scales)
    final_score = (annotator_1_score + annotator_2_score) / 2
    
    # Option 3: Expert review
    final_score = expert_annotator_review(item)
```

## Dataset Statistics

### Check Dataset Health

```python
import numpy as np

def analyze_dataset(texts):
    lengths = [len(text.split()) for text in texts]
    
    print(f"Total texts: {len(texts)}")
    print(f"Avg length: {np.mean(lengths):.1f} words")
    print(f"Min/Max: {np.min(lengths)} / {np.max(lengths)}")
    print(f"Std dev: {np.std(lengths):.1f}")
    
    # Check for duplicates
    unique = len(set(texts))
    dup_pct = (1 - unique / len(texts)) * 100
    print(f"Duplicates: {dup_pct:.1f}%")
```

### Data Distribution

```python
# For multilingual data
import matplotlib.pyplot as plt

languages = ['en', 'banso']
counts = [4000, 300]  # Number of samples

plt.figure(figsize=(8, 4))
plt.bar(languages, counts)
plt.ylabel('Number of samples')
plt.title('Dataset language distribution')
plt.savefig('distribution.png')
```

## Version Control

```bash
# Track dataset version
git add data/processed/dataset_v1.0.txt
git commit -m "data: add curated dataset v1.0 (4.3k samples, cleaned)"

# Add metadata
echo "version: 1.0
created: 2024-01-15
size: 4300 samples
languages: en, banso
quality: human-reviewed (kappa=0.82)" > data/processed/dataset_v1.0.meta.txt
```

## Privacy & Consent

For any community data collection:

```markdown
# Data Collection Consent Form

By contributing text data to MarkGPT, you agree that:

1. Your data will be used to train language models
2. Your data may be shared in open-source datasets (with attribution)
3. You hereby license your data under CC-BY-4.0
4. You have permission to share this data
5. You understand your name will be credited

I agree: [☐ Yes / ☐ No]
Name: _____________
Date: _____________
```

---

**Guide Version**: 1.0
**Last Updated**: 2024
**Maintained by**: MarkGPT Data Team
