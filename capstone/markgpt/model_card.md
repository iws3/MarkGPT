# MarkGPT Model Card
## A Language Model for Biblical and Banso Vernacular Text

---

## Model Overview

**Model name:** MarkGPT-Small (v1.0)  
**Architecture:** Decoder-only Transformer (GPT-style)  
**Parameters:** ~10M (configurable: Nano 2M, Small 10M, Base 85M)  
**Training data:** King James Bible + Banso/Lamnso' vernacular corpus  
**Developed by:** [Your name — you built this!]  
**Language(s):** English (Biblical register) + Banso/Lamnso' (Cameroonian vernacular)  
**License:** MIT  

---

## Intended Use

MarkGPT was developed as both an educational project (demonstrating LLM training from scratch) and a cultural preservation effort (representing the Banso vernacular in an AI system).

**Primary uses:**
- Generating text in the style of Biblical English (KJV register)
- Generating text with Banso/Lamnso' vocabulary and idiomatic patterns
- Completing verse-style sentences in mixed Biblical-Banso style
- Educational demonstration of autoregressive language model behavior

**Out-of-scope uses:**
- General-purpose question answering (MarkGPT is not instruction-tuned)
- Any application requiring factual accuracy (this is a generative model)
- Translation between English and Banso (MarkGPT is not a translation model)

---

## Training Data

### Primary corpus: King James Bible
- Source: Public domain (copyright expired)
- Size: ~800,000 words, ~4.5MB plain text
- Content: All 66 books of the Old and New Testament
- Preprocessing: Verse markers removed; chapter/book boundaries preserved as special tokens

### Secondary corpus: Banso/Lamnso' vernacular
- Source: SIL International Bible translation excerpts (Lamnso' Bible portions), collected oral proverbs, community-contributed text
- Size: ~50,000–200,000 words (varies by community contribution)
- Content: Scripture portions, proverbs, narrative text, liturgical phrases in Lamnso'
- Preprocessing: Orthographic normalization applied (see `src/tokenizer/tokenizer.py`); tone diacritics optionally preserved

### Data statement
A full datasheet for this dataset is available at `data/banso-vernacular/datasheet.md`, following the format of Gebru et al. (2021). Dataset collectors made best-effort attempts to ensure text was collected with appropriate permissions and community consent.

---

## Architecture Details

| Component | Specification |
|---|---|
| Architecture | GPT (decoder-only Transformer) |
| Context window | 512 tokens |
| Embedding dimension | 256 |
| Number of layers | 6 |
| Attention heads | 8 |
| FFN inner dimension | 1024 (4× embedding) |
| Activation | GELU |
| Positional encoding | Learned embeddings |
| Vocabulary size | 8,000 (BPE, custom Banso-aware) |
| Tokenizer | Custom BPE (see `src/tokenizer/`) |
| Dropout | 0.1 (training) |

---

## Training Details

| Setting | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Peak learning rate | 3×10⁻⁴ |
| LR schedule | Linear warmup (2000 steps) + cosine decay |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Batch size | 32 |
| Training iterations | 50,000 |
| Mixed precision | bfloat16 |
| Hardware | [Record your hardware here] |
| Training time | [Record your training time here] |

---

## Evaluation

### Language Modeling (Perplexity)

| Split | Perplexity |
|---|---|
| KJV Bible validation set | [Fill in after training] |
| Banso validation set | [Fill in after training] |

Note: Perplexity on Banso text is expected to be higher than on English due to the smaller training corpus. This is expected and not a failure of the model.

### Qualitative Evaluation

10 human-evaluated samples rated by the curriculum author on:
- Biblical register appropriateness (1–5)
- Banso vocabulary integration (1–5)
- Fluency and coherence (1–5)

[Fill in evaluation results after training]

---

## Sample Outputs

The following outputs were generated with temperature=0.8, top_k=50:

**Prompt:** "In the beginning, Nfor made the heavens"
**Output:** [Fill in after training]

**Prompt:** "The shepherd said to his flock, shiy wir,"
**Output:** [Fill in after training]

**Prompt:** "Blessed are those who hunger and thirst,"
**Output:** [Fill in after training]

---

## Limitations

**Small scale.** At 10M parameters, MarkGPT is approximately 17,000× smaller than GPT-3 (175B). Its capabilities reflect this scale gap. It can produce Biblical-register prose and Banso-inflected text, but it will not match the coherence or factual consistency of large production models.

**Limited Banso data.** The Lamnso' language is significantly underrepresented compared to English. MarkGPT's Banso outputs should be reviewed by a native speaker for authenticity and accuracy. The model's Banso competence is best understood as a beginning — an invitation to the community to contribute more data.

**No factual grounding.** MarkGPT generates plausible-sounding text; it does not reason about factual accuracy. It should never be used in a context where users might mistake its outputs for authoritative information.

**Tonal information loss.** Due to inconsistencies in tonal marking across the Banso corpus, tone diacritics were stripped during preprocessing. The model does not learn Lamnso' tonal contrasts. This is a meaningful linguistic loss and a direction for future work.

---

## Ethical Considerations

**Cultural sensitivity.** The Banso people and the Lamnso' language are not merely a dataset. They represent a living culture, an oral tradition, and a community. This model was developed with the intention of honoring that culture, not extracting from it. Future versions should involve native speakers and community members as active collaborators, not just data sources.

**Religious content.** Training on the Bible means MarkGPT may generate text that sounds religious or spiritually authoritative. Users should be aware that the model is generating statistically plausible continuations, not theological truth.

**Misrepresentation risk.** Generated text that sounds like Banso vernacular may not be authentic Banso. This risk is mitigated by the evaluation rubric in Module 09 and the recommendation to have outputs reviewed by native speakers.

---

## How to Use MarkGPT

```python
import torch
from src.model.markgpt import markgpt_small
from src.tokenizer.tokenizer import MarkGPTTokenizer

# Load the tokenizer
tokenizer = MarkGPTTokenizer.from_pretrained("data/banso-vernacular/tokenizer/")

# Load the model
model = markgpt_small(vocab_size=tokenizer.vocab_size)
checkpoint = torch.load("checkpoints/markgpt_best.pt", map_location="cpu")
model.load_state_dict(checkpoint['model'])
model.eval()

# Generate text
prompt = "In the beginning, Nfor made the heavens"
input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )

generated_text = tokenizer.decode(output_ids[0].tolist())
print(generated_text)
```

---

## Citation

If you use MarkGPT in academic work or build on this curriculum, please cite:

```bibtex
@misc{markgpt2024,
  title        = {MarkGPT: A Language Model for Biblical and Banso Vernacular Text},
  author       = {[Your Name]},
  year         = {2024},
  howpublished = {MarkGPT 60-Day LLM Curriculum, GitHub},
  note         = {Trained as a capstone project under the MarkGPT curriculum}
}
```

Also cite the foundational work this model builds on:

```bibtex
@article{vaswani2017attention,
  title   = {Attention Is All You Need},
  author  = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2017}
}
```

---

*Model card follows the format recommended by Mitchell et al. (2019). Model Cards for Model Reporting.*  
*Dataset documentation follows Gebru et al. (2021). Datasheets for Datasets.*
