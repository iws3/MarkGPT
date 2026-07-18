# Model Card: MarkGPT-Bible

## Model Details

- **Name**: MarkGPT-Bible-Nano
- **Architecture**: Decoder-only Transformer
- **Parameters**: 2M  
- **Training Data**: King James Bible (780K tokens)
- **Training Time**: 10 min (NVIDIA RTX 3060)
- **License**: MIT  

## Intended Use

Educational purpose: Learning how LLMs work from scratch.

**Not intended for:**
- Production text generation
- Financial/medical decisions
- Replacing professional services

## Performance

| Metric | Value |
|--------|-------|
| Validation Loss | 4.2 |
| Perplexity | 67 |
| BLEU@1 | 0.34 |

## Limitations

- Small training data (780K tokens)
- Trained on Bible only (domain-specific)
- May hallucinate or repeat text
- Temperature/sampling needed for diversity

## Datasets

- **Primary**: King James Bible (Public Domain)
- **Size**: 31K verses, ~780K tokens
- **Preprocessing**: BPE tokenization (vocab 4K)

## Training Procedure

```yaml
Model: MarkGPT-Nano
Optimizer: AdamW (lr=0.001)
Batch Size: 32
Epochs: 3
Hardware: GPU (NVIDIA RTX 3060)
```

## Ethical Considerations

- Trained on Biblical text with specific theological perspective
- May encode gender biases present in translation
- Not trained on Banso language (future work)

## References

- Vaswani et al. (2017). "Attention is All You Need"
- Radford et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)

