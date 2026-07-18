# PR: Day 23 - Attention Mechanisms (Fonyuy-pounds)

## 📝 Description

This PR implements **Day 23** of the MarkGPT curriculum. It covers the crucial addition of **Attention Mechanisms** to Sequence-to-Sequence models, effectively solving the "context vector bottleneck".

**Key changes:**
- Added `day23_exercises.py` featuring a PyTorch implementation of Dot-Product (Soft) Attention.
- Upgraded the Decoder from Day 22 to an `AttentionDecoder` that dynamically calculates attention weights for every encoder output at each decoding time step.
- Demonstrated that the model converges faster and maintains accuracy on longer sequences during the Sequence Reversal task compared to the vanilla Seq2Seq model.
- Authored `day23_journal.md` reflecting on the mathematical elegance of `torch.bmm` (batch matrix multiplication) and the intuition behind Queries, Keys, and Values.
- Updated the contributor `README.md` to track progress.

## 🎯 Type of Change

- [x] 🎓 New lesson or exercise content
- [ ] 🐛 Bug fix (non-breaking)
- [ ] 📚 Documentation improvement
- [x] ✨ New feature (Attention Module)
- [ ] ♻️ Code refactor
- [ ] 🧪 Test addition

## 📖 Related Module(s)

- **Module 04**: Sequence Modeling (Attention Mechanisms)

## 🧪 Testing

- [x] Tested model on randomly generated string sequences with padding.
- [x] Extracted `attention_weights` to ensure the Decoder correctly aligns its focus across the input sequence length.
- [x] Verified faster convergence rates over Day 22's model.
- [x] Tested with Python 3.10+ and PyTorch `2.0+`.

## ✅ Checklist

- [x] Followed style guidelines ([BEST_PRACTICES.md](../../BEST_PRACTICES.md))
- [x] Added comments explaining Query, Key, and Value interactions.
- [x] No hardcoded file paths.
- [x] Commit messages follow conventional format.
- [x] Updated relevant README.

## 📌 Additional Notes

Adding attention represents the final conceptual leap needed before moving to fully parallelized self-attention architectures like the Transformer. By allowing the decoder to "look back" at specific parts of the source text, we've successfully shattered the information bottleneck that vanilla encoder-decoders suffer from.

---

**Thanks for contributing to MarkGPT!** 🚀
