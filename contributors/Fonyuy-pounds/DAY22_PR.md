# PR: Day 22 - Seq2Seq & Encoder-Decoder in PyTorch (Fonyuy-pounds)

## 📝 Description

This PR implements **Day 22** of the MarkGPT curriculum. It covers the transition to PyTorch and the implementation of a full Sequence-to-Sequence (Seq2Seq) model using an Encoder-Decoder architecture.

**Key changes:**
- Added `day22_exercises.py` featuring a PyTorch implementation of a Seq2Seq model using GRU cells.
- Implemented **Teacher Forcing** during the training phase to accelerate convergence.
- Demonstrated the architecture using a character-level sequence reversal task.
- Authored `day22_journal.md` reflecting on the "context vector bottleneck" and the immense relief of switching from manual NumPy calculus to PyTorch's `autograd`.
- Updated the contributor `README.md` to track progress through Module 04.

## 🎯 Type of Change

- [x] 🎓 New lesson or exercise content
- [ ] 🐛 Bug fix (non-breaking)
- [ ] 📚 Documentation improvement
- [x] ✨ New feature (PyTorch Transition)
- [ ] ♻️ Code refactor
- [ ] 🧪 Test addition

## 📖 Related Module(s)

- **Module 04**: Sequence Modeling (Seq2Seq architectures)

## 🧪 Testing

- [x] Tested model on randomly generated string sequences.
- [x] Verified that Teacher Forcing correctly feeds ground truth targets during training.
- [x] Verified that evaluation inference successfully generates reversed sequences from the context vector.
- [x] Tested with Python 3.10+ and PyTorch `2.0+`.

## ✅ Checklist

- [x] Followed style guidelines ([BEST_PRACTICES.md](../../BEST_PRACTICES.md))
- [x] Added comments explaining model architecture and Teacher Forcing logic.
- [x] No hardcoded file paths.
- [x] Commit messages follow conventional format.
- [x] Updated relevant README.

## 📌 Additional Notes

This day marks a significant milestone in the curriculum. Implementing backpropagation through time across an Encoder and Decoder manually in NumPy would have been overly complex, so making the leap to PyTorch here was highly beneficial. The sequence reversal task perfectly highlighted the structural limits of using a single context vector, teeing up Day 23's topic perfectly: **Attention Mechanisms**.

---

**Thanks for contributing to MarkGPT!** 🚀
