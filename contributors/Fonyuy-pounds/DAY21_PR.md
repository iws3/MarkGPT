# PR: Day 21 - LSTM & GRU Implementation (Fonyuy-pounds)

## 📝 Description

This contribution completes **Day 21** of the curriculum. I have implemented deep sequence models (LSTM and GRU) from scratch using NumPy to showcase how gating mechanisms solve the vanishing gradient problem encountered in vanilla RNNs.

Key changes:
- Added `day21_exercises.py` with full `CharLSTM` and `CharGRU` implementations.
- Implemented Backpropagation Through Time (BPTT) for gated architectures.
- Added `day21_journal.md` with detailed reflections and comparison between RNN and LSTM performance.
- Updated contributor README to track Module 04 progress.

## 🎯 Type of Change

- [x] 🎓 New lesson or exercise content
- [ ] 🐛 Bug fix (non-breaking)
- [ ] 📚 Documentation improvement
- [ ] ✨ New feature
- [x] ♻️ Code refactor
- [ ] 🧪 Test addition
- [ ] 🔧 Other: ____________

## 📖 Related Module(s)

- Module 04: Sequence Modeling

## 🧪 Testing

- [x] All Python scripts run without errors.
- [x] LSTM/GRU training loops converge on the John character dataset.
- [x] Tested with Python 3.10+.
- [x] Verified numerical stability of gate gradients.

## ✅ Checklist

- [x] Followed style guidelines ([BEST_PRACTICES.md](../../BEST_PRACTICES.md))
- [x] Added comments/docstrings where needed
- [x] No hardcoded file paths (using relative paths for dataset loading)
- [x] Commit messages follow conventional format
- [x] No large files (>10MB) committed
- [ ] For lessons: Clear learning objectives
- [x] For exercises: Includes solutions
- [x] Updated relevant README if adding new content
- [ ] Referenced issues with #issue_number

## 📸 Screenshots (if applicable)

*No visualizations added for this PR, but sampling output is documented in the journal.*

## 📌 Additional Notes

The backward pass for the LSTM implementation was particularly complex due to the gradient splitting across four gates. The current implementation uses AdaGrad for parameter updates, matching the Day 20 RNN setup for fair comparison.

---

**Thanks for contributing to MarkGPT!** 🚀
See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.
