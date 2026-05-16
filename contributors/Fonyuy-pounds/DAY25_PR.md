# PR: Day 25 - Tokenization Deep Dive (Fonyuy-pounds)

## 📝 Description

This PR implements **Day 25** of the MarkGPT curriculum. It focuses on the transition from character-level modeling to **Subword Tokenization** using the **Byte-Pair Encoding (BPE)** algorithm.

**Key changes:**
- Implemented `day25_exercises.py`: A from-scratch implementation of BPE including pair counting, merging, and encoding.
- Trained a BPE tokenizer on a 50KB sample of the KJV Bible.
- Analyzed the "Fertility" metric to demonstrate how vocabulary size impacts sequence length.
- Documented findings on BPE's effectiveness for handling low-resource and complex languages like Banso in `day25_journal.md`.

## 🎯 Type of Change

- [x] 🎓 New lesson or exercise content
- [ ] 🐛 Bug fix (non-breaking)
- [ ] 📚 Documentation improvement
- [x] ✨ New feature (BPE Implementation)
- [ ] ♻️ Code refactor
- [ ] 🧪 Test addition

## 📖 Related Module(s)

- **Module 05**: NLP Foundations (Tokenization)

## 🧪 Testing

- [x] All scripts run without errors.
- [x] Verified BPE merges (`t+h -> th`, `e+</w> -> e</w>`) on KJV Bible text.
- [x] Calculated fertility across different merge counts.
- [x] Tested with Python 3.10+.

## ✅ Checklist

- [x] Followed style guidelines ([BEST_PRACTICES.md](../../BEST_PRACTICES.md))
- [x] Added comments/docstrings explaining the BPE merge logic.
- [x] No hardcoded file paths.
- [x] Commit messages follow conventional format.
- [x] Updated relevant README.

## 📌 Additional Notes

This work lays the groundwork for Day 26, where we will begin converting these subword tokens into continuous vector representations (Word Embeddings).

---

**Thanks for contributing to MarkGPT!** 🚀
