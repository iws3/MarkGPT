# PR: Day 24 - Module 04 Capstone: MarkLSTM with Attention (Fonyuy-pounds)

## 📝 Description

This PR concludes **Module 04: Sequence Modeling** by implementing **Mini-Project 4**. The project involves building a generative, character-level LSTM model enhanced with **Dot-Product Attention**, trained specifically on the **Gospel of Mark**.

**Key accomplishments:**
- Extracted and preprocessed the Gospel of Mark into a character-level binary dataset.
- Developed a Seq2Seq architecture in PyTorch where the Decoder uses Attention to "look back" at the seed verse during text generation.
- Achieved a significant leap in stylistic fidelity and coherence over the previous vanilla RNN and LSTM models.
- Provided a full training script with generative sampling.

## 🎯 Type of Change

- [x] 🎓 New lesson or exercise content
- [ ] 🐛 Bug fix (non-breaking)
- [ ] 📚 Documentation improvement
- [x] ✨ New feature (Attention-based Generative Model)
- [ ] ♻️ Code refactor
- [x] 🧪 Test addition

## 📖 Related Module(s)

- **Module 04**: Sequence Modeling (Day 24 Review & Mini-Project 4)

## 🧪 Testing

- [x] All Jupyter notebook cells/scripts run without errors.
- [x] Code examples work correctly (successfully generated Biblical prose continuations).
- [x] Tested with Python 3.10+ and PyTorch 2.0+.
- [x] Verified on Windows.

## ✅ Checklist

- [x] Followed style guidelines ([BEST_PRACTICES.md](../../BEST_PRACTICES.md))
- [x] Added comments/docstrings where needed (explained Attention logic and Seq2Seq flow).
- [x] No hardcoded file paths (using relative paths for dataset loading).
- [x] Commit messages follow conventional format.
- [x] No large files (>10MB) committed (data files are ignored via `.gitignore`).
- [x] For lessons: Clear learning objectives.
- [x] For exercises: Includes solutions (full model implementation provided).
- [x] Updated relevant README if adding new content.
- [ ] Referenced issues with #issue_number (N/A)

## 📸 Screenshots (if applicable)

Sample Output:
```text
Seed: "The beginning of the gospel of Jesus Christ, the Son of God; 1:2 As it is written in the prophets,"
Continuation: "Behold, I send my messenger before thy face, which shall prepare thy way before thee. The voice of one crying in the wilderness..."
```

## 📌 Additional Notes

This project demonstrates the practical application of attention mechanisms to solve memory bottlenecks in RNNs. It serves as the definitive baseline before we transition to parallelized attention in the Transformer module.

---

**Thanks for contributing to MarkGPT!** 🚀
See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.
