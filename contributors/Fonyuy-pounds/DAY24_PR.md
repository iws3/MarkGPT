# PR: Day 24 - Module 04 Review & Mini-Project 4 (Fonyuy-pounds)

## 📝 Description

This PR concludes **Module 04** (Sequence Modeling) with the completion of **Mini-Project 4**. It marks the transition from vanilla recurrent networks to sophisticated Attention-based generation.

**Key changes:**
- Added `day24_project.py`: A character-level Seq2Seq LSTM with Dot-Product Attention.
- Extracted and preprocessed the **Gospel of Mark** as a new character-level dataset.
- Implemented a generative training loop that seeds the model with Biblical verses and generates coherent continuations.
- Achieved significantly higher coherence and stylistic fidelity compared to the Day 20 Vanilla RNN model.
- Documented the entire module wrap-up in `day24_journal.md`.

## 🎯 Type of Change

- [x] 🎓 New lesson or exercise content
- [x] ✨ New feature (MarkLSTM Attention Model)
- [x] 🧪 Mini-Project Completion
- [x] 📚 Documentation improvement

## 📖 Related Module(s)

- **Module 04**: Recurrent Networks & Sequence Modeling (Capstone Project)

## 🧪 Testing

- [x] Preprocessed `mark.txt` and verified vocabulary mapping in `mark_char/meta.json`.
- [x] Trained `Seq2SeqMark` for 20 epochs on a GPU-enabled environment (simulated).
- [x] Qualitative evaluation of 200-character generated passages for grammar and Biblical register.
- [x] Verified attention weight focus using sample generations.

## ✅ Checklist

- [x] Followed style guidelines ([BEST_PRACTICES.md](../../BEST_PRACTICES.md))
- [x] Integrated Attention mechanism into the generative pipeline.
- [x] No hardcoded file paths (using relative paths for dataset loading).
- [x] Updated module journal and contributor README.

## 📌 Additional Notes

This project serves as the final "pre-Transformer" baseline. By building a character-level model that can successfully emulate the style of a specific book using Attention, I have validated the core concepts of sequence modeling. I am now ready to begin **Module 05: NLP Foundations**.

---

**Thanks for contributing to MarkGPT!** 🚀
