# [Day 32] Scaled Dot-Product Attention from Scratch (Fonyuy-pounds)

Implements the Day 32 Module 06 exercise: Scaled Dot-Product Attention built entirely from
first principles in PyTorch. The implementation verifies mathematical alignment with PyTorch's
native `F.scaled_dot_product_attention` on both forward outputs and backpropagated gradients.
Includes Banso (Lamnso') linguistic context-disambiguation application on the polysemous term *Nfor*.

Key contributions:
- `ScaledDotProductAttention`: `Q @ K^T / sqrt(d_k)` → softmax → `@V` (pure PyTorch, no `nn.MultiheadAttention`)
- `get_causal_mask()`: lower-triangular mask preventing future-token attention in autoregressive generation
- `run_verification_tests()`: batch/multi-head forward pass + gradient alignment checks against `F.scaled_dot_product_attention`
- `run_banso_linguistic_application()`: polysemy resolution for *Nfor* (God vs. King) in praise/lament contexts
- `day32_journal.md`: full mathematical derivation of the scaling factor, variance analysis, numerical walkthrough, and linguistic insight

All verification tests pass to `atol=1e-5` (forward) and `atol=1e-4` (gradients).

Contributor: Fonyuy-pounds

## 📝 Description

This PR implements the Day 32 exercise on Scaled Dot-Product Attention — the foundational
computation underlying every modern Transformer and LLM. The implementation covers all
components required by the syllabus:

**Attention formula implemented:**  
`Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k) + mask) @ V`

Key implementation choices:
- Uses `torch.matmul` with `transpose(-2, -1)` for batched multi-head shape support
- Uses `float('-inf')` mask convention (compatible with both `F.softmax` and `F.scaled_dot_product_attention`)
- Gradient verification compares `∇Q`, `∇K`, and `∇V` against PyTorch's native implementation
- Banso application constructs conceptual embeddings in a 4-dimensional semantic space (Divine / Royal / Praise / Lament)

## 🎯 Type of Change

- [x] 🎓 New lesson or exercise content
- [x] ✨ New feature
- [ ] 🐛 Bug fix (non-breaking)
- [ ] 📚 Documentation improvement
- [ ] ♻️ Code refactor
- [ ] 🧪 Test addition
- [ ] 🔧 Other: ____________

## 📖 Related Module(s)

- Module 06

## 🧪 Testing

- [x] All Python scripts run without errors in Google Colab environment
- [x] Code examples work correctly and pass all verification assertions
- [x] Tested with Python 3.10+, PyTorch 2.0+
- [x] Verified on Windows and Colab (Linux)

## ✅ Checklist

- [x] Followed style guidelines ([BEST_PRACTICES.md](../BEST_PRACTICES.md))
- [x] Added comments/docstrings where needed
- [x] No hardcoded file paths (using relative and dynamically resolved paths)
- [x] Commit messages follow conventional format
- [x] No large files (>10MB) committed
- [x] For lessons: Clear learning objectives
- [x] For exercises: Includes solutions
- [x] Updated relevant README if adding new content

## 📸 Screenshots (if applicable)

No image exports in this exercise. Attention weights are printed as ASCII tables in console output showing the disambiguation of *Nfor* across praise and lament contexts.

## 📌 Additional Notes

- The Banso linguistic application demonstrates that scaled dot-product attention, even with
  identity projection matrices and no training, successfully resolves the polysemy of the
  Lamnso' word *Nfor* (God/King) based solely on the surrounding theological context words *kibor* (praise) vs *kighaa* (lament).
- This exercise forms the foundation for Day 33 (Multi-Head Attention + Positional Encoding)
  and eventually the full MarkGPT-Nano Transformer block (Day 36).

---

**Thanks for contributing to MarkGPT!** 🚀  
See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.
