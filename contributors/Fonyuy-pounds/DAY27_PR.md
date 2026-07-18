# [Day 27] Text Classification & Sentiment — Psalms of Praise vs. Lament (Fonyuy-pounds)

Implements the Day 27 NLP Foundations exercise: full from-scratch text classification pipeline to distinguish Biblical Psalms of Praise from Psalms of Lament, with Banso cross-linguistic enrichment.

Key contributions:
- **NumpyTfidfVectorizer**: smoothed TF-IDF with L2 row normalization (pure NumPy)
- **NumpyLogisticRegression**: binary cross-entropy + L2 reg + batch gradient descent
- **NumpyMLP**: Input->Dense(32,ReLU)->Dense(1,Sigmoid), He/Xavier init, analytical backprop, SGD with momentum (mu=0.9)
- **compute_metrics()**: Accuracy, Precision, Recall, F1-Score, confusion matrix
- **Banso cultural enrichment**: 12 Kibor (praise) + 12 Kighaa (lament) phrases
- **Sklearn benchmarking**: (optional, guarded by try/except)
- **Dynamic workspace root detection**: (local + Google Colab compatible)
- **Visualizations**: loss curves, metrics comparison, feature importance charts

Target accuracy (>=80%) achieved by both custom NumPy models. Feature analysis confirms classical form-criticism categories (Gunkel 1933).

Contributor: Fonyuy-pounds

## 📝 Description

This PR introduces a from-scratch text classification system designed to classify Biblical Psalms of Praise vs. Lament, along with a validation test on low-resource Lamnso' (Banso) theological statements:
- **Day 27 Exercises Python script** containing complete vectorizer, logistic regression, MLP, metrics computations, Banso datasets, and visualization export logic.
- **Day 27 Learning Journal** with complete mathematical derivations of cross-entropy gradients, backpropagation equations, and Banso socio-cultural integration details.
- **Three custom visualizations**:
  - `classification_loss_curves.png`
  - `classification_metrics_comparison.png`
  - `feature_importance.png`
- Updated contributor `README.md` to reflect completed Day 27 milestones.

## 🎯 Type of Change

- [x] 🎓 New lesson or exercise content
- [x] ✨ New feature
- [ ] 🐛 Bug fix (non-breaking)
- [ ] 📚 Documentation improvement
- [ ] ♻️ Code refactor
- [ ] 🧪 Test addition
- [ ] 🔧 Other: ____________

## 📖 Related Module(s)

- Module 05: NLP Foundations (Day 27)

## 🧪 Testing

- [x] All python scripts run without errors locally and in Google Colab environment
- [x] Code examples work correctly and achieve >=80% accuracy targets
- [x] Tested with Python 3.10+
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
- [x] Referenced issues with #issue_number

## 📸 Screenshots (if applicable)

Visualizations generated and exported successfully:
1. **Training Loss Curves** (`classification_loss_curves.png`): Demonstrates smooth optimization progress and weight decay convergence.
2. **Metrics Comparison** (`classification_metrics_comparison.png`): Visualizes performance benchmarks across Numpy Logistic Regression, Numpy MLP, and optional scikit-learn baseline models.
3. **Feature Importance Attributions** (`feature_importance.png`): Identifies high-weight linguistic markers aligning with form-criticism categories.

## 📌 Additional Notes

- Culturally validated using **Kibor** (Praise) and **Kighaa** (Lament) sentences to verify contextual disambiguation of the shared term *nfor* (God/King) in the Banso linguistic space.
- Fully compatible with Google Colab environment via dynamic directory path resolving.

---

**Thanks for contributing to MarkGPT!** 🚀
See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.
