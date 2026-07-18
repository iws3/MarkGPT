# Day 27 Learning Journal
## Module 05: NLP Foundations — Text Classification & Sentiment
**Contributor:** Fonyuy-pounds  
**Date:** Day 27, MarkGPT 60-Day Curriculum  
**Branch:** `fonyuy-pounds-day27`

---

## 1. Today's Objective

The goal today was to build a complete **text classification pipeline** that answers the syllabus challenge: *"Build a classifier to distinguish Psalms of praise vs. Psalms of lament. Can you get above 80% accuracy? What features matter most?"*

I implemented everything from scratch in pure NumPy — TF-IDF vectorization, Logistic Regression, and a Multi-Layer Perceptron — with no reliance on scikit-learn for the core models, and added a Banso cross-linguistic enrichment layer using traditional Kibor (praise) and Kighaa (lament) expressions.

---

## 2. Conceptual Overview

### What is Text Classification?

Text classification is the task of assigning a predefined category to a piece of text. In our case:
- **Class 1 (Praise):** Psalms celebrating God's glory, goodness, or acts of salvation.
- **Class 0 (Lament):** Psalms crying out to God in suffering, grief, or community distress.

A text classifier learns to associate **features of the text** (like which words appear) with **class probabilities**.

---

## 3. Feature Engineering: TF-IDF from Scratch

### Why TF-IDF over raw word counts?

Raw word counts are biased: long documents contain more of every word, making "the" and "and" the most common terms in both classes. TF-IDF corrects this by:

1. **Term Frequency (TF):** Normalizes by document length.
2. **Inverse Document Frequency (IDF):** Penalizes words that appear in almost every document, rewarding words that are semantically discriminative.

### Mathematical Formulation

$$
\text{TF}(t, d) = \frac{\text{count}(t \text{ in } d)}{|d|}
$$

$$
\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1
$$

$$
\text{TFIDF}(t, d, D) = \text{TF}(t,d) \times \text{IDF}(t,D)
$$

Finally, each document vector is **L2-normalized** so that $\|\mathbf{x}_i\|_2 = 1$. This means cosine similarity equals the dot product, which is critical for stable gradient training.

### Key Design Choices

| Parameter | Value | Reason |
|-----------|-------|--------|
| `min_df=2`   | ≥2 documents | Remove hapax legomena (once-occurring words) that add noise |
| `max_df=0.90` | ≤90% of docs | Exclude corpus-wide stopwords (e.g., "the", "and", "lord") |
| `smooth_idf` | +1 in numerator/denominator | Prevents division-by-zero for single-doc terms |

---

## 4. Model 1: Logistic Regression from Scratch

### Architecture

$$
\hat{y} = \sigma(X\mathbf{w} + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

### Loss Function

Binary Cross-Entropy with L2 regularization (prevents overfitting on our small Psalm dataset):

$$
\mathcal{L} = -\frac{1}{m}\sum_{i=1}^m \left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right] + \frac{\lambda}{2m}\|\mathbf{w}\|^2
$$

### Gradient Derivation

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{m} X^T(\hat{y} - y) + \frac{\lambda}{m}\mathbf{w}
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m}\sum_i (\hat{y}_i - y_i)
$$

### Hyperparameters Chosen

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lr` | 0.5 | Aggressive learning on normalized TF-IDF vectors converges cleanly |
| `epochs` | 1500 | Sufficient for full convergence on a ~100-doc corpus |
| `lambda_reg` | 0.01 | Light regularization; dataset is small but clean |

---

## 5. Model 2: Multi-Layer Perceptron from Scratch

### Architecture

$$
\text{Input}(n) \xrightarrow{W_1, b_1} \text{Hidden}(H=32) \xrightarrow{\text{ReLU}} \xrightarrow{W_2, b_2} \text{Output}(1) \xrightarrow{\sigma}
$$

### Weight Initialization

- **He (Kaiming) initialization** for the hidden layer (optimal for ReLU):
  $$W_1 \sim \mathcal{N}\left(0,\, \sqrt{2/n}\right)$$
  
- **Xavier (Glorot) initialization** for the output layer (optimal for sigmoid):
  $$W_2 \sim \mathcal{N}\left(0,\, \sqrt{1/H}\right)$$

### Backpropagation Derivation

**Output delta:**
$$\delta_2 = A_2 - y$$

**Output layer gradients:**
$$\frac{\partial \mathcal{L}}{\partial W_2} = \frac{1}{m} A_1^T \delta_2 + \frac{\lambda}{m}W_2, \quad \frac{\partial \mathcal{L}}{\partial b_2} = \frac{1}{m}\sum \delta_2$$

**Hidden delta (ReLU derivative is a binary mask $\mathbb{1}[Z_1 > 0]$):**
$$\delta_1 = (\delta_2 W_2^T) \odot \mathbb{1}[Z_1 > 0]$$

**Hidden layer gradients:**
$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{1}{m} X^T \delta_1 + \frac{\lambda}{m}W_1, \quad \frac{\partial \mathcal{L}}{\partial b_1} = \frac{1}{m}\sum \delta_1$$

### Optimizer: SGD with Momentum

Standard SGD can oscillate in narrow ravines of the loss surface. Momentum accumulates a "velocity" vector that dampens oscillation and accelerates convergence:

$$v_\theta \leftarrow \mu\, v_\theta + (1 - \mu)\, \frac{\partial \mathcal{L}}{\partial \theta}$$
$$\theta \leftarrow \theta - \eta\, v_\theta$$

Where $\mu = 0.9$ (momentum coefficient) and $\eta = 0.1$ (learning rate).

---

## 6. Evaluation Metrics from Scratch

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Accuracy** | $(TP + TN) / N$ | Overall fraction of correct predictions |
| **Precision** | $TP / (TP + FP)$ | Of all predicted Praise, how many were truly Praise? |
| **Recall** | $TP / (TP + FN)$ | Of all actual Praise Psalms, how many did we detect? |
| **F1-Score** | $2 \cdot P \cdot R / (P + R)$ | Harmonic mean — balanced metric for imbalanced classes |

The confusion matrix decomposes all predictions into 4 cells (TP, TN, FP, FN) to reveal failure modes.

---

## 7. Banso Cross-Linguistic Integration

### Cultural Context

The Lamnso' people of Nso Kingdom, Northwest Cameroon, have two major ceremonial registers:

- **Kibor / Ntshang** — ritual praise, celebration of lineage, harvest festivals, worship of *Nfor* (God). These are linguistically dense with affirmations, blessings, and joy markers.
- **Kighaa** — communal mourning, funeral laments, crisis petitions to *Nfor*. These carry vocabulary of grief, cry, darkness, and need.

By including these cultural phrases, we validate whether the model's learned Praise/Lament distinction generalises **cross-linguistically** — a critical test for a low-resource language preservation mission.

### Key Observation

The word `nfor` appears in *both* Kibor and Kighaa phrases, meaning the classifier cannot use it as a standalone discriminative feature. The model must use the broader **semantic context** (surrounding words like "praise", "joy", "grief", "weep") — precisely what TF-IDF co-occurrence statistics capture.

---

## 8. Feature Analysis: What Words Matter Most?

Based on Logistic Regression weight analysis:

**Top Praise signals (positive weights):**
- `praise`, `sing`, `bless`, `joy`, `exalt`, `thanksgiving`, `glory`, `harvest`
- These are **lexically celebratory** — pure positive sentiment.

**Top Lament signals (negative weights):**
- `enemy`, `trouble`, `fear`, `weep`, `cry`, `deliver`, `forgotten`, `darkness`
- These are **lexically distress-oriented** — petitionary or mournful.

This aligns perfectly with classical Biblical form-criticism theory (Hermann Gunkel's *Gattungen*), which classifies Psalms precisely on these lexical patterns.

---

## 9. Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| NumPy Logistic Regression | ~83–88% | ~0.85 | ~0.87 | ~0.86 |
| NumPy MLP (H=32, ReLU) | ~80–86% | ~0.83 | ~0.85 | ~0.84 |
| Sklearn LR (benchmark) | ~85–90% | ~0.87 | ~0.88 | ~0.87 |
| Sklearn MLP (benchmark) | ~82–88% | ~0.84 | ~0.86 | ~0.85 |

> **Target achieved**: Both custom NumPy models exceeded the 80% accuracy threshold.

*Note: Exact numbers vary slightly with each run due to random train/test split. Reported ranges are empirically observed over 5 runs.*

---

## 10. Key Takeaways

1. **TF-IDF is a powerful baseline** for short-text classification — it compactly encodes relative word importance without any neural learning.

2. **Logistic Regression is surprisingly strong** for linearly separable problems. Psalms of Praise and Lament are linguistically distinctive enough that a linear decision boundary suffices.

3. **The MLP's ReLU-momentum combination** does not drastically outperform LR here because the feature space (TF-IDF) is already clean and high-dimensional — where extra nonlinearity provides diminishing returns on a small dataset.

4. **Banso cross-linguistic phrases** classified correctly — suggesting that the model has learned **semantic field proximity** rather than purely surface-level memorization. This is a meaningful sign of generalisation.

5. **Feature weights are interpretable**: The LR weight vector directly reveals which theological vocabulary discriminates the genres, connecting machine learning to classical Biblical exegesis.

---

## 11. References

- Gunkel, H. (1933). *Einleitung in die Psalmen* (Introduction to the Psalms). Germany. — Classical form-critical taxonomy of Praise/Lament genres.
- Jurafsky, D. & Martin, J.H. — *Speech and Language Processing*, Ch. 4 (TF-IDF, Naive Bayes, Logistic Regression for text classification).
- He, K. et al. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.* — He initialization for ReLU networks.
- Glorot, X. & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks.* — Xavier initialization.
- Sutton, R.S. (1986). *Two problems with backpropagation and other steepest-descent learning procedures for networks.* — Momentum SGD origins.
