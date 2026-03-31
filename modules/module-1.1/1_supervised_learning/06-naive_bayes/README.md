# 🎲 Lesson 06 — Naive Bayes

> **Core Idea**: Use Bayes' theorem to calculate the probability that an input belongs to each class, making the "naive" assumption that all features are independent of each other. Despite this oversimplification, it works surprisingly well in practice — especially for text.

---

## 📋 Table of Contents

1. [Bayes' Theorem — The Foundation](#1-bayes-theorem)
2. [The Naive Assumption](#2-the-naive-assumption)
3. [Three Flavours of Naive Bayes](#3-three-flavours)
4. [Laplace Smoothing](#4-laplace-smoothing)
5. [Python Implementation](#5-python-implementation)
6. [Visual Summary](#6-visual-summary)
7. [When to Use](#7-when-to-use)

---

## 1. Bayes' Theorem — The Foundation

Naive Bayes is built entirely on one formula: Bayes' theorem. Let's decode it:

```
P(class | features) = P(features | class) × P(class)
                      ─────────────────────────────
                              P(features)

In words:
  P(class | features) = "Given I see these features, what's the probability of this class?"
                        This is what we want to compute. Called the POSTERIOR.

  P(features | class) = "If I were in this class, how likely are these features?"
                        The LIKELIHOOD. We estimate this from training data.

  P(class) = "How common is this class in the training data?"
              The PRIOR. Easy to compute: count(class) / count(total examples).

  P(features) = A constant (the same for all classes). We ignore it because we
                only need to compare P(class|features) across classes, not compute it exactly.
```

**Example: Is this email spam?**

```
Training data shows:
  40% of emails are spam (P(spam) = 0.4)
  60% are not spam (P(not spam) = 0.6)

The word "FREE" appears in:
  80% of spam emails → P("FREE" | spam) = 0.8
  10% of normal emails → P("FREE" | not spam) = 0.1

New email contains "FREE". Which class is more likely?

  P(spam | "FREE") ∝ P("FREE" | spam) × P(spam)     = 0.8 × 0.4 = 0.32
  P(not spam | "FREE") ∝ P("FREE" | not spam) × P(not spam) = 0.1 × 0.6 = 0.06

  Normalise: 0.32 / (0.32 + 0.06) = 84% probability this email is spam.
  → Classify as spam.
```

---

## 2. The Naive Assumption

In reality, features are not independent. The word "FREE" and the phrase "WIN NOW" often appear together in spam emails — they're correlated. The "naive" assumption pretends they're independent:

```
P(features | class) = P(feature₁ | class) × P(feature₂ | class) × ... × P(featureₙ | class)

"Naive" because assuming independence is almost certainly wrong.
But in practice, this simplification is harmless for *ranking* classes
because we only care which class scores highest, not the exact probabilities.
```

This assumption turns an exponentially complex joint probability into a simple product of individual probabilities — making it extremely fast to compute even with thousands of features.

---

## 3. Three Flavours of Naive Bayes

```
┌─────────────────────┬─────────────────────────────┬───────────────────────┐
│ Variant             │ Feature Type                 │ When to Use           │
├─────────────────────┼─────────────────────────────┼───────────────────────┤
│ GaussianNB          │ Continuous real-valued       │ Numeric features,     │
│                     │ Assumes P(xᵢ|class) follows  │ where Gaussian        │
│                     │ a Gaussian (bell curve)       │ assumption is OK      │
├─────────────────────┼─────────────────────────────┼───────────────────────┤
│ MultinomialNB       │ Integer counts (e.g., word   │ Text classification   │
│                     │ frequency in a document)      │ with bag-of-words     │
├─────────────────────┼─────────────────────────────┼───────────────────────┤
│ BernoulliNB         │ Binary features (0 or 1)     │ Text with binary      │
│                     │ "Does this word appear?"      │ word presence/absence │
└─────────────────────┴─────────────────────────────┴───────────────────────┘
```

---

## 4. Laplace Smoothing

What if a feature never appears with a certain class in training? Then P(feature | class) = 0, which makes the entire product 0 — the model becomes completely certain about one class, which is wrong.

Laplace (additive) smoothing adds a small count (alpha) to every possibility, preventing zero probabilities:

```
P(xᵢ | class) = (count(xᵢ, class) + α) / (count(class) + α × |vocabulary|)

With α = 1 (default "Laplace smoothing"):
  Even words never seen in spam get a small non-zero probability.
  This prevents the model from confidently making decisions based on absence.
```

---

## 5. Python Implementation

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import numpy as np

# ─── GaussianNB for numeric features (Iris dataset) ─────────────────────
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.4f}")
# After training, model stores:
# gnb.theta_   → mean of each feature for each class
# gnb.sigma_   → variance of each feature for each class
print("Class means (theta):", gnb.theta_)

# ─── MultinomialNB for text classification ───────────────────────────────
categories = ['sci.space', 'talk.politics.guns', 'rec.autos', 'comp.graphics']
train_data = fetch_20newsgroups(subset='train', categories=categories)
test_data  = fetch_20newsgroups(subset='test',  categories=categories)

# Build pipeline: raw text → word counts → Naive Bayes
text_pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english', max_features=10000)),
    ('clf',  MultinomialNB(alpha=1.0))   # alpha is Laplace smoothing
])
text_pipeline.fit(train_data.data, train_data.target)
y_pred = text_pipeline.predict(test_data.data)
print(f"\nText classification accuracy: {text_pipeline.score(test_data.data, test_data.target):.4f}")
print(classification_report(test_data.target, y_pred, target_names=categories))

# ─── What words most predict each category? ──────────────────────────────
vectorizer = text_pipeline.named_steps['vect']
clf        = text_pipeline.named_steps['clf']
feature_names = vectorizer.get_feature_names_out()

for class_idx, class_name in enumerate(categories):
    top_words = np.argsort(clf.feature_log_prob_[class_idx])[-10:]
    words = [feature_names[i] for i in top_words]
    print(f"\nTop words for '{class_name}': {words}")
```

---

## 6. Visual Summary

```
╔══════════════════════════════════════════════════════════════════╗
║                   NAIVE BAYES — OVERVIEW                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Training:                                                       ║
║    For each class: compute prior P(class) = count/total         ║
║    For each feature: compute P(feature | class) from counts     ║
║    → Very fast! Just counting and dividing.                     ║
║                                                                  ║
║  Prediction:                                                     ║
║    Score(class) = P(class) × Π P(featureᵢ | class)             ║
║    Predict the class with the highest score                      ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  STRENGTHS: Extremely fast, works great for text, handles         ║
║             high-dimensional data, robust to irrelevant features ║
║  WEAKNESS: Independence assumption often violated in practice    ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 7. When to Use

Naive Bayes is ideal for text classification tasks (spam detection, sentiment analysis, topic classification), real-time prediction where speed is critical, and situations where you have a very small training dataset but many features. It serves as an excellent baseline for any classification problem. Its main limitation is the independence assumption — when features are highly correlated (e.g., medical symptoms that tend to occur together), more sophisticated models like logistic regression or random forests will outperform it.

> 📂 Next: [exercises.md](exercises.md)
