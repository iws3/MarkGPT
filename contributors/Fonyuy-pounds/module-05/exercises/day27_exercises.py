"""
Day 27 Exercise: Text Classification & Sentiment (Psalms of Praise vs. Lament)
Module 05: NLP Foundations
=============================================================================

Contributor: Fonyuy-pounds
Purpose   : Build a classifier to distinguish Biblical Psalms of Praise from Psalms of Lament,
            integrating Banso cultural expressions of Kibor (praise) and Kighaa (lament).

How to Run in Google Colab:
    1. Mount Google Drive or upload the MarkGPT repo:
         from google.colab import drive
         drive.mount('/content/drive')
         WORKSPACE_ROOT = "/content/drive/MyDrive/MarkGPT"    # adjust to your path
    2. Run: exec(open(f"{WORKSPACE_ROOT}/contributors/Fonyuy-pounds/module-05/exercises/day27_exercises.py").read())

Objectives:
    - Parse KJV Bible to extract all 150 Psalms as labeled text documents.
    - Implement TF-IDF vectorization from scratch in pure NumPy.
    - Build custom Logistic Regression and MLP classifiers with analytical backpropagation.
    - Evaluate via Accuracy, Precision, Recall, F1-score computed from scratch.
    - Cross-linguistically evaluate Banso Kibor/Kighaa phrases.
    - Benchmark against Scikit-learn baselines.
    - Export training curves and metric comparison charts as PNG files.
"""

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab/server environments
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# PART 0: Workspace Root Detection (local + Colab compatible)
# ─────────────────────────────────────────────────────────────────────────────

def find_workspace_root():
    """
    Detect the MarkGPT workspace root dynamically.
    Searches from the script's own location upward, then tries common Colab paths.
    Falls back to the current working directory if nothing is found.
    """
    # Start from this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    candidate = script_dir
    for _ in range(10):
        if os.path.exists(os.path.join(candidate, "data", "raw", "kjv_bible.txt")):
            return candidate
        candidate = os.path.dirname(candidate)

    # Common Colab Google Drive paths
    colab_paths = [
        "/content/drive/MyDrive/MarkGPT",
        "/content/MarkGPT",
        "/content/drive/MyDrive/ML/AI/MarkGPT",
    ]
    for path in colab_paths:
        if os.path.exists(os.path.join(path, "data", "raw", "kjv_bible.txt")):
            return path

    # Last resort: current directory
    return os.getcwd()


WORKSPACE_ROOT = find_workspace_root()
EXERCISES_DIR = os.path.join(WORKSPACE_ROOT, "contributors", "Fonyuy-pounds", "module-05", "exercises")
os.makedirs(EXERCISES_DIR, exist_ok=True)

print("=" * 75)
print("         DAY 27: TEXT CLASSIFICATION & SENTIMENT PIPELINE")
print("=" * 75)
print(f"Workspace root : {WORKSPACE_ROOT}")
print(f"Outputs will be saved to: {EXERCISES_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Data Parsing & Cultural Enrichment
# ─────────────────────────────────────────────────────────────────────────────

def load_psalms(workspace_root):
    """
    Parse the Book of Psalms from the KJV Bible text file.

    The Book of Psalms occupies lines 45,998 – 53,300 (1-indexed) in the
    Gutenberg KJV edition bundled with this repo. Each Psalm is assembled by
    concatenating all of its verses into a single text document.

    Returns:
        psalm_docs (dict): {psalm_number (int): full_text (str)}
    """
    bible_path = os.path.join(workspace_root, "data", "raw", "kjv_bible.txt")
    if not os.path.exists(bible_path):
        raise FileNotFoundError(
            f"KJV Bible not found at '{bible_path}'.\n"
            "Please ensure WORKSPACE_ROOT points to the correct MarkGPT directory."
        )

    with open(bible_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    psalm_docs = {i: [] for i in range(1, 151)}
    psalm_pattern = re.compile(r"^(\d+):(\d+)\s+(.*)$")
    current_psalm = None

    # Lines 45997–53299 in 0-indexed Python slice
    for line_raw in lines[45997:53300]:
        line = line_raw.strip()
        if not line:
            continue
        m = psalm_pattern.match(line)
        if m:
            current_psalm = int(m.group(1))
            verse_text = m.group(3)
            if 1 <= current_psalm <= 150:
                psalm_docs[current_psalm].append(verse_text)
        else:
            if current_psalm and 1 <= current_psalm <= 150 and psalm_docs[current_psalm]:
                psalm_docs[current_psalm][-1] += " " + line

    # Collapse each Psalm to a single string
    result = {}
    for num, verses in psalm_docs.items():
        if verses:
            result[num] = " ".join(verses)

    print(f"\n[1/6] Parsed {len(result)} Psalms from KJV Bible text.")
    return result


def get_labeled_psalms(psalm_docs):
    """
    Assign genre labels according to classical Biblical form-criticism categories.

    Labels:
        1 → Psalm of Praise (Hymn / Thanksgiving)
        0 → Psalm of Lament (Individual / Communal cry)

    Mixed or Wisdom Psalms are excluded for a clean binary classification task.

    Returns:
        documents (list[str]), labels (list[int]), metadata (list[str])
    """
    # Classical Praise Psalms
    PRAISE = {
        8, 19, 23, 29, 30, 33, 34, 47, 65, 66, 67, 93, 95, 96, 97, 98, 99,
        100, 103, 104, 111, 113, 114, 115, 117, 118, 134, 135, 136, 138,
        145, 146, 147, 148, 149, 150
    }

    # Classical Lament Psalms
    LAMENT = {
        3, 4, 5, 6, 7, 10, 12, 13, 14, 17, 22, 25, 26, 27, 28, 31, 35, 38,
        39, 41, 42, 43, 44, 51, 53, 54, 55, 56, 57, 59, 60, 61, 64, 69, 70,
        71, 74, 77, 79, 80, 83, 85, 86, 88, 90, 102, 109, 120, 123, 130,
        137, 140, 141, 142, 143
    }

    documents, labels, metadata = [], [], []
    for num, doc in psalm_docs.items():
        if num in PRAISE:
            documents.append(doc)
            labels.append(1)
            metadata.append(f"Psalm {num} (Praise)")
        elif num in LAMENT:
            documents.append(doc)
            labels.append(0)
            metadata.append(f"Psalm {num} (Lament)")

    n_praise = sum(labels)
    n_lament = len(labels) - n_praise
    print(f"[2/6] Labeled {len(labels)} Psalms — Praise: {n_praise}, Lament: {n_lament}.")
    return documents, labels, metadata


def enrich_with_banso_culture(documents, labels, metadata):
    """
    Augment the dataset with Banso cultural expressions of:
        - Kibor  (praise, celebration, worship of Nfor / God)
        - Kighaa (lamentation, grief, crying out in hardship)

    These phrases bridge Lamnso' theology with Biblical sentiment,
    providing cross-linguistic validation for the classifier.

    References:
        - Kibor  : traditional Nso celebration / thanksgiving rite
        - Kighaa : traditional Nso mourning / communal grief ritual
        - Nfor   : the Lamnso' supreme deity (analogous to the Biblical LORD)
    """
    # ── Banso Kibor (Praise) ──────────────────────────────────────────────────
    KIBOR_PHRASES = [
        "nfor is highly exalted in kibor and praise above all things",
        "we celebrate nfor with ntshang song and joyful harvest dances",
        "the harvest is a blessing from nfor our god and sovereign",
        "nfor has filled our barns with corn palm oil and abundance",
        "the lineage of the palace sings praise and thanksgiving to nfor",
        "nfor has done great and glorious and wonderful things for banso people",
        "bless nfor who guards our ancestral gates and keeps us in peace",
        "nfor our light fills the community with kibor and overflowing joy",
        "give thanks to nfor who is eternally good and full of grace and mercy",
        "nfor reigneth on high praise him all the earth and all peoples",
        "praise nfor who brought rain and restored the fields with goodness",
        "nfor is faithful to the children of nso through every generation",
    ]

    # ── Banso Kighaa (Lament) ─────────────────────────────────────────────────
    KIGHAA_PHRASES = [
        "our eyes weep in kighaa for the fallen and beloved elders of nso",
        "nfor look upon our grief and kighaa in this dark season of death",
        "why does the pestilence strike so heavily the highlands of nso",
        "the enemy has destroyed our compound and burned our precious maize",
        "nfor hear our lament and the sound of kighaa rising from the dust",
        "we wander in darkness and seek nfor's mercy and help and strength",
        "deliver us nfor from the plague and bitter famine in these hills",
        "our hearts are broken with kighaa and heavy unbearable sorrow",
        "my god my god why hast thou forsaken the children of nso kingdom",
        "terror and fear are on every side nfor have mercy and deliver us",
        "the elders have died and no one is left to counsel the young men",
        "our songs of joy have turned to kighaa and weeping in the night",
    ]

    for phrase in KIBOR_PHRASES:
        documents.append(phrase)
        labels.append(1)
        metadata.append("Banso Kibor (Praise)")

    for phrase in KIGHAA_PHRASES:
        documents.append(phrase)
        labels.append(0)
        metadata.append("Banso Kighaa (Lament)")

    n_praise = sum(labels)
    n_lament = len(labels) - n_praise
    print(
        f"[3/6] Banso enrichment complete — Total: {len(labels)} docs "
        f"(Praise: {n_praise}, Lament: {n_lament})."
    )
    return documents, labels, metadata


def clean_and_tokenize(text):
    """Lowercase, remove punctuation/digits, split into word tokens."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    text = re.sub(r"\d+", " ", text)
    return [w for w in text.split() if len(w) > 1]


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Custom TF-IDF Vectorizer (pure NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class NumpyTfidfVectorizer:
    """
    Term Frequency – Inverse Document Frequency Vectorizer.

    Mathematical formulation
    ─────────────────────────────────────────────────────────────────────────
        TF(t, d)  = count(t, d) / |d|                   (relative frequency)

        IDF(t, D) = log((1 + |D|) / (1 + DF(t))) + 1   (smoothed, scikit-learn style)
                                                          prevents log(0) and preserves
                                                          single-document terms

        TFIDF(t, d, D) = TF(t,d) × IDF(t, D)

    Final feature vectors are L2-normalised so that cosine similarity equals
    the dot product — important for gradient-based optimisers.
    """

    def __init__(self, min_df=1, max_df=0.95):
        self.min_df = min_df          # minimum document frequency (int or fraction)
        self.max_df = max_df          # maximum document frequency (int or fraction)
        self.vocabulary_   = {}       # word → column index
        self.feature_names_ = []      # column index → word
        self.idf_          = None     # 1-D array of IDF scores

    def fit(self, raw_documents):
        """Build vocabulary and compute IDF weights on the training corpus."""
        tokenized = [clean_and_tokenize(d) for d in raw_documents]
        N = len(tokenized)

        # Document frequency counts
        doc_freq = Counter()
        for doc in tokenized:
            for word in set(doc):
                doc_freq[word] += 1

        # Determine numeric thresholds
        min_cnt = self.min_df if isinstance(self.min_df, int) else max(1, int(self.min_df * N))
        max_cnt = self.max_df if isinstance(self.max_df, int) else int(self.max_df * N)

        # Build sorted vocabulary (deterministic ordering)
        vocab = {}
        for idx, word in enumerate(sorted(doc_freq)):
            df = doc_freq[word]
            if min_cnt <= df <= max_cnt:
                vocab[word] = len(vocab)

        self.vocabulary_    = vocab
        self.feature_names_ = sorted(vocab, key=vocab.get)

        # IDF vector
        self.idf_ = np.array(
            [np.log((1 + N) / (1 + doc_freq[w])) + 1 for w in self.feature_names_],
            dtype=np.float64
        )
        return self

    def transform(self, raw_documents):
        """Convert raw documents to a L2-normalised TF-IDF matrix (n_docs × vocab)."""
        tokenized  = [clean_and_tokenize(d) for d in raw_documents]
        V          = len(self.feature_names_)
        mat        = np.zeros((len(tokenized), V), dtype=np.float64)

        for i, doc in enumerate(tokenized):
            if not doc:
                continue
            total = len(doc)
            for word, cnt in Counter(doc).items():
                if word in self.vocabulary_:
                    j = self.vocabulary_[word]
                    mat[i, j] = cnt / total          # TF

            mat[i] *= self.idf_                      # × IDF

            # L2 row-normalisation
            norm = np.linalg.norm(mat[i])
            if norm > 0:
                mat[i] /= norm

        return mat

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Custom Logistic Regression (pure NumPy)
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(z):
    """Numerically stable sigmoid via clipping."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))


class NumpyLogisticRegression:
    """
    Binary Logistic Regression with L2 regularisation trained by batch
    gradient descent.

    Forward pass
    ─────────────────────────────────────────────────────────────────────────
        ŷ = σ(X w + b)        where σ(z) = 1 / (1 + e^{-z})

    Loss (Binary Cross-Entropy + L2 penalty)
    ─────────────────────────────────────────────────────────────────────────
        L = -(1/m) Σ [y log ŷ + (1-y) log(1-ŷ)]  +  (λ/2m) ‖w‖²

    Gradients
    ─────────────────────────────────────────────────────────────────────────
        ∂L/∂w = (1/m) Xᵀ(ŷ - y)  +  (λ/m) w
        ∂L/∂b = (1/m) Σ(ŷ - y)

    Update rule (SGD)
    ─────────────────────────────────────────────────────────────────────────
        w ← w - η (∂L/∂w)
        b ← b - η (∂L/∂b)
    """

    def __init__(self, lr=0.5, epochs=1500, lambda_reg=0.01):
        self.lr         = lr
        self.epochs     = epochs
        self.lambda_reg = lambda_reg
        self.weights    = None
        self.bias       = 0.0
        self.loss_history = []

    def fit(self, X, y):
        m, n = X.shape
        self.weights  = np.zeros(n)
        self.bias     = 0.0
        self.loss_history = []
        y = np.array(y, dtype=np.float64)

        for epoch in range(1, self.epochs + 1):
            # ── Forward ─────────────────────────────────────────────────────
            z    = X @ self.weights + self.bias
            yhat = _sigmoid(z)

            # ── Loss ────────────────────────────────────────────────────────
            bce  = -np.mean(y * np.log(yhat + 1e-15) + (1 - y) * np.log(1 - yhat + 1e-15))
            l2   = (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
            self.loss_history.append(bce + l2)

            # ── Gradients ───────────────────────────────────────────────────
            dw = (X.T @ (yhat - y)) / m + (self.lambda_reg / m) * self.weights
            db = np.mean(yhat - y)

            # ── Update ──────────────────────────────────────────────────────
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

            if epoch % 500 == 0:
                print(f"  [LR] epoch {epoch:>4d}/{self.epochs} | loss {self.loss_history[-1]:.5f}")

        return self

    def predict_proba(self, X):
        return _sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Custom MLP Classifier (pure NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class NumpyMLP:
    """
    1-hidden-layer Multi-Layer Perceptron for binary classification.

    Architecture
    ─────────────────────────────────────────────────────────────────────────
        Input (n)  →  Dense(H, ReLU)  →  Dense(1, Sigmoid)

    Weight initialisation
    ─────────────────────────────────────────────────────────────────────────
        W₁ ~ He (Kaiming)   :  N(0, √(2/n))  — optimal for ReLU layers
        W₂ ~ Xavier (Glorot):  N(0, √(1/H))  — optimal for sigmoid output

    Forward pass
    ─────────────────────────────────────────────────────────────────────────
        Z₁ = X W₁ + b₁      A₁ = ReLU(Z₁) = max(0, Z₁)
        Z₂ = A₁ W₂ + b₂     A₂ = σ(Z₂)

    Backpropagation (chain rule)
    ─────────────────────────────────────────────────────────────────────────
        δ₂ = A₂ - y                     (output delta)
        ∂L/∂W₂ = (1/m) A₁ᵀ δ₂ + (λ/m) W₂
        ∂L/∂b₂ = (1/m) Σ δ₂

        δ₁ = (δ₂ W₂ᵀ) ⊙ 𝟙[Z₁>0]     (hidden delta via ReLU derivative)
        ∂L/∂W₁ = (1/m) Xᵀ δ₁ + (λ/m) W₁
        ∂L/∂b₁ = (1/m) Σ δ₁

    Optimiser: SGD with Momentum
    ─────────────────────────────────────────────────────────────────────────
        v_θ ← μ v_θ + (1 - μ) ∂L/∂θ
        θ   ← θ - η v_θ

    where μ is the momentum coefficient (default 0.9).
    """

    def __init__(self, hidden_dim=32, lr=0.1, momentum=0.9, weight_decay=0.001, epochs=1500):
        self.hidden_dim   = hidden_dim
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.loss_history = []

    def _init_weights(self, n_in):
        H = self.hidden_dim
        self.W1 = np.random.randn(n_in, H) * np.sqrt(2.0 / n_in)   # He init
        self.b1 = np.zeros((1, H))
        self.W2 = np.random.randn(H, 1)    * np.sqrt(1.0 / H)      # Xavier init
        self.b2 = np.zeros((1, 1))

    def fit(self, X, y):
        m, n = X.shape
        self._init_weights(n)
        self.loss_history = []
        y = np.array(y, dtype=np.float64).reshape(-1, 1)

        # Momentum velocity buffers
        vW1 = np.zeros_like(self.W1); vb1 = np.zeros_like(self.b1)
        vW2 = np.zeros_like(self.W2); vb2 = np.zeros_like(self.b2)

        mu = self.momentum

        for epoch in range(1, self.epochs + 1):
            # ── Forward ─────────────────────────────────────────────────────
            Z1 = X  @ self.W1 + self.b1
            A1 = np.maximum(0, Z1)          # ReLU
            Z2 = A1 @ self.W2 + self.b2
            A2 = _sigmoid(Z2)               # Sigmoid output

            # ── Loss ────────────────────────────────────────────────────────
            bce = -np.mean(y * np.log(A2 + 1e-15) + (1 - y) * np.log(1 - A2 + 1e-15))
            l2  = (self.weight_decay / (2 * m)) * (np.sum(self.W1**2) + np.sum(self.W2**2))
            self.loss_history.append(bce + l2)

            # ── Backprop ─────────────────────────────────────────────────────
            delta2 = A2 - y                                           # (m, 1)
            dW2 = (A1.T @ delta2) / m + (self.weight_decay / m) * self.W2
            db2 = delta2.mean(axis=0, keepdims=True)

            delta1 = (delta2 @ self.W2.T) * (Z1 > 0)                 # ReLU grad
            dW1 = (X.T  @ delta1) / m + (self.weight_decay / m) * self.W1
            db1 = delta1.mean(axis=0, keepdims=True)

            # ── Momentum Update ──────────────────────────────────────────────
            vW1 = mu * vW1 + (1 - mu) * dW1;  self.W1 -= self.lr * vW1
            vb1 = mu * vb1 + (1 - mu) * db1;  self.b1 -= self.lr * vb1
            vW2 = mu * vW2 + (1 - mu) * dW2;  self.W2 -= self.lr * vW2
            vb2 = mu * vb2 + (1 - mu) * db2;  self.b2 -= self.lr * vb2

            if epoch % 500 == 0:
                print(f"  [MLP] epoch {epoch:>4d}/{self.epochs} | loss {self.loss_history[-1]:.5f}")

        return self

    def predict_proba(self, X):
        A1 = np.maximum(0, X @ self.W1 + self.b1)
        return _sigmoid(A1 @ self.W2 + self.b2)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: Evaluation Metrics (pure NumPy)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics from scratch.

    Formulas
    ─────────────────────────────────────────────────────────────────────────
        Accuracy  = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)    — of all predicted Praise, how many correct?
        Recall    = TP / (TP + FN)    — of all true Praise, how many detected?
        F1-Score  = 2 × (P × R) / (P + R)   — harmonic mean of P and R
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc  = (TP + TN) / len(y_true)          if len(y_true) > 0       else 0.0
    prec = TP / (TP + FP)                   if (TP + FP) > 0         else 0.0
    rec  = TP / (TP + FN)                   if (TP + FN) > 0         else 0.0
    f1   = 2*prec*rec / (prec + rec)        if (prec + rec) > 0      else 0.0

    return {
        "accuracy":          acc,
        "precision":         prec,
        "recall":            rec,
        "f1_score":          f1,
        "confusion_matrix":  {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
    }


def print_metrics(label, m):
    cm = m["confusion_matrix"]
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {m['accuracy']:.4%}")
    print(f"  Precision : {m['precision']:.4%}")
    print(f"  Recall    : {m['recall']:.4%}")
    print(f"  F1-Score  : {m['f1_score']:.4%}")
    print(f"  Confusion : TP={cm['TP']}  TN={cm['TN']}  FP={cm['FP']}  FN={cm['FN']}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 6: Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(lr_history, mlp_history, out_dir):
    """Export epoch-loss curves for both custom models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colours = ["#1a73e8", "#d93025"]
    titles  = ["NumPy Logistic Regression Loss", "NumPy MLP Loss"]
    hists   = [lr_history, mlp_history]

    for ax, hist, title, col in zip(axes, hists, titles, colours):
        ax.plot(hist, color=col, linewidth=2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary Cross-Entropy Loss")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "Day 27 · Text Classification Training Curves\n"
        "Fonyuy-pounds  |  Module 05  |  MarkGPT 60-Day Curriculum",
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "classification_loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved loss curves → {path}")


def plot_metrics_comparison(all_metrics, labels, out_dir):
    """Bar chart comparing Accuracy & F1-Score across all evaluated models."""
    accs = [m["accuracy"]  for m in all_metrics]
    f1s  = [m["f1_score"]  for m in all_metrics]

    x     = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width/2, accs, width, label="Accuracy", color="#34a853", edgecolor="white")
    ax.bar(x + width/2, f1s,  width, label="F1-Score", color="#fbbc05", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(
        "Day 27 · Model Performance Comparison\n"
        "Psalm Praise vs. Lament Classification  (+ Banso Kibor / Kighaa)",
        fontsize=11, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Annotate bars
    for rect in ax.patches:
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, h + 0.01,
            f"{h:.1%}", ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()
    path = os.path.join(out_dir, "classification_metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics chart  → {path}")


def plot_feature_importance(weights, feature_names, out_dir, top_n=15):
    """Horizontal bar chart of top praise/lament feature terms from LR weights."""
    sorted_idx = np.argsort(weights)

    lament_idx = sorted_idx[:top_n]
    praise_idx  = sorted_idx[-top_n:][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Praise features
    ax1.barh(range(top_n), weights[praise_idx[::-1]], color="#34a853")
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(feature_names[praise_idx[::-1]], fontsize=9)
    ax1.set_xlabel("LR Weight (positive → Praise)")
    ax1.set_title("Top Praise Features", fontweight="bold")
    ax1.axvline(0, color="black", linewidth=0.8)

    # Lament features
    ax2.barh(range(top_n), weights[lament_idx], color="#d93025")
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(feature_names[lament_idx], fontsize=9)
    ax2.set_xlabel("LR Weight (negative → Lament)")
    ax2.set_title("Top Lament Features", fontweight="bold")
    ax2.axvline(0, color="black", linewidth=0.8)

    plt.suptitle(
        "Day 27 · Feature Importance Analysis\n"
        "Logistic Regression Weights (Praise=+, Lament=−)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 7: Stratified Train / Test Split (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def stratified_train_test_split(documents, labels, test_size=0.2):
    """
    80/20 stratified split preserving class balance.
    Implemented from scratch (no sklearn dependency required).
    """
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]

    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    n_pos_test = max(1, int(len(pos_idx) * test_size))
    n_neg_test = max(1, int(len(neg_idx) * test_size))

    test_idx  = pos_idx[:n_pos_test]  + neg_idx[:n_neg_test]
    train_idx = pos_idx[n_pos_test:]  + neg_idx[n_neg_test:]

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    X_train = [documents[i] for i in train_idx]
    y_train = [labels[i]    for i in train_idx]
    X_test  = [documents[i] for i in test_idx]
    y_test  = [labels[i]    for i in test_idx]

    return X_train, y_train, X_test, y_test, train_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. ── Data loading ────────────────────────────────────────────────────
    psalm_docs = load_psalms(WORKSPACE_ROOT)
    documents, labels, metadata = get_labeled_psalms(psalm_docs)
    documents, labels, metadata = enrich_with_banso_culture(documents, labels, metadata)

    # 2. ── Stratified split ───────────────────────────────────────────────
    X_train, y_train, X_test, y_test, train_idx, test_idx = stratified_train_test_split(
        documents, labels, test_size=0.2
    )
    print(f"\n[4/6] Split — Train: {len(y_train)} | Test: {len(y_test)}")

    # 3. ── TF-IDF Vectorization ───────────────────────────────────────────
    print("\n[5/6] Fitting custom NumPy TF-IDF Vectorizer …")
    vectorizer = NumpyTfidfVectorizer(min_df=2, max_df=0.90)
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)
    print(f"       Vocabulary size: {len(vectorizer.feature_names_)} terms  |  "
          f"Feature matrix: {X_tr.shape}")

    # 4. ── Train models ───────────────────────────────────────────────────
    print("\n[6/6] Training models …\n")

    print("── NumPy Logistic Regression ──────────────────────")
    lr_model = NumpyLogisticRegression(lr=0.5, epochs=1500, lambda_reg=0.01)
    lr_model.fit(X_tr, y_train)

    print("\n── NumPy Multi-Layer Perceptron (H=32, ReLU) ──────")
    mlp_model = NumpyMLP(hidden_dim=32, lr=0.1, momentum=0.9, weight_decay=0.001, epochs=1500)
    mlp_model.fit(X_tr, y_train)

    # 5. ── Evaluate ───────────────────────────────────────────────────────
    lr_preds  = lr_model.predict(X_te)
    mlp_preds = mlp_model.predict(X_te)
    lr_m      = compute_metrics(y_test, lr_preds)
    mlp_m     = compute_metrics(y_test, mlp_preds)

    print_metrics("Custom NumPy Logistic Regression", lr_m)
    print_metrics("Custom NumPy MLP  (H=32, ReLU, Momentum SGD)", mlp_m)

    # 6. ── Feature importance ─────────────────────────────────────────────
    weights       = lr_model.weights
    feature_names = np.array(vectorizer.feature_names_)
    sorted_idx    = np.argsort(weights)

    print("\n\n══ Top 15 Praise-predictive Features ══════════════════════")
    for rank, idx in enumerate(sorted_idx[-15:][::-1], 1):
        print(f"  {rank:2d}. {feature_names[idx]:<18s}  w = +{weights[idx]:.4f}")

    print("\n══ Top 15 Lament-predictive Features ══════════════════════")
    for rank, idx in enumerate(sorted_idx[:15], 1):
        print(f"  {rank:2d}. {feature_names[idx]:<18s}  w =  {weights[idx]:.4f}")

    # 7. ── Banso-specific evaluation ──────────────────────────────────────
    test_meta_list = [metadata[i] for i in test_idx]
    banso_docs, banso_true, banso_meta = [], [], []
    for doc, label, meta in zip(X_test, y_test, test_meta_list):
        if "Banso" in meta:
            banso_docs.append(doc)
            banso_true.append(label)
            banso_meta.append(meta)

    print("\n\n══ Banso Vernacular Predictions ════════════════════════════")
    if banso_docs:
        X_ban     = vectorizer.transform(banso_docs)
        lr_ban    = lr_model.predict(X_ban)
        mlp_ban   = mlp_model.predict(X_ban)
        for i, doc in enumerate(banso_docs):
            true_s  = "Praise" if banso_true[i] == 1 else "Lament"
            lr_s    = "Praise" if lr_ban[i]    == 1 else "Lament"
            mlp_s   = "Praise" if mlp_ban[i]   == 1 else "Lament"
            correct = "✓" if lr_ban[i] == banso_true[i] else "✗"
            print(f"  {correct} [{banso_meta[i]}]")
            print(f"    \"{doc[:70]}…\"" if len(doc) > 70 else f"    \"{doc}\"")
            print(f"    True: {true_s:<7} | LR: {lr_s:<7} | MLP: {mlp_s}")
        ban_m = compute_metrics(banso_true, lr_ban)
        print(f"\n  Banso subset accuracy (LR): {ban_m['accuracy']:.2%}")
    else:
        print("  No Banso samples fell in the test split this seed — re-run with a different seed.")

    # 8. ── Scikit-learn benchmark ─────────────────────────────────────────
    sk_lr_m   = None
    sk_mlp_m  = None
    model_labels  = ["Custom LR", "Custom MLP"]
    all_metrics   = [lr_m, mlp_m]

    try:
        from sklearn.linear_model import LogisticRegression as SklearnLR
        from sklearn.neural_network import MLPClassifier as SklearnMLP
        from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf

        sk_vec   = SkTfidf(min_df=2, max_df=0.9, norm="l2", smooth_idf=True)
        X_tr_sk  = sk_vec.fit_transform(X_train)
        X_te_sk  = sk_vec.transform(X_test)

        sk_lr  = SklearnLR(C=1.0, max_iter=1500, random_state=42)
        sk_lr.fit(X_tr_sk, y_train)
        sk_lr_m = compute_metrics(y_test, sk_lr.predict(X_te_sk))
        print_metrics("Scikit-Learn Logistic Regression (benchmark)", sk_lr_m)

        sk_mlp = SklearnMLP(hidden_layer_sizes=(32,), activation="relu",
                            solver="adam", max_iter=1500, random_state=42)
        sk_mlp.fit(X_tr_sk, y_train)
        sk_mlp_m = compute_metrics(y_test, sk_mlp.predict(X_te_sk))
        print_metrics("Scikit-Learn MLP  (benchmark)", sk_mlp_m)

        model_labels += ["Sklearn LR", "Sklearn MLP"]
        all_metrics  += [sk_lr_m, sk_mlp_m]

    except ImportError:
        print("\n[INFO] scikit-learn not found — skipping benchmarks.")

    # 9. ── Save plots ─────────────────────────────────────────────────────
    plot_loss_curves(lr_model.loss_history, mlp_model.loss_history, EXERCISES_DIR)
    plot_metrics_comparison(all_metrics, model_labels, EXERCISES_DIR)
    plot_feature_importance(weights, feature_names, EXERCISES_DIR)

    # 10. ─ Summary banner ─────────────────────────────────────────────────
    target_hit = lr_m["accuracy"] >= 0.80 or mlp_m["accuracy"] >= 0.80
    print("\n" + "=" * 75)
    print("  DAY 27 PIPELINE COMPLETE")
    print(f"  Custom LR Accuracy   : {lr_m['accuracy']:.2%}  (F1 {lr_m['f1_score']:.2%})")
    print(f"  Custom MLP Accuracy  : {mlp_m['accuracy']:.2%}  (F1 {mlp_m['f1_score']:.2%})")
    print(f"  Target ≥ 80%         : {'✓ ACHIEVED' if target_hit else '✗ Not yet — tune hyperparameters'}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
