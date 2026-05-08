"""
Mini-Project 3: Character-Level MLP Language Model
===================================================
Module 03 - Neural Networks from Scratch | Day 18/60

Build a character-level language model using an MLP trained on Psalm 23
in both English and Banso (Lamnso') translation.

Tasks:
1. Prepare character-level data from Psalm 23 (English + Banso)
2. Build an MLP with embedding + hidden layers
3. Train on both languages separately
4. Generate continuations and compare outputs
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================================
# DATA: Psalm 23 in English (KJV) and Banso (Lamnso')
# ============================================================================

PSALM_23_ENGLISH = """The Lord is my shepherd I shall not want
He maketh me to lie down in green pastures
He leadeth me beside the still waters
He restoreth my soul
He leadeth me in the paths of righteousness for his name sake
Yea though I walk through the valley of the shadow of death
I will fear no evil for thou art with me
Thy rod and thy staff they comfort me
Thou preparest a table before me in the presence of mine enemies
Thou anointest my head with oil my cup runneth over
Surely goodness and mercy shall follow me all the days of my life
And I will dwell in the house of the Lord for ever"""

PSALM_23_BANSO = """Nyuiy a wir wiy a la shi wan
A gha kiy moo a nshi bvu njiy
A gha la moo a mbaa nduu shi ko
A gha yooni ntam wiy
A gha la moo a menjiy ma bvuu fo shin nyam yi
Ndaa moo nkiy ta weh a nshi kuuyn wuu
Mi ta bvong shi yi fo wan a weh wan boo moo
Nkang wan boo ngkiy nkwan yi gha kooni won moo
Wan gha nyehsi fainyi a nshi fo mi wan a ki bfoh ba mi
Wan gha nyeh fimi ntsu wan a nshi nyu fo sor wiy gha shey
A bvuu fo shin nyuiy boo ki dzeyn gha la weh boo moo
Mi gha yii a nda Nyuiy fo a mfey"""


# ============================================================================
# PART 1: Character Vocabulary & Data Preparation
# ============================================================================

class CharDataset:
    """Builds a character-level dataset with sliding window contexts."""

    def __init__(self, text, context_size=5):
        self.text = text.lower()
        self.context_size = context_size

        # Build vocabulary
        chars = sorted(set(self.text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)

        # Build training examples
        self.X, self.y = self._build_examples()

    def _build_examples(self):
        encoded = [self.char_to_idx[c] for c in self.text]
        X, y = [], []
        for i in range(len(encoded) - self.context_size):
            X.append(encoded[i : i + self.context_size])
            y.append(encoded[i + self.context_size])
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)


# ============================================================================
# PART 2: Character-Level MLP Model (from scratch with numpy)
# ============================================================================

class CharMLP:
    """
    Character-level MLP language model.

    Architecture:
      Input: context_size character indices
      -> Embedding lookup (vocab_size x embed_dim)
      -> Flatten to (context_size * embed_dim)
      -> Hidden layer (tanh activation)
      -> Output layer (logits over vocab)
    """

    def __init__(self, vocab_size, context_size, embed_dim=16, hidden_dim=128, lr=0.01):
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Xavier initialization
        self.C = np.random.randn(vocab_size, embed_dim) * 0.5
        flat_dim = context_size * embed_dim
        self.W1 = np.random.randn(flat_dim, hidden_dim) * np.sqrt(2.0 / flat_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, vocab_size) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(vocab_size)

    def forward(self, X):
        """Forward pass. X shape: (batch, context_size) of int indices."""
        self.emb = self.C[X]                                    # (B, ctx, emb)
        self.flat = self.emb.reshape(X.shape[0], -1)            # (B, ctx*emb)
        self.h_pre = self.flat @ self.W1 + self.b1              # (B, hidden)
        self.h = np.tanh(self.h_pre)                            # (B, hidden)
        self.logits = self.h @ self.W2 + self.b2                # (B, vocab)

        # Stable softmax
        exp_logits = np.exp(self.logits - self.logits.max(axis=1, keepdims=True))
        self.probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return self.probs

    def loss(self, probs, targets):
        """Negative log-likelihood loss."""
        B = len(targets)
        log_probs = -np.log(probs[np.arange(B), targets] + 1e-9)
        return log_probs.mean()

    def backward(self, X, targets):
        """Backpropagation through all layers."""
        B = len(targets)

        # Gradient of loss w.r.t. logits (softmax + cross-entropy shortcut)
        dlogits = self.probs.copy()
        dlogits[np.arange(B), targets] -= 1
        dlogits /= B

        # Output layer gradients
        dW2 = self.h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T

        # Hidden layer gradients (tanh derivative)
        dh_pre = dh * (1 - self.h ** 2)
        dW1 = self.flat.T @ dh_pre
        db1 = dh_pre.sum(axis=0)
        dflat = dh_pre @ self.W1.T

        # Embedding gradients (vectorized)
        demb = dflat.reshape(self.emb.shape)
        dC = np.zeros_like(self.C)
        np.add.at(dC, X, demb)

        # Update parameters
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.C  -= self.lr * dC

    def train_step(self, X, targets):
        probs = self.forward(X)
        l = self.loss(probs, targets)
        self.backward(X, targets)
        return l


# ============================================================================
# PART 3: Training Loop
# ============================================================================

def train_model(dataset, n_epochs=300, batch_size=64, lr=0.05, embed_dim=16, hidden_dim=128):
    """Train a CharMLP on the given dataset."""
    model = CharMLP(
        vocab_size=dataset.vocab_size,
        context_size=dataset.context_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        lr=lr
    )

    losses = []
    n = len(dataset)
    for epoch in range(n_epochs):
        # Shuffle
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            Xb, yb = dataset.X[idx], dataset.y[idx]
            l = model.train_step(Xb, yb)
            epoch_loss += l
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | Loss: {avg_loss:.4f}")

    return model, losses


# ============================================================================
# PART 4: Text Generation
# ============================================================================

def generate(model, dataset, seed_text, length=150, temperature=0.8):
    """Generate text from a trained model given a seed."""
    seed = seed_text.lower()
    # Pad or truncate seed to context_size
    ctx = seed[-model.context_size:]
    if len(ctx) < model.context_size:
        ctx = ' ' * (model.context_size - len(ctx)) + ctx

    result = list(ctx)
    context = [dataset.char_to_idx.get(c, 0) for c in ctx]

    for _ in range(length):
        X = np.array([context])
        probs = model.forward(X)[0]

        # Temperature sampling
        logits = np.log(probs + 1e-9) / temperature
        exp_l = np.exp(logits - logits.max())
        scaled_probs = exp_l / exp_l.sum()

        idx = np.random.choice(len(scaled_probs), p=scaled_probs)
        result.append(dataset.idx_to_char[idx])
        context = context[1:] + [idx]

    return ''.join(result)


# ============================================================================
# PART 5: Main Driver
# ============================================================================

def main():
    np.random.seed(42)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("MINI-PROJECT 3: Character-Level MLP Language Model")
    print("Module 03 Review — Day 18/60")
    print("=" * 70)

    # --- Prepare datasets ---
    CONTEXT_SIZE = 5
    print("\n[1] Preparing datasets...")
    ds_en = CharDataset(PSALM_23_ENGLISH, context_size=CONTEXT_SIZE)
    ds_bn = CharDataset(PSALM_23_BANSO, context_size=CONTEXT_SIZE)

    print(f"  English — Vocab: {ds_en.vocab_size} chars, Samples: {len(ds_en)}")
    print(f"  Banso   — Vocab: {ds_bn.vocab_size} chars, Samples: {len(ds_bn)}")
    print(f"  English vocab: {''.join(sorted(ds_en.char_to_idx.keys()))}")
    print(f"  Banso   vocab: {''.join(sorted(ds_bn.char_to_idx.keys()))}")

    # --- Train English model ---
    print("\n[2] Training on English Psalm 23...")
    model_en, losses_en = train_model(ds_en, n_epochs=500, lr=0.05, embed_dim=16, hidden_dim=128)

    # --- Train Banso model ---
    print("\n[3] Training on Banso Psalm 23...")
    model_bn, losses_bn = train_model(ds_bn, n_epochs=500, lr=0.05, embed_dim=16, hidden_dim=128)

    # --- Generate samples ---
    print("\n[4] Generating text samples...")
    print("\n" + "-" * 50)
    print("ENGLISH MODEL — Seed: 'the l'")
    print("-" * 50)
    for i in range(3):
        sample = generate(model_en, ds_en, "the l", length=120, temperature=0.7 + i * 0.2)
        print(f"  Sample {i+1} (T={0.7 + i*0.2:.1f}): {sample}")

    print("\n" + "-" * 50)
    print("BANSO MODEL — Seed: 'nyuiy'")
    print("-" * 50)
    for i in range(3):
        sample = generate(model_bn, ds_bn, "nyuiy", length=120, temperature=0.7 + i * 0.2)
        print(f"  Sample {i+1} (T={0.7 + i*0.2:.1f}): {sample}")

    # --- Comparison Analysis ---
    print("\n[5] Comparative Analysis")
    print("-" * 50)
    print(f"  {'Metric':<25} {'English':>12} {'Banso':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Vocabulary size':<25} {ds_en.vocab_size:>12} {ds_bn.vocab_size:>12}")
    print(f"  {'Training samples':<25} {len(ds_en):>12} {len(ds_bn):>12}")
    print(f"  {'Final loss':<25} {losses_en[-1]:>12.4f} {losses_bn[-1]:>12.4f}")
    print(f"  {'Unique chars in text':<25} {ds_en.vocab_size:>12} {ds_bn.vocab_size:>12}")

    # --- Plot loss curves ---
    print("\n[6] Saving loss curve plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(losses_en, color='#2196F3', linewidth=2)
    axes[0].set_title('English Psalm 23 — Training Loss', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(losses_bn, color='#4CAF50', linewidth=2)
    axes[1].set_title('Banso Psalm 23 — Training Loss', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Cross-Entropy Loss')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Mini-Project 3: Character-Level MLP — Loss Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'day18_loss_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved to {plot_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("MINI-PROJECT 3 COMPLETE")
    print("=" * 70)
    print("""
Key Observations to note in your journal:
  1. How does the vocab size differ between English and Banso?
  2. Which model converges faster and why?
  3. Does the Banso model capture tonal markers or diacritics?
  4. How does temperature affect generation quality?
  5. What are the limits of a fixed-context MLP vs. a recurrent model?
""")


if __name__ == "__main__":
    main()
