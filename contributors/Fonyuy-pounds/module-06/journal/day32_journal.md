# Day 32 Learning Journal: Scaled Dot-Product Attention

**Date:** 2026-06-01  
**Author:** Fonyuy-pounds  
**Module:** Module 06 — The Transformer Architecture  
**Day:** 32/60

---

## 1. Today's Objective

Implement **Scaled Dot-Product Attention** entirely from scratch in PyTorch (no `nn.MultiheadAttention`), verify the implementation's mathematical alignment with PyTorch's native `F.scaled_dot_product_attention`, and apply attention to disambiguate the polysemous Banso (Lamnso') term *Nfor* (God vs. King) in contrasting theological phrases.

---

## 2. The Core Formula — Line-By-Line Derivation

The complete forward pass in one equation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right) V$$

### What each component does:

| Component | Shape | Interpretation |
|-----------|-------|----------------|
| $Q$ (Query) | $(B, H, T_q, d_k)$ | "What am I searching for?" — maps from the current token's representation to a search key |
| $K$ (Key) | $(B, H, T_k, d_k)$ | "What am I?" — maps from every token to a label/descriptor |
| $V$ (Value) | $(B, H, T_v, d_v)$ | "What do I contain?" — the information payload retrieved when attended to |
| $QK^T$ | $(B, H, T_q, T_k)$ | Dot-product similarity score between every query–key pair |
| $/ \sqrt{d_k}$ | scalar | Stabilisation factor |
| $\text{softmax}(\cdot)$ | $(B, H, T_q, T_k)$ | Converts raw scores to a valid probability distribution over source positions |
| $\cdot V$ | $(B, H, T_q, d_v)$ | Weighted sum of values, i.e., a soft retrieval |

### Step-by-step numerical derivation (d_k = 2, 2-position example)

```
Q = [[1, 0],    K = [[1, 0],    V = [[2, 0],
     [0, 1]]         [1, 1]]         [0, 3]]

Step 1: QK^T
= [[1·1 + 0·0,  1·1 + 0·1],   = [[1, 1],
   [0·1 + 1·0,  0·1 + 1·1]]      [0, 1]]

Step 2: Scale (sqrt(2) ≈ 1.414)
= [[0.707, 0.707],
   [0.000, 0.707]]

Step 3: Softmax(row-wise)
Row 0: exp([0.707, 0.707]) = [2.028, 2.028], sum=4.056 → [0.5, 0.5]
Row 1: exp([0.000, 0.707]) = [1.000, 2.028], sum=3.028 → [0.330, 0.670]

Step 4: Multiply by V
= [[0.5·2 + 0.5·0,  0.5·0 + 0.5·3],    = [[1.0,  1.5],
   [0.33·2 + 0.67·0, 0.33·0 + 0.67·3]]     [0.66, 2.01]]
```

Position 0 receives a 50/50 blend of both values. Position 1 is skewed toward position 1 (higher key similarity).

---

## 3. Why Scale by √d_k?

**The variance problem:**

When $Q$ and $K$ entries are drawn from $\mathcal{N}(0, 1)$, the dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has:

$$\text{Var}(q \cdot k) = d_k$$

So the standard deviation grows as $\sqrt{d_k}$. For typical $d_k = 64$ (as in the original paper), dot products are ~8× larger in magnitude.

**Why this breaks training:**

Large logits push softmax into nearly one-hot distributions:
$$\text{softmax}([8.0, 0.0, 0.0]) = [0.9997, 0.0001, 0.0001]$$

The gradient of softmax at this extreme saturates to ~0 — the **attention head stops learning**.

**The fix:**

Divide by $\sqrt{d_k}$ to normalise variance back to 1:
$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

This keeps the softmax in its "smooth" regime where gradients are informative.

---

## 4. The Causal Mask

For autoregressive language modelling, position $i$ must only attend to positions $\leq i$ (past tokens, not future ones). We achieve this by adding a mask before the softmax:

$$\text{scores}_{ij}^{\text{masked}} = \text{scores}_{ij} + M_{ij}$$

where:
$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

Because $e^{-\infty} = 0$, the softmax output at masked positions is exactly zero — effectively blocked.

The resulting lower-triangular mask for seq_len=4:
```
Position:  0      1      2      3
0:         0     -∞     -∞     -∞
1:         0      0     -∞     -∞
2:         0      0      0     -∞
3:         0      0      0      0
```

---

## 5. Implementation Architecture

```
ScaledDotProductAttention
  └── __init__(d_k):  store scale = 1/√d_k
  └── forward(Q, K, V, mask):
        1. scores  = Q @ K^T          (batch matmul)
        2. scores *= scale            (normalise)
        3. scores += mask             (optional, causal)
        4. weights = softmax(scores)  (prob dist)
        5. output  = weights @ V      (weighted sum)
        return output, weights
```

**Parameter count: 0** — pure computation, no learnable weights. The projection matrices $W_Q$, $W_K$, $W_V$ live in the `MultiHeadAttention` wrapper layer (implemented Day 33).

---

## 6. Verification Results

The forward pass output and all input gradients ($\nabla_Q$, $\nabla_K$, $\nabla_V$) were verified to align with PyTorch's native `F.scaled_dot_product_attention` to within `atol=1e-5` for forward values and `atol=1e-4` for gradients:

| Test Case | Forward Match | ∇Q Match | ∇K Match | ∇V Match |
|-----------|:---:|:---:|:---:|:---:|
| Without Mask | ✅ | ✅ | ✅ | ✅ |
| With Causal Mask | ✅ | ✅ | ✅ | ✅ |

Small numerical differences (< 1e-7 in forward, < 1e-5 in backward) are attributable to floating point operation ordering differences, not mathematical errors.

---

## 7. Banso Linguistic Validation: Context Disambiguation of *Nfor*

### Problem Statement

The Lamnso' (Banso) word **"Nfor"** is a perfect real-world case of polysemy that a language model must resolve through context:

| Context | Meaning | Semantic Field |
|---------|---------|----------------|
| Theological/Praise (*Kibor*) | God, King of Heaven | Celestial, Divine |
| Political/Governance | King, Fon, Traditional Ruler | Human authority, Royal |

A system incapable of context modelling (e.g. bag-of-words, TF-IDF) would represent both instances identically — an irreparable semantic error.

### Phrases Used

| Phrase | Lamnso' | Translation |
|--------|---------|-------------|
| Praise | *Nfor a shii kibor* | "God is worthy of praise" |
| Lament | *Nfor a shii kighaa* | "The King hears lamentation" |

### Key Observations

After computing self-attention over the 4-dimensional conceptual embedding space:

1. **In the praise phrase**, the embedding of *Nfor* after attention absorbed significant weight from *kibor* (praise token), shifting its representation toward the Celestial/Divine pole (Dim 0: High, Dim 2: High praise).

2. **In the lament phrase**, the embedding of *Nfor* after attention absorbed significant weight from *kighaa* (lament/governance token), shifting its representation toward the Royal/Sorrowful pole (Dim 1: High, Dim 3: High lament).

This demonstrates that **attention alone (without any training)** creates contextually distinct representations for an identical surface form — the fundamental capability needed for a MarkGPT model to generate linguistically accurate Banso theological text.

---

## 8. What Confused Me & How I Resolved It

| Confusion | Resolution |
|-----------|-----------|
| Why `transpose(-2, -1)` and not `.T`? | `.T` is only defined for 2-D tensors in PyTorch. `transpose(-2, -1)` operates on the last two dims of any rank tensor — essential for batched multi-head shapes. |
| Why does `attn_mask` in PyTorch take a *float* mask and not a *bool* mask? | Both are supported; a float mask (with `-inf` / `0.0`) is added directly to logits. A bool mask (where `False` means masked) triggers the float conversion internally. Using float masks is faster and more numerically explicit. |
| Does the softmax applied across `-inf` produce `nan`? | Only if **all** values in a row are `-inf` (which would produce a `0/0 = nan`). A proper causal mask always allows at least one position per row to be attended (the diagonal), so this never occurs in valid sequences. |

---

## 9. What I Want to Explore Tomorrow (Day 33)

- **Multi-Head Attention**: Wrap `ScaledDotProductAttention` with $h$ parallel heads, project with $W_Q$, $W_K$, $W_V$, concat heads, apply $W_O$
- **Sinusoidal Positional Encoding**: Implement `PE(pos, 2i) = sin(pos / 10000^{2i/d})` and verify the geometric property that relative positions produce learnable offsets
- **BertViz (optional)**: Visualise what different heads specialise in on a pre-trained BERT model

---

## 10. Resources Used

- [x] Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- [x] PyTorch Documentation: `torch.nn.functional.scaled_dot_product_attention`
- [x] *The Illustrated Transformer* — Jay Alammar (jalammar.github.io)
- [x] Day 31 journal (reading guide equations used as foundation)
- [x] Syllabus Lesson: L32.1_scaled-dot-product / L32.2_attention-matrix
