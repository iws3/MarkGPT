# Day 31 Learning Journal: "Attention Is All You Need"

**Date:** 2026-05-29  
**Author:** Fonyuy-pounds  
**Module:** Module 06 - The Transformer Architecture  
**Day:** 31/60

---

## Morning Lesson: The 2017 Breakthrough (45-60 min)

### What I Learned Today

**Why RNNs Have Fundamental Limitations:**

- **Sequential Processing Bottleneck:** RNNs process tokens one at a time, left-to-right. This makes parallelization impossible—modern GPUs wait idle most of the time.
- **Long-Range Forgetting:** Information from early tokens gets compressed through hidden states. By token 100, information from token 1 is essentially forgotten (vanishing gradients despite LSTM tricks).
- **Computational Inefficiency:** Depth = number of tokens. To connect token 1 to token 100, information must traverse 100 layers, making deep models unstable.
- **Linear vs. Logarithmic Paths:** In an RNN, the shortest path from token 1 to token 100 is 100 steps. In Transformers with attention, ANY two tokens are connected in 1 step.

**The Transformer Solution - "Attention Is All You Need":**

The Vaswani et al. (2017) paper proposed a revolutionary idea: **use only attention, no recurrence at all**.

Instead of sequential RNNs, process all tokens in **parallel** using:

- **Scaled Dot-Product Attention:** Each token can "look at" all other tokens simultaneously
- **Multi-Head Attention:** Multiple attention heads specialize in different relationships (e.g., one head tracks pronouns, another tracks verb-object relations)
- **Positional Encoding:** Add position information (no recurrence needed)
- **Feed-Forward Networks:** Transform each position after attention
- **Layer Normalization & Residual Connections:** Stabilize training of deep networks

**The Attention Formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:

- **Q (Query):** "What am I looking for?"
- **K (Key):** "What am I?" (for all other positions)
- **V (Value):** "Here's my information"
- **Scaling by √d_k:** Prevents dot products from becoming too large (keeps gradients in good range)

**Why This Works:**

- All positions process in parallel → GPU-friendly
- Any token can directly attend to any other → no information loss
- Depth is independent of sequence length → can train very deep models
- Attention weights are interpretable → we can see what the model focuses on

### What Confused Me

- [x] Why scale by exactly √d_k and not some other value?
  - **Clarity gained:** When d_k is large, dot products become huge, making softmax nearly one-hot (gradient → 0). Scaling by √d_k keeps variance ~1, so softmax remains smooth and gradients flow well.

- [x] How positional encoding works without recurrence
  - **Clarity gained:** We don't need recurrence to encode position! We can add sinusoidal signals: PE(pos, 2i) = sin(pos/10000^(2i/d)) and PE(pos, 2i+1) = cos(pos/10000^(2i/d)). The model learns to use these signals.

- [ ] The exact relationship between multi-head attention and ensemble learning
  - Still pondering: Each head learns different attention patterns. Is this like an ensemble? Need to visualize actual attention heads.

- [x] Why layer normalization specifically (not batch norm)?
  - **Clarity gained:** Layer norm normalizes across features (not across batch), so it's independent of batch size. Transformers need this consistency because they're used at inference time with batch size = 1.

### What I Want to Explore Next

- Implement scaled dot-product attention from scratch (Day 32)
- Visualize what different attention heads learn (BertViz)
- Compare performance of Transformers vs. RNNs on text tasks
- Understand why MarkGPT uses this architecture for Banso text

---

## Midday Exercise (30-45 min)

### Exercise 1: Read & Annotate the Paper

**Task:** Carefully read Vaswani et al. (2017) and annotate key sections

- [x] Completed

**Key Sections Annotated:**

- **Section 3.1 (Scaled Dot-Product Attention):** Understood why scaling prevents gradient problems
- **Section 3.2 (Multi-Head Attention):** Realized each head learns different position relationships
- **Section 3.3 (Feed-Forward):** Two-layer MLP per position, same across all positions
- **Section 3.4 (Embeddings & Positional Encoding):** Position + token embedding combined
- **Section 4 (Results):** Transformers outperformed previous SOTA on WMT'14 English-German and English-French

### Exercise 2: Architecture Diagram

**Task:** Draw the complete Transformer architecture from memory

- [x] Completed

**Diagram Components Drawn:**

- Input embedding + positional encoding layer
- Multi-head attention (show that output = concatenation of 8 heads)
- Feed-forward network (Dense → ReLU → Dense)
- Residual connections around each sub-layer
- Layer normalization
- Encoder stack (6 layers)
- Decoder stack (6 layers) with causal masking in self-attention
- Output linear projection + softmax

### Exercise 3: RNN vs. Transformer Comparison

**Task:** Compare computational complexity and capabilities

- [x] Completed

**Comparison Table:**

| Aspect | RNN | Transformer |
|--------|-----|-------------|
| **Sequence Length** | T | T |
| **Self-Attention Steps** | T (sequential) | 1 (parallel) |
| **Max Path Length** | O(T) | O(1) |
| **Operations per Step** | O(d²) | O(T·d²) per layer |
| **Parallel? (GPU-friendly)** | No | Yes ✓ |
| **Long-range dependency learning** | Difficult | Easy ✓ |
| **Example: 1000-token sequence** | 1000 sequential steps | 1 parallel step per layer |

**Key Insight:** Transformers trade sequence dependency (harder to capture order) for parallelization. That's why positional encoding is crucial!

### Exercise 4: Attention Mechanism Intuition

**Task:** Understand the intuition behind Q, K, V

- [x] Completed

**Librarian Analogy:**

- **Query (Q):** "I'm looking for information about Banso language linguistics"
- **Key (K):** "I'm a chapter about phonology", "I'm a chapter about morphology", "I'm a chapter about syntax"
- **Value (V):** The actual content of each chapter

The attention mechanism:

1. Compares Query to all Keys (relevance scores)
2. Softmax normalizes these into attention weights
3. Sums the Values weighted by how relevant they are
4. Returns the weighted average of values

---

## Evening Journal (15 min)

### Summary (3 Sentences)

1. **What I learned:** The Transformer architecture is revolutionary because it replaces sequential RNNs with parallel attention mechanisms, allowing information to flow directly between any two positions. This single change (from recurrence to pure attention) enabled scaling to billions of parameters and became the foundation for all modern LLMs including those that power MarkGPT.

2. **What confused me:** The scaling factor √d_k seemed arbitrary until I realized it's a precision hack—preventing gradient vanishing when dot products get too large. It's a small detail with huge practical impact.

3. **What I want to explore:** I'm excited to implement scaled dot-product attention from scratch tomorrow and actually code up the mathematics that I've been reading about today. I want to see how the attention weights form patterns in real data.

---

## Resources Used

- [x] Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*
- [x] Lesson: L31_attention_paper.md
- [x] Lesson: L31.1_attention-paper/ (exercises)
- [x] The Illustrated Transformer (<http://jalammar.github.io/illustrated-transformer/>) for visualizations
- [x] Paper exercises in `day31_reading_guide.md`

---

## Code Review Checklist

- [x] Read and understood core Attention formula
- [x] Annotated all major equations
- [x] Drew architecture from memory (no peeking!)
- [x] Grasped why Transformers solve RNN limitations
- [x] Understood positional encoding necessity
- [x] Ready to implement tomorrow

---

## Important Notes for Day 32

**Day 32 will be about implementation!** Here's what to prepare:

1. **Scaled Dot-Product Attention:** Implement Q, K, V projection and attention computation
2. **Multi-Head Attention:** Combine multiple attention heads in parallel
3. **Positional Encoding:** Generate sinusoidal position embeddings
4. **Feed-Forward Network:** Simple 2-layer MLP per position
5. **Residual Connections:** Build residual blocks with layer norm

**For Banso Language Work:**
This architecture will enable us to train a language model that understands context in both English and Banso. The multi-head attention can learn different linguistic patterns—some heads might learn grammar, others learn semantic relationships.
