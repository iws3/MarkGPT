"""
Day 31 Reading Guide & Annotation Exercises

Module: Module 06 - The Transformer Architecture
Day: 31/60

Topic: "Attention Is All You Need" (Vaswani et al., 2017)

This is primarily a READING and ANALYSIS day, not an implementation day.
The goal is to deeply understand the paper before we implement it on Days 32-33.

Main Paper: https://arxiv.org/abs/1706.03762

Instructions:
1. Read the paper section by section
2. Answer the questions below for each section
3. Annotate equations in your own words
4. Draw diagrams to visualize concepts
5. Compare RNNs to Transformers
"""

# ============================================================================
# SECTION-BY-SECTION READING GUIDE
# ============================================================================

class Day31ReadingGuide:
    """
    Structured guide for reading and annotating Vaswani et al. (2017).
    """
    
    PAPER_METADATA = {
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani, Noam Shazeer, Parmar, Jones, Gomez, Kaiser, Polosukhin",
        "conference": "NeurIPS 2017",
        "arxiv": "https://arxiv.org/abs/1706.03762",
        "significance": "MOST IMPORTANT PAPER IN THIS CURRICULUM - Foundation for GPT, BERT, Claude, ChatGPT",
        "pages": 15,
    }
    
    # ====================================================================
    # ABSTRACT (Write 2-3 sentences in your own words)
    # ====================================================================
    
    SECTION_ABSTRACT = {
        "title": "Abstract",
        "key_claims": [
            "RNNs are the dominant sequence model",
            "But they have fundamental limitations",
            "We propose to use ONLY attention, no recurrence",
            "This new architecture (Transformer) achieves state-of-the-art",
        ],
        "your_summary": "[YOUR SUMMARY IN OWN WORDS]",
        "question": "What is the main problem with RNNs that this paper solves?",
    }
    
    # ====================================================================
    # INTRODUCTION (Why this work matters)
    # ====================================================================
    
    SECTION_INTRO = {
        "title": "Introduction",
        "key_points": [
            "RNNs, LSTMs, GRUs became dominant for sequence modeling",
            "But they process sequences SEQUENTIALLY",
            "Sequential = impossible to parallelize on GPUs",
            "Factorization tricks and conditional computation helped, but sequential dependency remains",
            "This paper: use multi-headed self-attention instead",
        ],
        "questions": [
            "Why is sequential processing a problem for modern hardware?",
            "What is the 'sequential dependency' problem in RNNs?",
            "How many sequential steps does an RNN with 1000 tokens require?",
        ],
    }
    
    # ====================================================================
    # SECTION 3: MODEL ARCHITECTURE
    # ====================================================================
    
    SECTION_3_ARCHITECTURE = {
        "title": "Model Architecture",
        "subsections": {
            "3.1_ScaledDotProductAttention": {
                "title": "Scaled Dot-Product Attention",
                "formula": "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V",
                "components": {
                    "Q": "Query - 'What am I looking for?'",
                    "K": "Key - 'What am I?' (for all positions)",
                    "V": "Value - 'Here's my information'",
                    "d_k": "Dimension of keys (e.g., 64 in paper)",
                    "sqrt_scale": "Prevents QK^T from getting too large",
                },
                "your_annotations": "[ANNOTATE THIS FORMULA]",
                "questions": [
                    "Why divide by sqrt(d_k) specifically?",
                    "What happens if you don't scale?",
                    "Why softmax after the scaled dot product?",
                ],
            },
            "3.2_MultiHeadAttention": {
                "title": "Multi-Head Attention",
                "key_idea": "Instead of 1 attention head, use H=8 heads in parallel",
                "formula": "MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O",
                "where": "head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)",
                "intuition": "Each head learns to focus on different linguistic patterns",
                "your_annotations": "[EXPLAIN THIS FORMULA]",
                "questions": [
                    "Why 8 heads? What if we used 1 or 16?",
                    "Can different heads learn different things? (Yes!)",
                    "What does head specialization mean?",
                ],
            },
            "3.3_PositionwiseFeedForward": {
                "title": "Position-wise Feed-Forward Networks",
                "formula": "FFN(x) = max(0, x W_1 + b_1) W_2 + b_2",
                "key_points": [
                    "Simple 2-layer network applied to each position independently",
                    "First layer: 512 → 2048 (expand)",
                    "ReLU activation",
                    "Second layer: 2048 → 512 (contract back)",
                ],
                "your_annotations": "[WHY THIS ARCHITECTURE?]",
                "questions": [
                    "Why expand to 2048 and contract back?",
                    "Why is this applied per-position, not globally?",
                    "Can this learn position-specific transformations?",
                ],
            },
            "3.4_EmbeddingsPositionalEncoding": {
                "title": "Embeddings and Positional Encoding",
                "key_points": [
                    "Token embeddings: each word → d-dimensional vector",
                    "Positional encoding: each position → d-dimensional vector",
                    "ADD them together: combined = token_embedding + positional_encoding",
                ],
                "positional_formula": [
                    "PE(pos, 2i) = sin(pos / 10000^(2i/d))",
                    "PE(pos, 2i+1) = cos(pos / 10000^(2i/d))",
                ],
                "your_annotations": "[HOW DOES POSITIONAL ENCODING WORK?]",
                "questions": [
                    "Why sinusoidal instead of learned embeddings?",
                    "Why those specific frequencies?",
                    "Could we use learned positional embeddings instead? (Paper tested both)",
                ],
            },
        },
    }
    
    # ====================================================================
    # SECTION 4: WHY SELF-ATTENTION (Justification)
    # ====================================================================
    
    SECTION_4_JUSTIFICATION = {
        "title": "Why Self-Attention",
        "comparison_table": {
            "layer_type": ["Self-Attention", "Recurrent", "Convolutional"],
            "complexity_per_layer": ["O(n^2 d)", "O(n d^2)", "O(k n d^2)"],
            "sequential_operations": ["O(1)", "O(n)", "O(log_k n)"],
            "max_path_length": ["O(1)", "O(n)", "O(log_k n)"],
            "interpretability": ["High", "Low", "Medium"],
        },
        "key_insight": "Self-attention has O(1) max path length - any token can directly attend to any other",
        "your_analysis": "[INTERPRET THIS TABLE]",
    }
    
    # ====================================================================
    # SECTION 5: TRAINING (How to train Transformers)
    # ====================================================================
    
    SECTION_5_TRAINING = {
        "title": "Training",
        "key_components": {
            "optimizer": "Adam optimizer",
            "learning_rate_schedule": "Linearly increase for 4000 steps, then decay",
            "regularization": [
                "Dropout: 0.1 on attention weights and FFN",
                "Label smoothing: 0.1 on cross-entropy loss",
            ],
            "training_data": [
                "WMT 2014 English-German: 4.5M sentence pairs",
                "WMT 2014 English-French: 36M sentence pairs",
            ],
            "hardware": "8 NVIDIA P100 GPUs",
            "training_time": "12 hours for big model (base: 10 hours)",
        },
        "your_notes": "[TRAINING DETAILS FOR FUTURE REFERENCE]",
    }
    
    # ====================================================================
    # SECTION 6: RESULTS
    # ====================================================================
    
    SECTION_6_RESULTS = {
        "title": "Results & Analysis",
        "key_results": {
            "WMT_2014_EnDe": {
                "task": "English → German translation",
                "transformer_bleu": 28.4,
                "previous_sota_bleu": 25.2,
                "improvement": "+3.2 BLEU points",
            },
            "WMT_2014_EnFr": {
                "task": "English → French translation",
                "transformer_bleu": 41.8,
                "previous_sota_bleu": 39.2,
                "improvement": "+2.6 BLEU points",
            },
        },
        "interpretability_findings": [
            "Different attention heads specialize in different tasks",
            "Some heads track word relationships (e.g., pronouns → antecedents)",
            "Some heads focus on local structure (grammatical relationships)",
            "Visualizations show attention patterns are linguistically meaningful",
        ],
        "your_analysis": "[WHAT DO THESE RESULTS MEAN?]",
    }


# ============================================================================
# EXERCISES FOR TODAY
# ============================================================================

class Day31Exercises:
    """
    Exercises to deepen understanding of the paper.
    """
    
    @staticmethod
    def exercise_1_comparison_table():
        """
        Exercise 1: Compare RNNs, LSTMs, and Transformers
        
        Fill in this table with your understanding:
        """
        return """
        | Aspect | RNN | LSTM | Transformer |
        |--------|-----|------|-------------|
        | Processing | [Sequential] | [Sequential] | [Parallel] |
        | Max path length token 1 to token 100 | [100 steps] | [100 steps] | [1 step] |
        | Can parallelize on GPU? | No | No | Yes |
        | Easy to capture long-range dependencies? | No | Somewhat | Yes |
        | Number of parameters | [Your answer] | [Your answer] | [Your answer] |
        | Training time (100 tokens, modern GPU) | [Your answer] | [Your answer] | [Your answer] |
        """
    
    @staticmethod
    def exercise_2_attention_visualization():
        """
        Exercise 2: Visualize what attention looks like
        
        For the sentence: "The cat sat on the mat"
        
        Draw what a Query-Key attention matrix might look like.
        Each cell (i,j) = attention weight from position i to position j.
        
        Hint: "cat" should attend strongly to "sat" and "mat"
              "sat" should attend to "cat", "on", "mat"
              etc.
        """
        return """
        Positions:    the   cat   sat   on   the   mat
        
        the          [ ]    [ ]   [ ]   [ ]   [ ]   [ ]
        cat          [ ]    [ ]   [H]   [ ]   [ ]   [H]
        sat          [ ]    [H]   [ ]   [ ]   [ ]   [H]
        on           [ ]    [ ]   [H]   [ ]   [ ]   [H]
        the          [ ]    [ ]   [ ]   [ ]   [ ]   [ ]
        mat          [ ]    [M]   [M]   [ ]   [ ]   [ ]
        
        [H] = high attention (model focuses here)
        [M] = medium attention
        [ ] = low attention
        """
    
    @staticmethod
    def exercise_3_positional_encoding():
        """
        Exercise 3: Understand positional encoding
        
        For d=4 (4 dimensions), positions 0-2:
        
        PE(0, 0) = sin(0 / 10000^0/4) = sin(0) = 0
        PE(0, 1) = cos(0 / 10000^1/4) = cos(0) = 1
        PE(0, 2) = sin(0 / 10000^2/4) = sin(0) = 0
        PE(0, 3) = cos(0 / 10000^3/4) = cos(0) = 1
        PE(0) = [0, 1, 0, 1]
        
        PE(1, 0) = sin(1 / 10000^0/4) = sin(1) ≈ 0.84
        PE(1, 1) = cos(1 / 10000^0/4) = cos(1) ≈ 0.54
        PE(1, 2) = sin(1 / 10000^2/4) = sin(0.01) ≈ 0.01
        PE(1, 3) = cos(1 / 10000^2/4) = cos(0.01) ≈ 1.0
        PE(1) = [0.84, 0.54, 0.01, 1.0]
        
        Notice: Different positions have different encodings!
        Different dimensions oscillate at different frequencies.
        
        Question: Can a position-aware model learn positional patterns from these?
        Answer: Yes! The model learns to decode position from these signals.
        """
        pass
    
    @staticmethod
    def exercise_4_compute_attention():
        """
        Exercise 4: Manually compute scaled dot-product attention
        
        Simplified example: 2 positions, d_k = 2
        
        Query (Q) = [[1, 0],
                     [0, 1]]
        
        Key (K) = [[1, 0],
                   [1, 1]]
        
        Value (V) = [[2, 0],
                     [0, 3]]
        
        d_k = 2, so sqrt(d_k) = sqrt(2) ≈ 1.414
        
        Step 1: Compute QK^T
        QK^T = Q @ K.T = ...
        
        Step 2: Scale by 1/sqrt(d_k)
        scaled = QK^T / 1.414
        
        Step 3: Apply softmax (normalize rows to sum to 1)
        weights = softmax(scaled)
        
        Step 4: Multiply by V
        output = weights @ V
        """
        pass


# ============================================================================
# QUESTIONS TO PONDER
# ============================================================================

REFLECTION_QUESTIONS = """
After reading the paper, reflect on these questions:

1. RNNs have been the gold standard for sequence modeling for 10+ years.
   Why would researchers even TRY an approach with no recurrence?
   
   [Your answer: _____________]

2. The title is "Attention Is All You Need."
   What does this claim mean?
   Did they really need NOTHING besides attention?
   
   [Your answer: _____________]

3. Look at Figure 2 (Transformer architecture).
   Trace the path from input embedding to output probability.
   What happens at each stage?
   
   [Your answer: _____________]

4. The paper mentions "residual connections" and "layer normalization."
   Why are these crucial?
   What would happen without them?
   
   [Your answer: _____________]

5. For Banso language support:
   How could multi-head attention help learn Banso-specific grammar patterns?
   Could different heads specialize in:
   - Verb conjugations?
   - Noun classes?
   - Tone patterns?
   
   [Your answer: _____________]

6. Look at Table 3 (inference speed).
   Why is Transformer SLOWER than RNN at inference?
   When does Transformer speed advantage appear?
   
   [Your answer: _____________]

7. This paper won "Best Paper Award" at NeurIPS 2017.
   Why do you think this particular paper has had such massive impact?
   (Consider: timing, simplicity, empirical results, generalizability)
   
   [Your answer: _____________]
"""


# ============================================================================
# KEY EQUATIONS TO ANNOTATE
# ============================================================================

KEY_EQUATIONS = {
    "scaled_dot_product_attention": {
        "equation": "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V",
        "meaning_q": "What is each position querying for?",
        "meaning_k": "What does each position represent?",
        "meaning_v": "What information does each position hold?",
        "meaning_scaling": "Why divide by sqrt(d_k)?",
        "meaning_softmax": "Why softmax (not just normalized sum)?",
    },
    "multi_head_attention": {
        "equation": "MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O",
        "where": "head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)",
        "benefit": "Why not just use 1 big attention head?",
        "specialization": "What can different heads learn?",
    },
    "feed_forward": {
        "equation": "FFN(x) = max(0, xW_1 + b_1)W_2 + b_2",
        "dimensions": "Input d_model=512 → Hidden=2048 → Output d_model=512",
        "purpose": "What does expanding→contract do?",
    },
    "positional_encoding": {
        "even_dimensions": "PE(pos, 2i) = sin(pos / 10000^(2i/d))",
        "odd_dimensions": "PE(pos, 2i+1) = cos(pos / 10000^(2i/d))",
        "frequency": "Different dimensions encode different frequencies",
        "purpose": "How does model extract position from these signals?",
    },
}


if __name__ == "__main__":
    print("=" * 70)
    print("DAY 31 READING GUIDE: 'Attention Is All You Need'")
    print("=" * 70)
    print("\nThis is a READING and ANALYSIS day, not coding.")
    print("Your tasks:")
    print("1. Read Vaswani et al. (2017) carefully")
    print("2. Answer the exercises above")
    print("3. Annotate key equations in your journal")
    print("4. Reflect on the comparison between RNNs and Transformers")
    print("5. Complete the reflection questions")
    print("\nWhy today matters:")
    print("- This paper is the foundation of ALL modern LLMs")
    print("- GPT, BERT, Claude, ChatGPT all use this architecture")
    print("- MarkGPT uses this to understand Banso language")
    print("\nReady for Day 32 (Implementation) tomorrow!")
    print("=" * 70)
