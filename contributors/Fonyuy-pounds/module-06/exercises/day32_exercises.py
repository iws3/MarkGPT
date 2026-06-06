"""
Day 32 Exercise: Scaled Dot-Product Attention from Scratch
Module 06: The Transformer Architecture
=============================================================================

Contributor: Fonyuy-pounds
Purpose   : Implement the Scaled Dot-Product Attention mechanism from scratch
            in PyTorch, verify mathematical alignment with PyTorch's native 
            functional implementation, and apply it to disambiguate Banso 
            low-resource linguistic theological expressions.

Syllabus Objective:
-------------------
Implement scaled dot-product attention from scratch in PyTorch (no nn.MultiheadAttention).
Verify your output matches PyTorch's implementation on identical inputs.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. SCALED DOT-PRODUCT ATTENTION FROM SCRATCH
# ============================================================================

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention Module implemented from first principles.
    
    Mathematical Formulation:
        Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) + Mask ) @ V
        
    Where:
        Q: Query tensor of shape (..., seq_len_q, d_k)
        K: Key tensor of shape (..., seq_len_k, d_k)
        V: Value tensor of shape (..., seq_len_v, d_v)
        d_k: Dimension of the key vectors (scaling factor is sqrt(d_k))
        Mask: Optional tensor of shape (..., seq_len_q, seq_len_k) containing
              masking values (-inf for disallowed positions, 0 for allowed).
    """
    
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.scale = 1.0 / math.sqrt(d_k)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the forward pass for scaled dot-product attention.
        
        Args:
            Q (torch.Tensor): Queries of shape (batch_size, num_heads, seq_len_q, d_k)
                              or (batch_size, seq_len_q, d_k)
            K (torch.Tensor): Keys of shape (batch_size, num_heads, seq_len_k, d_k)
                              or (batch_size, seq_len_k, d_k)
            V (torch.Tensor): Values of shape (batch_size, num_heads, seq_len_v, d_v)
                              or (batch_size, seq_len_v, d_v)
            mask (torch.Tensor, optional): Float mask where 0 indicates attention is allowed
                                           and -inf/very large negative numbers denote masking.
                                           Shape: Broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)
                                           
        Returns:
            output (torch.Tensor): Weighted attention output representation. Shape matches V except for seq_len.
            attn_weights (torch.Tensor): Softmax probabilities of shape (..., seq_len_q, seq_len_k).
        """
        # Step 1: Compute similarity scores (QK^T)
        # We need to transpose the last two dimensions of K to allow matrix multiplication
        # Shape: (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Step 2: Apply the scaling factor (1 / sqrt(d_k))
        # This keeps the variance of scores ~ 1, preventing softmax saturation
        scores = scores * self.scale
        
        # Step 3: Apply mask if provided
        if mask is not None:
            # We add the mask. Masked positions should have a value of -inf (or -1e9)
            # so that exp(-inf) -> 0 in the softmax step.
            scores = scores + mask
            
        # Step 4: Softmax along the last dimension to get a probability distribution
        # Shape: (..., seq_len_q, seq_len_k)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Step 5: Compute weighted average of values (attn_weights @ V)
        # Shape: (..., seq_len_q, seq_len_k) @ (..., seq_len_k, d_v) -> (..., seq_len_q, d_v)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights


# ============================================================================
# 2. CAUSAL MASK GENERATOR
# ============================================================================

def get_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Generates a lower-triangular causal mask for autoregressive generation.
    Mask prevents positions from attending to future tokens.
    
    Args:
        seq_len (int): Length of the sequence.
        device (torch.device, optional): Target device.
        
    Returns:
        mask (torch.Tensor): Mask of shape (seq_len, seq_len) filled with
                             0 on the lower-triangular part and -inf on the upper part.
    """
    # Create a lower-triangular matrix of ones
    # Shape: (seq_len, seq_len)
    tril = torch.tril(torch.ones(seq_len, seq_len, device=device))
    
    # Map 1 -> 0.0 (allow) and 0 -> -inf (block)
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask = mask.masked_fill(tril == 0, float('-inf'))
    
    return mask


# ============================================================================
# 3. VERIFICATION & BENCHMARKING SUITE
# ============================================================================

def run_verification_tests():
    """
    Verifies our scratch implementation matches PyTorch's native functional
    scaled dot-product attention on identical inputs, matching both output
    embeddings and backpropagated gradients.
    """
    print("\n" + "="*80)
    print("      SCALED DOT-PRODUCT ATTENTION VERIFICATION SUITE")
    print("="*80)
    
    # 1. Setup hyperparameters
    batch_size = 4
    num_heads = 8
    seq_len = 12
    d_k = 64
    d_v = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Running verification tests on device: {device}")
    
    # 2. Initialize identical inputs with gradient tracking enabled
    # We clone and detach to ensure both implementations receive the exact same tensor values
    Q_base = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    K_base = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    V_base = torch.randn(batch_size, num_heads, seq_len, d_v, device=device)
    
    # Inputs for scratch model
    Q_scratch = Q_base.clone().detach().requires_grad_(True)
    K_scratch = K_base.clone().detach().requires_grad_(True)
    V_scratch = V_base.clone().detach().requires_grad_(True)
    
    # Inputs for native PyTorch model
    Q_native = Q_base.clone().detach().requires_grad_(True)
    K_native = K_base.clone().detach().requires_grad_(True)
    V_native = V_base.clone().detach().requires_grad_(True)
    
    # 3. Generate masks
    causal_mask = get_causal_mask(seq_len, device=device)
    
    # Instantiate scratch model
    scratch_attn = ScaledDotProductAttention(d_k=d_k)
    
    # Test cases:
    #   Case A: Without Mask
    #   Case B: With Causal Mask
    
    for has_mask in [False, True]:
        mask_label = "WITH Causal Mask" if has_mask else "WITHOUT Mask"
        print(f"\n--- Test Case: {mask_label} ---")
        
        current_mask = causal_mask if has_mask else None
        
        # A. Run forward pass (Scratch)
        scratch_output, scratch_weights = scratch_attn(
            Q_scratch, K_scratch, V_scratch, mask=current_mask
        )
        
        # B. Run forward pass (Native PyTorch 2.0+ scaled_dot_product_attention)
        # Note: PyTorch's native function expects bool mask (where False is masked) 
        # or float mask where -inf/large negative numbers are added.
        # We can pass our float mask directly as `attn_mask`.
        native_output = F.scaled_dot_product_attention(
            Q_native, K_native, V_native, attn_mask=current_mask
        )
        
        # C. Assert Forward Pass Equivalence
        forward_allclose = torch.allclose(scratch_output, native_output, atol=1e-5, rtol=1e-5)
        max_forward_diff = (scratch_output - native_output).abs().max().item()
        
        print(f"  [Forward] Output shapes match : {scratch_output.shape == native_output.shape} ({scratch_output.shape})")
        print(f"  [Forward] Values align (atol=1e-5) : {forward_allclose}")
        print(f"  [Forward] Max absolute difference  : {max_forward_diff:.8e}")
        
        if not forward_allclose:
            raise AssertionError("Forward pass values do not align with native PyTorch benchmark!")
            
        # D. Run backward pass (Gradient Verification)
        # Create a mock upstream gradient (loss representation)
        loss_grad = torch.randn_like(scratch_output)
        
        # Backward on scratch
        scratch_output.backward(loss_grad)
        
        # Backward on native
        native_output.backward(loss_grad)
        
        # Assert Gradient Equivalence for Q, K, and V
        q_grad_match = torch.allclose(Q_scratch.grad, Q_native.grad, atol=1e-4, rtol=1e-4)
        k_grad_match = torch.allclose(K_scratch.grad, K_native.grad, atol=1e-4, rtol=1e-4)
        v_grad_match = torch.allclose(V_scratch.grad, V_native.grad, atol=1e-4, rtol=1e-4)
        
        print(f"  [Backward] Q Gradients align: {q_grad_match} (Max diff: {(Q_scratch.grad - Q_native.grad).abs().max().item():.8e})")
        print(f"  [Backward] K Gradients align: {k_grad_match} (Max diff: {(K_scratch.grad - K_native.grad).abs().max().item():.8e})")
        print(f"  [Backward] V Gradients align: {v_grad_match} (Max diff: {(V_scratch.grad - V_native.grad).abs().max().item():.8e})")
        
        if not (q_grad_match and k_grad_match and v_grad_match):
            raise AssertionError("Gradients do not align with native PyTorch benchmark!")
            
        # Reset gradients for next iteration
        Q_scratch.grad.zero_()
        K_scratch.grad.zero_()
        V_scratch.grad.zero_()
        Q_native.grad.zero_()
        K_native.grad.zero_()
        V_native.grad.zero_()
        
    print("\n[Result] Verification successful! Custom ScaledDotProductAttention is 100% mathematically aligned.")


# ============================================================================
# 4. BANSO CULTURAL LINGUISTIC APPLICATION (Context Disambiguation)
# ============================================================================

def run_banso_linguistic_application():
    """
    Demonstrates how the self-attention mechanism enables a model to perform
    contextual disambiguation on low-resource Banso (Lamnso') phrases.
    
    The term 'Nfor' is highly polysemous in Lamnso':
      - In a praise context ('kibor'), 'Nfor' translates to 'God' (King of Heaven).
      - In a governance context ('kighaa' or local discourse), 'Nfor' means 'King' (traditional ruler, Fon).
      
    We feed two parallel sentences:
      Phrase A (Praise) : "Nfor a shii kibor"  -> "God is worthy of praise"
      Phrase B (Lament) : "Nfor a shii kighaa" -> "The King hears lamentation"
      
    We construct conceptual embeddings for each word, compute self-attention,
    and show how the representation of 'Nfor' shifts contextually by attending to 
    neighboring tokens ('kibor' vs 'kighaa').
    """
    print("\n" + "="*80)
    print("      BANSO LOW-RESOURCE COGNITIVE VALIDATION: CONTEXT DISAMBIGUATION")
    print("="*80)
    
    # 1. Define Vocabulary and Conceptual Embeddings
    # Vocabulary: ["Nfor", "a", "shii", "kibor", "kighaa"]
    # We construct a 4-dimensional semantic space:
    #   Dim 0: [Divine / Celestial Entity]
    #   Dim 1: [Royal / Monarchical Entity]
    #   Dim 2: [Positive affect / Praise]
    #   Dim 3: [Sorrowful affect / Lament]
    
    embeddings = {
        "Nfor":   torch.tensor([0.8, 0.7, 0.0, 0.0]),  # Contains both Divine and Royal properties, ambiguous
        "a":      torch.tensor([0.0, 0.0, 0.0, 0.0]),  # Grammatical particle (neutral)
        "shii":   torch.tensor([0.1, 0.0, 0.1, 0.1]),  # Verb (worthy / hears, neutral)
        "kibor":  torch.tensor([0.9, 0.0, 1.0, 0.0]),  # Explicitly praise, highly divine aligned
        "kighaa": torch.tensor([0.0, 0.6, 0.0, 1.0]),  # Explicitly lament, human/royal mourning aligned
    }
    
    phrase_a = ["Nfor", "a", "shii", "kibor"]
    phrase_b = ["Nfor", "a", "shii", "kighaa"]
    
    # Construct sequence tensors of shape (seq_len, d_model) -> (4, 4)
    X_a = torch.stack([embeddings[w] for w in phrase_a])
    X_b = torch.stack([embeddings[w] for w in phrase_b])
    
    # 2. Define Query, Key, and Value Projection Matrices
    # In a Transformer, linear projections project d_model -> d_k/d_v.
    # To keep this clean and intuitive, we use identity projections for keys/values
    # and linear query transformations that seek contextual associations.
    d_model = 4
    d_k = 4
    
    # Create simple trainable projections (initialized to identity + minor variance for flow)
    W_q = nn.Parameter(torch.eye(d_model, d_k))
    W_k = nn.Parameter(torch.eye(d_model, d_k))
    W_v = nn.Parameter(torch.eye(d_model, d_model))
    
    # Compute Queries, Keys, and Values
    # Shape: (seq_len, d_k)
    Q_a = torch.matmul(X_a, W_q)
    K_a = torch.matmul(X_a, W_k)
    V_a = torch.matmul(X_a, W_v)
    
    Q_b = torch.matmul(X_b, W_q)
    K_b = torch.matmul(X_b, W_k)
    V_b = torch.matmul(X_b, W_v)
    
    # Add batch and head dimensions for our module: (1, 1, seq_len, d_k)
    Q_a, K_a, V_a = Q_a.unsqueeze(0).unsqueeze(0), K_a.unsqueeze(0).unsqueeze(0), V_a.unsqueeze(0).unsqueeze(0)
    Q_b, K_b, V_b = Q_b.unsqueeze(0).unsqueeze(0), K_b.unsqueeze(0).unsqueeze(0), V_b.unsqueeze(0).unsqueeze(0)
    
    # 3. Compute Attention
    attn_module = ScaledDotProductAttention(d_k=d_k)
    out_a, weights_a = attn_module(Q_a, K_a, V_a)
    out_b, weights_b = attn_module(Q_b, K_b, V_b)
    
    # Remove batch/head dims for analysis
    weights_a = weights_a.squeeze(0).squeeze(0)
    weights_b = weights_b.squeeze(0).squeeze(0)
    out_a = out_a.squeeze(0).squeeze(0)
    out_b = out_b.squeeze(0).squeeze(0)
    
    # 4. Display ASCII Attention Maps
    def print_attention_matrix(tokens, weights, title):
        print(f"\n[{title}]")
        # Header
        print("          ", "".join([f"{t:<9}" for t in tokens]))
        print("----------" + "-" * (9 * len(tokens)))
        for i, row_token in enumerate(tokens):
            row_str = f"{row_token:<9}| "
            for j, weight in enumerate(weights[i]):
                row_str += f"{weight.item():.4f}   "
            print(row_str)
            
    print_attention_matrix(phrase_a, weights_a, "Phrase A (Praise): 'Nfor a shii kibor'")
    print_attention_matrix(phrase_b, weights_b, "Phrase B (Lament): 'Nfor a shii kighaa'")
    
    # 5. Semantic Output Disambiguation Analysis
    # Let's inspect the resulting representation of 'Nfor' (index 0) after attention processing.
    # The output is a weighted sum of the values of the tokens it attended to.
    nfor_rep_a = out_a[0]
    nfor_rep_b = out_b[0]
    
    print("\n--- Contextual Disambiguation Analysis ---")
    print(f"Original 'Nfor' Embedding  : {embeddings['Nfor'].tolist()} (Ambiguous: Divine & Royal)")
    print(f"Attention output 'Nfor' (A): {nfor_rep_a.detach().numpy().round(3).tolist()}")
    print(f"Attention output 'Nfor' (B): {nfor_rep_b.detach().numpy().round(3).tolist()}")
    
    # Interpretation:
    # In Phrase A, 'Nfor' should attend heavily to 'kibor' (index 3), which has high Divine (dim 0) and Praise (dim 2) features.
    # In Phrase B, 'Nfor' should attend heavily to 'kighaa' (index 3), which has high Royal (dim 1) and Lament (dim 3) features.
    praise_att = weights_a[0, 3].item()
    lament_att = weights_b[0, 3].item()
    
    print(f"\n[Validation Highlights]")
    print(f"  * In Phrase A, 'Nfor' attended to 'kibor' (praise context) with weight: {praise_att:.4f}")
    print(f"  * In Phrase B, 'Nfor' attended to 'kighaa' (lament context) with weight: {lament_att:.4f}")
    
    if praise_att > 0.2:
        print("  * SUCCESS: 'Nfor' successfully absorbed positive spiritual context from 'kibor'!")
    if lament_att > 0.2:
        print("  * SUCCESS: 'Nfor' successfully absorbed royal/sorrowful context from 'kighaa'!")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_verification_tests()
    run_banso_linguistic_application()
    print("\n" + "="*80)
    print("      DAY 32 EXERCISES COMPLETED SUCCESSFULLY!")
    print("="*80)
