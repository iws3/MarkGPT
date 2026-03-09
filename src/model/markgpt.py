"""
MarkGPT Model Architecture
===========================
A clean, educational implementation of a GPT-style Transformer.
Every component is commented to explain not just WHAT it does but WHY.

This file is the heart of the MarkGPT project. By the time you reach Module 06,
you will understand every line here from first principles.

Architecture: Decoder-only Transformer (GPT-style)
Inspiration: Karpathy's nanoGPT, Vaswani et al. (2017)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarkGPTConfig:
    """
    Hyperparameters for MarkGPT.
    
    Think of this as the "blueprint" of the model. Before any computation 
    happens, we decide: how wide is each layer? How many layers deep?
    How many attention heads? These choices determine the model's 
    capacity — its ability to learn complex patterns from text.
    
    Preset configurations:
      - MarkGPT-Nano:  vocab=8000,  d=128,  heads=4,  layers=4   (~2M params)
      - MarkGPT-Small: vocab=8000,  d=256,  heads=8,  layers=6   (~10M params)  
      - MarkGPT-Base:  vocab=16000, d=512,  heads=8,  layers=12  (~85M params)
    """
    # Vocabulary: how many unique tokens can the model know?
    vocab_size: int = 8000
    
    # Context window: how many tokens can the model "see" at once?
    # The model generates the next token based on up to block_size previous tokens.
    block_size: int = 512
    
    # Model dimension (d_model): the size of every embedding vector.
    # Every token, every position, every hidden state lives in d_model-dimensional space.
    n_embd: int = 256
    
    # Number of Transformer blocks stacked on top of each other.
    # Depth = the model's ability to compose complex, hierarchical representations.
    n_layer: int = 6
    
    # Number of attention heads per layer.
    # Each head learns to attend to different types of relationships.
    # Must divide n_embd evenly. E.g., 8 heads with n_embd=256 → 32 dims per head.
    n_head: int = 8
    
    # Dropout rate: fraction of activations randomly zeroed during training.
    # Prevents overfitting. Set to 0.0 for inference (no dropout during generation).
    dropout: float = 0.1
    
    # Bias: whether to add a bias term in Linear layers and LayerNorm.
    # GPT-2 uses bias=True; more recent models often use bias=False.
    bias: bool = True
    
    @property
    def head_size(self) -> int:
        """Dimension of each attention head. Must be an integer."""
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        return self.n_embd // self.n_head
    
    def parameter_count(self) -> int:
        """Estimate total parameters (useful for sanity-checking before training)."""
        # Embedding table
        emb = self.vocab_size * self.n_embd
        # Each Transformer block: QKV projections, output, FFN, two LayerNorms
        per_block = 4 * self.n_embd * self.n_embd + 2 * 4 * self.n_embd * self.n_embd
        # Final LayerNorm + output projection
        final = self.n_embd + self.n_embd * self.vocab_size
        return emb + self.n_layer * per_block + final


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL SELF-ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.
    
    The "self" in self-attention means: each token attends to ALL other tokens
    in the same sequence (including itself).
    
    The "causal" (or "masked") part means: token at position i can ONLY attend
    to positions 0, 1, ..., i. It cannot "look ahead" to future tokens.
    This is crucial for language modeling: when predicting the next word,
    you shouldn't be able to cheat by seeing what the next word already is.
    
    The "multi-head" part means: instead of computing attention once with
    n_embd dimensions, we split into n_head parallel attention computations,
    each with head_size dimensions. Each head can specialize in different
    relationship types (syntax, coreference, semantics, etc.)
    """
    
    def __init__(self, config: MarkGPTConfig):
        super().__init__()
        self.config = config
        
        # Single combined projection for Queries, Keys, and Values.
        # Output dimension is 3 * n_embd because we need Q, K, V each of size n_embd.
        # We project all three at once for efficiency.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection: after attending and concatenating heads, project back to n_embd.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # The causal mask: a lower-triangular matrix of 1s.
        # We register it as a buffer so it moves with the model (CPU ↔ GPU)
        # without being a trainable parameter.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through causal self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Output tensor of same shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.shape  # batch size, sequence length, embedding dimension
        
        # ── Step 1: Project input to Q, K, V ───────────────────────────────
        # Each of Q, K, V has shape (B, T, n_embd)
        # Think of Q (query) as "what am I looking for?"
        #          K (key)   as "what do I offer?"
        #          V (value) as "what information do I carry?"
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each: (B, T, C)
        
        # ── Step 2: Reshape for multi-head attention ────────────────────────
        # Split embedding dimension C into n_head heads, each of size head_size.
        # After reshape: (B, T, n_head, head_size) → transpose → (B, n_head, T, head_size)
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        
        # ── Step 3: Compute scaled dot-product attention ────────────────────
        # Attention score matrix: Q @ K^T, scaled by 1/sqrt(head_size)
        # Why scale? If head_size is large, dot products grow large, pushing
        # softmax into saturated regions with near-zero gradients.
        # Shape: (B, n_head, T, T) — for each head and each position,
        # a score for how much to attend to every other position.
        scale = 1.0 / math.sqrt(head_size)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T)
        
        # ── Step 4: Apply the causal mask ───────────────────────────────────
        # Wherever the mask is 0 (upper triangle = "future" positions),
        # we set the attention score to -infinity, so softmax will give 0 weight.
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # ── Step 5: Softmax + dropout ────────────────────────────────────────
        # Softmax converts scores to probabilities (summing to 1 over each row).
        # After this, each row of attn tells us: "for token i, how much do I
        # attend to each of tokens 0..i?"
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # ── Step 6: Weighted sum of Values ──────────────────────────────────
        # The final output for each token is a weighted average of all value vectors,
        # weighted by the attention probabilities.
        out = attn @ v  # (B, nh, T, hs)
        
        # ── Step 7: Concatenate heads and project ───────────────────────────
        # Reshape back to (B, T, C) and apply the output projection.
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        
        return out


# ─────────────────────────────────────────────────────────────────────────────
# FEED-FORWARD NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    The feed-forward sublayer in each Transformer block.
    
    After attention (which mixes information across positions),
    the FFN processes each position independently, applying the same
    transformation to every token's representation.
    
    Structure: Linear → GELU → Linear → Dropout
    
    The inner dimension is 4 * n_embd. This 4x expansion factor
    was used in the original Transformer paper and has held up well empirically.
    The expanded dimension is where the model stores "factual" associations —
    this is why FFNs are sometimes called the "memory" of the Transformer.
    """
    
    def __init__(self, config: MarkGPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            # Expand to 4x dimension
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            # GELU activation: a smooth, non-zero-centered alternative to ReLU
            # GPT models use GELU; the exact choice has modest but measurable effects.
            nn.GELU(),
            # Project back down to n_embd
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMER BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    A single Transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual.
    
    Two design choices here worth understanding:
    
    1. Pre-norm vs Post-norm: Original Transformer uses post-norm (LayerNorm after residual).
       We use pre-norm (LayerNorm before the sublayer), which GPT-2 found more stable for training.
    
    2. Residual connections: x = x + sublayer(LayerNorm(x))
       These "skip connections" allow gradients to flow directly from the output all the way
       back to early layers, solving the vanishing gradient problem in deep networks.
       Without residuals, training 12+ layer networks is nearly impossible.
    """
    
    def __init__(self, config: MarkGPTConfig):
        super().__init__()
        # Layer normalization normalizes the feature dimension (not batch).
        # This stabilizes training by ensuring activations don't grow unbounded.
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = FeedForward(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sublayer with residual connection
        x = x + self.attn(self.ln_1(x))
        # Feed-forward sublayer with residual connection
        x = x + self.ffn(self.ln_2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# THE FULL MARKGPT MODEL
# ─────────────────────────────────────────────────────────────────────────────

class MarkGPT(nn.Module):
    """
    MarkGPT: A GPT-style language model for Biblical and Banso vernacular text.
    
    Architecture summary:
      1. Token embedding: convert token IDs to dense vectors
      2. Position embedding: add positional information
      3. N Transformer blocks: process and contextualize representations
      4. Final LayerNorm: normalize before output projection  
      5. Language model head: project to vocabulary logits
    
    The model predicts, for each input token, a probability distribution
    over all tokens in the vocabulary — what token is most likely to come next.
    Training minimizes the cross-entropy between these predictions and the
    actual next tokens in the training text.
    """
    
    def __init__(self, config: MarkGPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict({
            # Token embedding: each of vocab_size tokens gets a learned n_embd-dimensional vector
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            # Position embedding: each of block_size positions gets a learned n_embd-dimensional vector
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            # The stack of Transformer blocks — this is where learning happens
            'h': nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            # Final layer normalization before output
            'ln_f': nn.LayerNorm(config.n_embd, bias=config.bias),
        })
        
        # Language model head: project from n_embd to vocab_size (unnormalized logits)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: the token embedding and the lm_head share weights.
        # This is standard practice (Press & Wolf, 2017) and reduces parameter count
        # while often improving performance — the intuition is that the same
        # vector that represents a token as input should score it well as output.
        self.transformer['wte'].weight = self.lm_head.weight
        
        # Initialize weights following GPT-2's scheme
        self.apply(self._init_weights)
        
        # Special scaled initialization for residual projections (GPT-2 paper, Appendix B)
        # This keeps the residual stream's variance roughly constant with depth.
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        print(f"MarkGPT initialized: {self.count_parameters()/1e6:.2f}M parameters")
    
    def _init_weights(self, module: nn.Module):
        """Weight initialization following GPT-2: normal(0, 0.02) for Linear/Embedding."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ):
        """
        Forward pass.
        
        Args:
            idx:     Token indices, shape (batch_size, seq_len). Values in [0, vocab_size).
            targets: Target token indices (same shape). If provided, compute loss.
        
        Returns:
            logits: Unnormalized scores for each token in vocabulary. Shape: (B, T, vocab_size)
            loss:   Cross-entropy loss if targets provided, else None.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        
        device = idx.device
        
        # ── Step 1: Embeddings ───────────────────────────────────────────────
        # Create position indices [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)
        
        # Token embeddings + positional embeddings, added element-wise.
        # This is how the model knows both WHAT each token is and WHERE it is.
        tok_emb = self.transformer['wte'](idx)   # (B, T, n_embd)
        pos_emb = self.transformer['wpe'](pos)   # (T, n_embd) — broadcast over batch
        x = self.transformer['drop'](tok_emb + pos_emb)  # (B, T, n_embd)
        
        # ── Step 2: Pass through all Transformer blocks ─────────────────────
        for block in self.transformer['h']:
            x = block(x)  # (B, T, n_embd)
        
        # ── Step 3: Final layer norm ─────────────────────────────────────────
        x = self.transformer['ln_f'](x)  # (B, T, n_embd)
        
        # ── Step 4: Language model head → vocabulary logits ─────────────────
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # ── Step 5: Compute loss if targets provided ─────────────────────────
        loss = None
        if targets is not None:
            # Flatten to (B*T, vocab_size) and (B*T,) for cross-entropy computation
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # -1 is used as padding/ignore token
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Given a conditioning sequence (idx), generate max_new_tokens new tokens
        by repeatedly predicting the next token and appending it to the sequence.
        
        Args:
            idx:            Conditioning token indices, shape (1, T)
            max_new_tokens: Number of new tokens to generate
            temperature:    Controls randomness. Lower = more deterministic, more conservative.
                           Higher = more random, more creative. Range: (0, ∞). Typical: 0.7–1.2.
            top_k:          If set, only sample from the top-k most likely tokens.
                           Reduces incoherent outputs. Typical: 40–100.
            top_p:          If set, only sample from the smallest set of tokens whose
                           cumulative probability exceeds top_p (nucleus sampling).
                           Typical: 0.9–0.95.
        
        Returns:
            Token indices including the new generated tokens. Shape: (1, T + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to block_size if we've exceeded the context window
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                         else idx[:, -self.config.block_size:]
            
            # Forward pass to get logits for the last position
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (1, vocab_size) — only last token matters
            
            # Apply temperature scaling
            # Temperature = 1.0: unchanged distribution
            # Temperature → 0: becomes greedy (argmax)
            # Temperature > 1: flatter distribution (more random)
            logits = logits / temperature
            
            # Apply top-k filtering: zero out all but top-k tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_indices_to_remove] = float('-inf')
                logits = torch.scatter(logits, 1, sorted_indices, sorted_logits)
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)
            
            # Append to growing sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTIONS — PRE-DEFINED CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

def markgpt_nano(vocab_size: int = 8000) -> MarkGPT:
    """MarkGPT-Nano: ~2M params. Runs on CPU. Good for early modules."""
    config = MarkGPTConfig(
        vocab_size=vocab_size, block_size=256,
        n_embd=128, n_layer=4, n_head=4, dropout=0.1
    )
    return MarkGPT(config)


def markgpt_small(vocab_size: int = 8000) -> MarkGPT:
    """MarkGPT-Small: ~10M params. Needs GPU. The main training target."""
    config = MarkGPTConfig(
        vocab_size=vocab_size, block_size=512,
        n_embd=256, n_layer=6, n_head=8, dropout=0.1
    )
    return MarkGPT(config)


def markgpt_base(vocab_size: int = 16000) -> MarkGPT:
    """MarkGPT-Base: ~85M params. Needs 16GB+ GPU. For advanced users."""
    config = MarkGPTConfig(
        vocab_size=vocab_size, block_size=1024,
        n_embd=512, n_layer=12, n_head=8, dropout=0.1
    )
    return MarkGPT(config)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MarkGPT Architecture Smoke Test")
    print("=" * 60)
    
    # Test all three sizes
    for name, model_fn in [("Nano", markgpt_nano), ("Small", markgpt_small)]:
        model = model_fn()
        
        # Create a small fake batch: 2 sequences, each 64 tokens long
        batch_size, seq_len = 2, 64
        fake_input = torch.randint(0, 8000, (batch_size, seq_len))
        fake_targets = torch.randint(0, 8000, (batch_size, seq_len))
        
        # Forward pass with loss computation
        logits, loss = model(fake_input, fake_targets)
        
        print(f"\nMarkGPT-{name}:")
        print(f"  Parameters:   {model.count_parameters()/1e6:.2f}M")
        print(f"  Input shape:  {fake_input.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Loss:         {loss.item():.4f}")
        print(f"  Expected loss (random init ≈ log(vocab_size)): {math.log(8000):.4f}")
        
        # Test generation
        seed = torch.zeros((1, 1), dtype=torch.long)
        generated = model.generate(seed, max_new_tokens=20, temperature=0.8, top_k=40)
        print(f"  Generated sequence length: {generated.shape[1]}")
    
    print("\n✅ All smoke tests passed. MarkGPT is ready to train.")
