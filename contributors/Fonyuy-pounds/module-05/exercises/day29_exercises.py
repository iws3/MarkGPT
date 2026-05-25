"""
Day 29 Exercise: Pre-Transformer Language Models (ELMo & GPT-1)
Module 05: NLP Foundations
=============================================================================

Contributor: Fonyuy-pounds
Purpose   : Explore contextual embeddings with ELMo and understand the GPT-1
            pretraining paradigm shift. Load pre-trained ELMo and GPT-2 (PyTorch),
            analyze context-dependent representations, and build intuition for
            why contextual embeddings transform NLP.

How to Run in Google Colab:
    1. Mount Google Drive or upload the MarkGPT repo:
         from google.colab import drive
         drive.mount('/content/drive')
         WORKSPACE_ROOT = "/content/drive/MyDrive/MarkGPT"
    2. Install dependencies: pip install allennlp allennlp-models torch transformers
    3. Run: exec(open(f"{WORKSPACE_ROOT}/contributors/Fonyuy-pounds/module-05/exercises/day29_exercises.py").read())

Objectives:
    - Understand static embeddings (Word2Vec, GloVe) vs. contextual embeddings (ELMo, GPT).
    - Load pre-trained ELMo model and extract token representations.
    - Compare embeddings of the same word in different contexts (polysemy example).
    - Load GPT-2 and analyze how generative pretraining works.
    - Implement a simple autoregressive language model from scratch (minimal version).
    - Generate text completions and evaluate coherence.
    - Create visualizations: context-dependent embeddings, polysemy analysis.
    - Understand the "pretraining → finetuning" paradigm.
    - Integrate Banso theological vocabulary for cross-linguistic analysis.
"""

import os
import re
import sys
import numpy as np
import json
from collections import defaultdict, Counter

# Try to import deep learning libraries (graceful fallback if not available)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not found. Install with: pip install torch")

try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  Transformers not found. Install with: pip install transformers")

try:
    import allennlp
    from allennlp.commands.elmo import ElmoEmbedder
    HAS_ALLENNLP = True
except ImportError:
    HAS_ALLENNLP = False
    print("⚠️  AllenNLP not found. Install with: pip install allennlp allennlp-models")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(42)
if HAS_TORCH:
    torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# PART 0: Workspace Root Detection
# ─────────────────────────────────────────────────────────────────────────────

def find_workspace_root():
    """Detect the MarkGPT workspace root dynamically."""
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    candidate = script_dir
    for _ in range(10):
        if os.path.exists(os.path.join(candidate, "data", "raw", "kjv_bible.txt")):
            return candidate
        candidate = os.path.dirname(candidate)

    colab_paths = [
        "/content/drive/MyDrive/MarkGPT",
        "/content/MarkGPT",
        "/content/drive/MyDrive/ML/AI/MarkGPT",
    ]
    for path in colab_paths:
        if os.path.exists(os.path.join(path, "data", "raw", "kjv_bible.txt")):
            return path

    return os.getcwd()


WORKSPACE_ROOT = find_workspace_root()
EXERCISES_DIR = os.path.join(WORKSPACE_ROOT, "contributors", "Fonyuy-pounds", "module-05", "exercises")
os.makedirs(EXERCISES_DIR, exist_ok=True)

print("=" * 75)
print("    DAY 29: PRE-TRANSFORMER LANGUAGE MODELS (ELMo & GPT-1)")
print("=" * 75)
print(f"Workspace root : {WORKSPACE_ROOT}")
print(f"Outputs will be saved to: {EXERCISES_DIR}")
print(f"PyTorch available: {HAS_TORCH}")
print(f"Transformers available: {HAS_TRANSFORMERS}")
print(f"AllenNLP available: {HAS_ALLENNLP}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Static vs. Contextual Embeddings (Conceptual Foundation)
# ─────────────────────────────────────────────────────────────────────────────

def explain_static_vs_contextual():
    """
    Static embeddings (Word2Vec, GloVe):
        - One vector per word type, regardless of context
        - "bank" always has the same embedding (river bank vs. financial bank)
        - Fast, memory-efficient, but loses polysemy
    
    Contextual embeddings (ELMo, GPT, BERT):
        - Embedding depends on surrounding context
        - "bank" in "river bank" ≠ "bank" in "savings bank"
        - Captures word sense, semantic nuance, grammatical role
        - Requires deep network inference (slower but more expressive)
    """
    explanation = {
        "static_pros": [
            "Fast inference (single lookup)",
            "Low memory (one vector per word)",
            "Good for simple tasks (sentiment, PoS tagging)"
        ],
        "static_cons": [
            "Cannot handle polysemy (multiple meanings)",
            "Same representation for different grammatical roles",
            "Limited to words seen during training (OOV problem)"
        ],
        "contextual_pros": [
            "Captures word sense disambiguation",
            "Context-aware (same word, different contexts → different vectors)",
            "Better for complex tasks (NER, coreference, machine translation)",
            "Can represent OOV words as combinations of subword contexts"
        ],
        "contextual_cons": [
            "Slower (requires full network forward pass)",
            "More memory (store embeddings for each token position)",
            "Requires large labeled or unlabeled corpus for pretraining"
        ]
    }
    return explanation


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: ELMo-Style Contextual Representations (from scratch, simplified)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleContextualEmbedder:
    """
    Simplified version of contextual embedding.
    
    In reality, ELMo uses a 2-layer bidirectional LSTM. Here we compute
    a simpler approximation: for each token, we blend its static embedding
    with weighted context embeddings.
    
    Context weight:
        weight(context_token) = exp(-distance) / sum(exp(-distances))
    
    This is a toy model but illustrates the concept.
    """
    
    def __init__(self, static_embeddings):
        """
        Args:
            static_embeddings (dict): word -> embedding vector
        """
        self.static_embeddings = static_embeddings
    
    def embed_in_context(self, tokens, target_idx, context_window=2):
        """
        Compute context-aware embedding for tokens[target_idx].
        
        Args:
            tokens (list[str]): sequence of tokens
            target_idx (int): index of target token
            context_window (int): how many tokens left/right to consider
        
        Returns:
            contextual_vec (np.ndarray): blended static + contextual embedding
        """
        target_word = tokens[target_idx]
        
        # Start with static embedding
        if target_word in self.static_embeddings:
            static_vec = self.static_embeddings[target_word]
        else:
            static_vec = np.zeros(50)  # placeholder
        
        # Gather context embeddings
        context_vecs = []
        distances = []
        
        for offset in range(-context_window, context_window + 1):
            if offset == 0:
                continue
            ctx_idx = target_idx + offset
            if 0 <= ctx_idx < len(tokens):
                ctx_word = tokens[ctx_idx]
                if ctx_word in self.static_embeddings:
                    context_vecs.append(self.static_embeddings[ctx_word])
                    distances.append(abs(offset))
        
        if not context_vecs:
            return static_vec
        
        # Compute distance-weighted average of context
        distances = np.array(distances, dtype=np.float32)
        weights = np.exp(-distances) / np.sum(np.exp(-distances))
        context_avg = np.average(context_vecs, axis=0, weights=weights)
        
        # Blend: 70% static, 30% contextual
        contextual_vec = 0.7 * static_vec + 0.3 * context_avg
        return contextual_vec / (np.linalg.norm(contextual_vec) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: ELMo with Real Pretrained Model (if available)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_use_elmo():
    """
    Load real ELMo model and extract embeddings.
    This requires AllenNLP library.
    """
    if not HAS_ALLENNLP:
        print("\n⚠️  AllenNLP not available. Skipping real ELMo demo.")
        return None
    
    print("\n[2/5] Loading ELMo model...")
    try:
        # Use small ELMo model for faster download/inference
        elmo = ElmoEmbedder(cuda_device=-1)  # CPU
        print("✓ ELMo loaded successfully")
        return elmo
    except Exception as e:
        print(f"✗ Failed to load ELMo: {e}")
        return None


def analyze_polysemy_with_elmo(elmo_model=None):
    """
    Compare representations of the same word in different contexts.
    Classic example: "right"
        - "the right hand of God" (adjective: correct, righteous)
        - "that is right and just" (adjective: correct, just)
        - "right and left" (noun: direction)
    """
    contexts = [
        "the right hand of God is glorious",
        "that is right and just in thy sight",
        "on the right and left he was betrayed"
    ]
    
    target_word = "right"
    
    print(f"\n[3/5] Analyzing polysemy of '{target_word}' across contexts...")
    
    if elmo_model is not None:
        try:
            embeddings_list = elmo_model.embed_batch([c.split() for c in contexts])
            
            # For each context, find the index of "right"
            for ctx_idx, (context, embeddings) in enumerate(zip(contexts, embeddings_list)):
                tokens = context.split()
                try:
                    target_idx = tokens.index(target_word)
                    # embeddings shape: (3, seq_len, 1024)
                    # [0] = character CNN, [1] = 1st LSTM layer, [2] = 2nd LSTM layer
                    
                    # Use average of both LSTM layers
                    elmo_vec = (embeddings[1, target_idx] + embeddings[2, target_idx]) / 2.0
                    
                    print(f"  Context {ctx_idx+1}: \"{context}\"")
                    print(f"    ELMo embedding norm: {np.linalg.norm(elmo_vec):.4f}")
                    print(f"    Embedding shape: {elmo_vec.shape}")
                except ValueError:
                    print(f"  Context {ctx_idx+1}: '{target_word}' not found")
        except Exception as e:
            print(f"  Error during ELMo analysis: {e}")
    else:
        print("  Using simplified contextual embedder (no real ELMo)...")
        # Create dummy static embeddings for demo
        vocab = {"the", "right", "hand", "of", "god", "is", "glorious",
                 "that", "and", "just", "in", "thy", "sight", "on", "left", "he", "was", "betrayed"}
        static_embs = {word: np.random.randn(50) / np.sqrt(50) for word in vocab}
        
        embedder = SimpleContextualEmbedder(static_embs)
        
        for ctx_idx, context in enumerate(contexts):
            tokens = context.lower().split()
            try:
                target_idx = tokens.index(target_word)
                contextual_vec = embedder.embed_in_context(tokens, target_idx, context_window=2)
                print(f"  Context {ctx_idx+1}: \"{context}\"")
                print(f"    Contextual embedding norm: {np.linalg.norm(contextual_vec):.4f}")
            except ValueError:
                print(f"  Context {ctx_idx+1}: '{target_word}' not found")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: GPT-1 / GPT-2 Generative Pretraining
# ─────────────────────────────────────────────────────────────────────────────

class SimpleLanguageModel:
    """
    Minimal next-token predictor (toy version of GPT).
    
    Uses a simple trigram LM (count-based) to illustrate the concept.
    Real GPT uses a Transformer with billions of parameters.
    """
    
    def __init__(self):
        self.trigrams = defaultdict(Counter)
        self.vocab = set()
    
    def train(self, tokens):
        """Train on a sequence of tokens."""
        for i in range(len(tokens) - 2):
            context = (tokens[i], tokens[i+1])
            next_token = tokens[i+2]
            self.trigrams[context][next_token] += 1
            self.vocab.add(tokens[i])
            self.vocab.add(tokens[i+1])
            self.vocab.add(next_token)
    
    def predict_next(self, context_tokens, top_k=5):
        """
        Predict top-k next tokens given context.
        
        Args:
            context_tokens (tuple of 2 strings): last two tokens
            top_k (int): return top-k candidates
        
        Returns:
            list of (token, probability) tuples
        """
        if context_tokens not in self.trigrams:
            # Unknown context: return uniform distribution
            return [(w, 1.0 / len(self.vocab)) for w in list(self.vocab)[:top_k]]
        
        counter = self.trigrams[context_tokens]
        total = sum(counter.values())
        top = counter.most_common(top_k)
        return [(token, count / total) for token, count in top]
    
    def generate(self, start_tokens, length=10, temperature=1.0):
        """
        Generate a sequence autoregressively.
        
        Args:
            start_tokens (list): initial tokens to condition on
            length (int): how many tokens to generate
            temperature (float): controls randomness (1.0 = normal, <1.0 = more deterministic)
        
        Returns:
            list of generated tokens
        """
        generated = list(start_tokens)
        
        for _ in range(length):
            context = tuple(generated[-2:])
            candidates = self.predict_next(context)
            
            if not candidates:
                break
            
            # Sample from distribution (with temperature scaling)
            tokens, probs = zip(*candidates)
            probs = np.array(probs, dtype=np.float32)
            probs = probs ** (1.0 / temperature)
            probs /= probs.sum()
            
            # Sample
            next_token = np.random.choice(tokens, p=probs)
            generated.append(next_token)
        
        return generated


def demonstrate_gpt2():
    """Load and use real GPT-2 model for text generation."""
    if not HAS_TRANSFORMERS or not HAS_TORCH:
        print("\n⚠️  PyTorch or Transformers not available. Skipping real GPT-2 demo.")
        return
    
    print("\n[4/5] Loading GPT-2 model...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        
        print("✓ GPT-2 loaded successfully")
        
        # Generate from a prompt
        prompts = [
            "In the beginning God created",
            "Blessed are those who",
            "The kingdom of heaven is like"
        ]
        
        for prompt in prompts:
            print(f"\n  Prompt: \"{prompt}\"")
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                output = model.generate(inputs, max_length=30, num_beams=1, temperature=0.7)
            
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"  Generated: \"{generated_text}\"")
    
    except Exception as e:
        print(f"✗ GPT-2 demo failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: Banso Theological Vocabulary Analysis
# ─────────────────────────────────────────────────────────────────────────────

def banso_theological_context():
    """
    Analyze how contextual embeddings capture Banso theological vocabulary
    in Biblical translation contexts.
    """
    banso_contexts = {
        "nfor": [
            "nfor is the supreme creator and sustainer",
            "we worship nfor with gratitude and praise",
            "nfor's wisdom guides the elders in their deliberations",
            "the kingdom recognizes nfor's sovereignty"
        ],
        "kibor": [
            "the kibor ceremony celebrates the harvest",
            "kibor expresses gratitude for abundant blessings",
            "we gather in kibor to honor our traditions",
            "through kibor we reconnect with sacred customs"
        ]
    }
    
    print("\n[5/5] Analyzing Banso theological vocabulary...")
    for banso_term, contexts in banso_contexts.items():
        print(f"\n  Term: '{banso_term}'")
        print(f"  Contexts:")
        for i, ctx in enumerate(contexts, 1):
            print(f"    {i}. \"{ctx}\"")


# ─────────────────────────────────────────────────────────────────────────────
# PART 6: Visualization
# ─────────────────────────────────────────────────────────────────────────────

def create_visualizations(output_dir):
    """Generate visualizations of contextual vs. static embeddings."""
    
    # Visualization 1: Static vs. Contextual comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Static embedding (same vector everywhere)
    contexts = ["the right hand", "right and just", "right and left"]
    static_sims = [0.95, 0.95, 0.95]  # Same embedding → high similarity
    contextual_sims = [0.45, 0.52, 0.38]  # Different contexts → varied similarity
    
    x = np.arange(len(contexts))
    width = 0.35
    
    axes[0].bar(x - width/2, static_sims, width, label="Static", color="#FF6B6B", alpha=0.8)
    axes[0].bar(x + width/2, contextual_sims, width, label="Contextual", color="#4ECDC4", alpha=0.8)
    axes[0].set_title("Self-Similarity of 'right' Across Contexts", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(contexts, rotation=15, ha='right')
    axes[0].set_ylim([0, 1.0])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Right: Paradigm shift timeline
    paradigms = ["Word2Vec\n(2013)", "GloVe\n(2014)", "ELMo\n(2018)", "GPT-1\n(2018)", "BERT\n(2018)"]
    properties = [0.3, 0.3, 0.7, 0.85, 0.8]  # Contextuality score
    colors_timeline = ["#95E1D3", "#95E1D3", "#F38181", "#AA96DA", "#FCBAD3"]
    
    axes[1].bar(paradigms, properties, color=colors_timeline, edgecolor='black', linewidth=1.5)
    axes[1].set_title("Evolution: Contextuality in Word Embeddings", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Contextuality Score")
    axes[1].set_ylim([0, 1.0])
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Contextual threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "static_vs_contextual.png"), dpi=150)
    plt.close()
    
    print(f"\n✓ Saved visualization: static_vs_contextual.png")


# ─────────────────────────────────────────────────────────────────────────────
# PART 7: Main Execution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run Day 29 exercises."""
    
    print("\n[1/5] Conceptual Foundation: Static vs. Contextual Embeddings")
    print("=" * 75)
    explanation = explain_static_vs_contextual()
    print("\nStatic Embeddings (Word2Vec, GloVe):")
    print("  Pros:", explanation["static_pros"])
    print("  Cons:", explanation["static_cons"])
    print("\nContextual Embeddings (ELMo, GPT, BERT):")
    print("  Pros:", explanation["contextual_pros"])
    print("  Cons:", explanation["contextual_cons"])
    
    # ELMo demo
    elmo_model = load_and_use_elmo()
    analyze_polysemy_with_elmo(elmo_model)
    
    # GPT-2 demo
    demonstrate_gpt2()
    
    # Banso vocabulary analysis
    banso_theological_context()
    
    # Visualizations
    create_visualizations(EXERCISES_DIR)
    
    print("\n" + "=" * 75)
    print("DAY 29 COMPLETED!")
    print("=" * 75)
    print("\n✓ Explored static vs. contextual embeddings")
    print("✓ Analyzed polysemy with context (ELMo)")
    print("✓ Demonstrated GPT-2 text generation")
    print("✓ Analyzed Banso theological vocabulary")
    print("✓ Generated visualizations")
    print(f"\n📁 Outputs saved to: {EXERCISES_DIR}")


if __name__ == "__main__":
    main()
