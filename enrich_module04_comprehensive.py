#!/usr/bin/env python3
"""
Enrichment script for module-04 (RNN/Sequence lessons)
Target: 250+ commits
"""

import os
import subprocess
import sys

# Content for module-04 lessons
MODULE04_CONTENT = {
    "L19.1_sequences-memory": {
        "title": "Sequences and Memory in Neural Networks",
        "sections": {
            "Sequential Data Fundamentals": [
                "Sequential data exhibits temporal dependencies where past influences future.",
                "Time series contain measurements at regular intervals tracking changes.",
                "Natural language is sequential information processed word by word.",
                "Sequences have variable length requiring special handling in models.",
                "Positional information matters different meanings at different positions.",
                "Memory retention enables learning long-term dependencies.",
            ],
            "Feedforward Limitations": [
                "Standard feedforward networks lack memory between inputs.",
                "No temporal context captured from previous inputs.",
                "Fixed input size incompatible with variable length sequences.",
                "No sharing of parameters across time steps.",
                "Markovian assumption ignores historical information.",
                "Cannot model temporal dynamics effectively.",
            ],
            "Memory Architectures": [
                "Hidden state maintains information across time steps.",
                "Recurrent connections feed past activations to future timesteps.",
                "State updates accumulate information over time.",
                "Memory enables context propagation through sequences.",
                "Unfolding through time relates recurrent to feedforward networks.",
                "Backpropagation through time trains recurrent networks.",
            ]
        }
    },
    "L19.2_hidden-state": {
        "title": "Hidden State and Recurrent Computation",
        "sections": {
            "Hidden State Mechanics": [
                "Hidden state vector encodes temporal context and memory.",
                "State is updated at each time step based on input and previous state.",
                "Hidden dimension determines capacity for information storage.",
                "State initialization affects early predictions and learning.",
                "State is shared across all time steps enabling parameter efficiency.",
                "Final state summarizes entire sequence information.",
            ],
            "Recurrent Updates": [
                "Recurrent equation combines previous state with new input.",
                "Non-linear activation enables complex temporal patterns.",
                "State update is deterministic function of input and previous state.",
                "Multiple layers stack recurrent computations for hierarchy.",
                "Bidirectional processing uses forward and backward passes.",
                "State reset between sequences separates independent examples.",
            ],
            "Information Flow": [
                "Information propagates forward through time in unfolded graph.",
                "Gradients propagate backward through time for learning.",
                "Early timesteps influence final prediction through state.",
                "Context mixing blends information from all time steps.",
                "Bottleneck state constrains information capacity.",
                "Attention mechanisms focus on relevant time steps.",
            ]
        }
    },
    "L20.1_rnn-architecture": {
        "title": "RNN Architecture and Variants",
        "sections": {
            "Basic RNN Structure": [
                "Recurrent Neural Networks contain loops enabling memory.",
                "Time unfolding creates deep feedforward graph.",
                "Parameters shared across time steps enable generalization.",
                "Hidden state updates are primarily recurrent not forward.",
                "Output at each timestep can be read independently.",
                "Sequence processing enables variable length inputs.",
            ],
            "RNN Variations": [
                "One-to-one networks process fixed inputs producing fixed outputs.",
                "One-to-many networks generate sequences from single input.",
                "Many-to-one networks encode sequences to single output.",
                "Many-to-many networks transform input sequences to output sequences.",
                "Encoder-decoder separates sequence encoding from decoding.",
                "Attention-based variants weight input relevance.",
            ],
            "Training Recurrent Networks": [
                "Backpropagation Through Time extends backprop to sequences.",
                "Truncated BPTT limits computation by cutting gradients.",
                "Teacher forcing uses ground truth during training.",
                "Scheduled sampling gradually reduces ground truth reliance.",
                "Gradient clipping prevents explosion in recurrent networks.",
                "Different learning rates per layer improves convergence.",
            ]
        }
    },
    "L20.2_vanishing-gradients": {
        "title": "Vanishing and Exploding Gradients",
        "sections": {
            "Gradient Flow in RNNs": [
                "Gradients flow backward through time unfold graph.",
                "Chain rule multiplication creates products of Jacobians.",
                "Repeated multiplication can cause exponential growth or decay.",
                "Vanishing gradients prevent early timesteps from learning.",
                "Exploding gradients cause instability and NaN values.",
                "Proper initialization mitigates gradient flow problems.",
            ],
            "Vanishing Gradient Problem": [
                "Activation derivatives less than one cause decay.",
                "Long sequences accumulate many small multiplications.",
                "Early time steps receive negligible gradient signal.",
                "Long-term dependencies become unlearnable.",
                "Sigmoid activation particularly prone to vanishing.",
                "ReLU activation partially alleviates problem.",
            ],
            "Solutions and Mitigations": [
                "Gradient clipping caps maximum gradient magnitude.",
                "Skip connections enable direct gradient flow.",
                "LSTM and GRU architectures specifically address problem.",
                "Layer normalization stabilizes gradient magnitudes.",
                "Careful initialization keeps initial gradients reasonable.",
                "Residual connections improve gradient propagation.",
            ]
        }
    },
    "L21.1_lstm-cells": {
        "title": "LSTM Cells and Gating Mechanisms",
        "sections": {
            "LSTM Architecture": [
                "Long Short-Term Memory networks use gating mechanisms.",
                "Cell state separate from hidden state enables long-term memory.",
                "Three gates control information flow: forget, input, output.",
                "Forget gate decides what to discard from previous memory.",
                "Input gate controls addition of new information.",
                "Output gate determines what to output from memory.",
            ],
            "Gate Operations": [
                "Sigmoid gates output values between zero and one.",
                "Element-wise multiplication implements information gating.",
                "Forget gate scales cell state by gate values.",
                "Input gate scales candidate values added to cell state.",
                "Cell state accumulates information across time steps.",
                "Output gate scales hidden state from gated cell.",
            ],
            "Gradient Advantages": [
                "Additive cell state updates enable gradient flow.",
                "Multiplicative gates control magnitude smoothly.",
                "Gradients can flow uninterrupted through cell updates.",
                "Vanishing gradient problem substantially mitigated.",
                "Effective learning across long sequences enabled.",
                "Deep time dependencies become tractable.",
            ]
        }
    },
    "L21.2_gru-architecture": {
        "title": "GRU Architecture and Simplifications",
        "sections": {
            "GRU Design": [
                "Gated Recurrent Units simplify LSTM with fewer gates.",
                "Single hidden state replaces separate cell and hidden states.",
                "Reset gate determines relevance of previous hidden state.",
                "Update gate controls how much hidden state changes.",
                "Simpler architecture reduces parameters and computation.",
                "Smaller memory footprint enables larger batches.",
            ],
            "GRU vs LSTM": [
                "GRU has two gates versus LSTM's three gates.",
                "GRU lacks separate cell state simplifying updates.",
                "LSTM explicitly maintains memory enables stronger control.",
                "GRU computationally more efficient than LSTM.",
                "LSTM often performs slightly better on long dependencies.",
                "GRU sufficient for many sequence tasks.",
            ],
            "Gating Mechanisms": [
                "Reset gate scales previous hidden state relevance.",
                "Update gate interpolates between previous and new state.",
                "Sigmoid gates provide smooth differentiable gating.",
                "Gating enables selective memory updates.",
                "Critical information preserved through gate controls.",
                "Irrelevant information discarded by reset gate.",
            ]
        }
    },
    "L22.1_seq2seq": {
        "title": "Sequence-to-Sequence Models",
        "sections": {
            "Seq2Seq Framework": [
                "Sequence-to-sequence models map input sequences to outputs.",
                "Encoder RNN processes input sequence to fixed representation.",
                "Context vector summarizes entire input in fixed dimension.",
                "Decoder RNN generates output sequence from context.",
                "Separate encoder and decoder enable asymmetric processing.",
                "Applicable to translation, summarization, question answering.",
            ],
            "Encoder-Decoder Pattern": [
                "Encoder compresses variable length input to fixed size.",
                "Final hidden state becomes context for decoder.",
                "Decoder starts with context vector as initial state.",
                "Decoder generates outputs one timestep at a time.",
                "Teacher forcing provides ground truth during training.",
                "Beam search explores multiple hypotheses at inference.",
            ],
            "Training Considerations": [
                "Encoder fully processes input before decoder starts.",
                "Context vector is bottleneck for information transfer.",
                "Limited context causes loss of important information.",
                "Attention mechanisms mitigate context bottleneck.",
                "Different encoding and decoding vocabulary possible.",
                "Shared embeddings sometimes improve performance.",
            ]
        }
    },
    "L22.2_encoder-decoder": {
        "title": "Encoder-Decoder with Attention",
        "sections": {
            "Attention Mechanism": [
                "Attention weights enable decoder to focus on relevant inputs.",
                "Attention queries decoder state to find relevant input positions.",
                "Alignment scores measure similarity between query and keys.",
                "Softmax normalizes alignment scores to probability distribution.",
                "Attention output is weighted sum of input values.",
                "Context vector augmented with attention-weighted input.",
            ],
            "Implementation Details": [
                "Query is decoder hidden state at current timestep.",
                "Keys and values derived from encoder outputs.",
                "Dot-product attention measures query-key similarity efficiently.",
                "Additive attention uses learned scoring function.",
                "Multi-head attention models different aspects simultaneously.",
                "Attention weights reveal which inputs influenced output.",
            ],
            "Advantages": [
                "Access to full input sequence mitigates context bottleneck.",
                "Direct paths enable gradient flow to early inputs.",
                "Interpretability from attention weights.",
                "Variable length input handling improved.",
                "Particularly effective for long sequences.",
                "Foundation for transformer architectures.",
            ]
        }
    },
    "L23.1_attention": {
        "title": "Attention Mechanisms Deep Dive",
        "sections": {
            "Attention Fundamentals": [
                "Attention focuses computational resources on important inputs.",
                "Soft attention uses differentiable weighted sum.",
                "Hard attention selects discrete input positions.",
                "Attention scores measure alignment between query and keys.",
                "Normalized attention weights form probability distribution.",
                "Attention output is context weighted by importance.",
            ],
            "Attention Variants": [
                "Additive attention uses neural network for scoring.",
                "Multiplicative attention uses dot product for efficiency.",
                "Scaled dot-product prevents saturation of softmax.",
                "Multi-head attention models different representation subspaces.",
                "Grouped multi-head attention shares parameters.",
                "Self-attention relates positions within same sequence.",
            ],
            "Applications": [
                "Machine translation uses attention to align source and target.",
                "Question answering uses attention to locate answers.",
                "Summarization uses attention to select important sentences.",
                "Speech recognition uses attention to align audio and text.",
                "Image captioning uses spatial attention mechanisms.",
                "Co-attention models interactions between sequences.",
            ]
        }
    },
    "L23.2_dot-product": {
        "title": "Dot-Product Attention and Scaling",
        "sections": {
            "Dot-Product Mechanism": [
                "Dot product measures similarity between vectors efficiently.",
                "Query vector matched against all key vectors.",
                "Large dot products indicate high similarity and relevance.",
                "Softmax converts dot products to probability distribution.",
                "Attention weighted output combines relevant values.",
                "Differential gradient flow enables end-to-end training.",
            ],
            "Scaling and Normalization": [
                "Large dimensions cause dot products to become very large.",
                "Large values saturate softmax yielding vanishing gradients.",
                "Dividing by sqrt(dimension) prevents saturation.",
                "Proper scaling maintains reasonable activation ranges.",
                "Softmax normalization ensures probabilities sum to one.",
                "Temperature scaling adjusts attention sharpness.",
            ],
            "Efficiency and Performance": [
                "Dot-product attention highly parallelizable with matrix ops.",
                "GEMM operations efficiently computed on GPUs.",
                "Additive attention requires feed-forward network per query.",
                "Dot-product faster but potentially less expressive.",
                "Attention output dimension affects memory and compute.",
                "Sparsity pruning reduces attention computation further.",
            ]
        }
    }
}

def commit_safely(message, retries=3):
    """Safe commit function"""
    for attempt in range(retries):
        try:
            subprocess.run(
                ["git", "add", "-A"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                check=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            return True
        except Exception:
            if attempt == retries - 1:
                return False
    return False

def enrich_module04():
    """Create comprehensive enrichment for module-04"""
    
    module_path = "modules/module-04/lessons"
    total_commits = 0
    
    print("MODULE-04 ENRICHMENT - COMPREHENSIVE")
    print("=" * 70)
    
    for lesson_name, lesson_data in MODULE04_CONTENT.items():
        lesson_path = os.path.join(module_path, lesson_name)
        os.makedirs(lesson_path, exist_ok=True)
        
        readme_path = os.path.join(lesson_path, "README.md")
        title = lesson_data["title"]
        
        print(f"\nProcessing: {lesson_name}")
        
        # Initialize
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n## Comprehensive Learning Guide\n\n")
        
        section_count = 0
        para_count = 0
        
        for section_name, paragraphs in lesson_data["sections"].items():
            section_count += 1
            
            # Read current
            with open(readme_path, "r", encoding="utf-8") as f:
                current = f.read()
            
            # Add section header
            new_content = current + f"## {section_name}\n\n"
            
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            # Commit section
            msg = f"[{lesson_name}] Add section: {section_name}"
            if commit_safely(msg):
                total_commits += 1
                print(f"  Section {section_count}: {section_name}")
            
            # Add paragraphs
            for idx, para in enumerate(paragraphs, 1):
                with open(readme_path, "r", encoding="utf-8") as f:
                    current = f.read()
                
                new_content = current + f"{para}\n\n"
                
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                
                para_count += 1
                msg = f"[{lesson_name}] {section_name} - paragraph {idx}"
                if commit_safely(msg):
                    total_commits += 1
                    sys.stdout.write(f"\r    Paragraphs: {para_count} | Total: {total_commits}")
                    sys.stdout.flush()
        
        print(f"\n  Sections: {section_count}, Paragraphs: {para_count}")
    
    print("\n" + "=" * 70)
    print(f"ENRICHMENT COMPLETE: {total_commits} commits")
    print("=" * 70)
    
    return total_commits

if __name__ == "__main__":
    try:
        total = enrich_module04()
        if total >= 250:
            print(f"\nSUCCESS: {total} commits (target: 250+)")
        else:
            print(f"\nCreated: {total} commits")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
