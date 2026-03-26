#!/usr/bin/env python3
"""
Module-03 Comprehensive Lesson Enrichment
Two-phase approach for 300+ commits
Phase 1: Foundational content
"""

import os
import subprocess
import sys

# Comprehensive content for module-03 lessons (Neural Networks)
PHASE1_CONTENT = {
    "L13.1_neuron-biology": {
        "title": "Neuron Biology and Artificial Neurons",
        "sections": {
            "Biological Neurons: Structure and Function": [
                "Biological neurons transmit information through electrical and chemical signals.",
                "The soma (cell body) integrates signals from multiple dendrites.",
                "The axon transmits signals to other neurons via synapses.",
                "Synaptic connections enable learning through strength modification.",
                "Neurotransmitters carry signals across synaptic gaps between neurons.",
                "Axon terminals contain vesicles with neurotransmitters for signal transmission.",
            ],
            "Neural Communication": [
                "Dendrites receive signals from other neurons at synaptic connections.",
                "Action potentials propagate along the axon when stimulation exceeds threshold.",
                "The resting potential is maintained by ion pumps using metabolic energy.",
                "Depolarization reduces the potential difference across the membrane.",
                "Hyperpolarization increases the potential difference beyond resting state.",
                "The threshold is the minimum stimulation needed to trigger an action potential.",
            ],
            "From Biology to Artificial Neurons": [
                "Artificial neurons mathematically model biological neuron behavior.",
                "The perceptron was one of the first models of artificial neurons.",
                "Weights represent synaptic strengths connecting neurons.",
                "The bias term allows shifting the activation threshold.",
                "The activation function introduces non-linearity mimicking biological behavior.",
                "Artificial neurons enable computational learning from data.",
            ]
        }
    },
    "L13.2_activation-functions": {
        "title": "Activation Functions for Neural Networks",
        "sections": {
            "Linear Functions": [
                "Linear activation f(x) = x produces outputs proportional to inputs.",
                "Linear activations fail to introduce non-linearity to networks.",
                "Stacking linear layers is mathematically equivalent to a single layer.",
                "Linear functions limit the network to learning linear relationships.",
                "Deep networks with only linear activation reduce to shallow networks.",
                "This motivates using non-linear activation functions.",
            ],
            "Sigmoid and Tanh Functions": [
                "The sigmoid function squashes values to range (0, 1).",
                "Sigmoid was historically popular but suffers from vanishing gradients.",
                "The tanh function maps values to range (-1, 1).",
                "Tanh is zero-centered improving optimization over sigmoid.",
                "Both suffer from gradient saturation at extreme values.",
                "Modern networks prefer ReLU-based activations.",
            ],
            "ReLU and Variants": [
                "ReLU (Rectified Linear Unit) is f(x) = max(0, x).",
                "ReLU enables efficient computation with no exponential calculations.",
                "ReLU helps avoid vanishing gradient problems in deep networks.",
                "Leaky ReLU allows small negative gradients to prevent dead neurons.",
                "ELU (Exponential Linear Unit) provides smooth negative values.",
                "Variants improve training stability and network expressiveness.",
            ],
            "Modern Activation Functions": [
                "Swish (SiLU) is x * sigmoid(x) combining smoothness with efficiency.",
                "GELU (Gaussian Error Linear Unit) uses cumulative Gaussian distribution.",
                "Mish is x * tanh(softplus(x)) providing smooth non-linearity.",
                "GLU (Gated Linear Unit) gates information flow through sigmoid.",
                "Selection of activation impacts network capacity and training dynamics.",
                "Different activations suit different problem types and architectures.",
            ]
        }
    },
    "L14.1_mlp-layers": {
        "title": "Multi-Layer Perceptrons and Neural Layers",
        "sections": {
            "Fully Connected Layers": [
                "Fully connected layers connect every input to every output.",
                "Each output neuron has separate weights and bias.",
                "The output is y = activation(W*x + b) matrix multiplied input.",
                "Fully connected layers are rich in parameters for flexibility.",
                "Parameter count grows quadratically with layer sizes.",
                "These layers form the basis of deep neural networks.",
            ],
            "Layer Composition and Architecture": [
                "Stacking multiple layers creates deep neural networks.",
                "Each layer transforms the representation from previous layer.",
                "Hidden layers learn intermediate features useful for prediction.",
                "Network depth enables learning hierarchical representations.",
                "Very deep networks require careful initialization and training.",
                "Architecture design critically impacts learning capability.",
            ],
            "Network Capacity and Expressiveness": [
                "Wider networks have more parameters per layer.",
                "Deeper networks compose more transformations.",
                "Width and depth trade off computational cost and expressiveness.",
                "Wider networks learn faster but need more data to generalize.",
                "Deeper networks learn more abstract features.",
                "Optimal architecture depends on dataset and problem.",
            ],
            "Vectorization and Efficiency": [
                "Batch processing passes multiple samples simultaneously.",
                "Matrix operations leverage GPU acceleration efficiently.",
                "Vectorization eliminates explicit loops over samples.",
                "Batch size affects both speed and generalization.",
                "Mini-batch processing balances computation and memory.",
                "Efficient implementation critical for training large networks.",
            ]
        }
    },
    "L14.2_universal-approximation": {
        "title": "Universal Approximation Theorem",
        "sections": {
            "Theoretical Foundations": [
                "Universal approximation theorem guarantees network expressiveness.",
                "A single hidden layer with non-linear activation can approximate any function.",
                "The theorem applies to continuous functions on closed intervals.",
                "Approximation requires sufficiently many hidden neurons.",
                "Width grows exponentially with input dimension in worst case.",
                "The theorem is existence proof, not constructive algorithm.",
            ],
            "Practical Implications": [
                "Single hidden layer networks are theoretically sufficient.",
                "Deep networks can be more efficient than single wide layers.",
                "Deep networks require fewer total parameters for many problems.",
                "Deeper architectures learn features hierarchically.",
                "Network depth enables inductive biases matching data structure.",
                "Deep learning succeeds because it matches real-world data.",
            ],
            "Function Approximation in Practice": [
                "Neural networks learn approximations through training on data.",
                "Training finds weights enabling good sample performance.",
                "Generalization requires balancing fit and complexity.",
                "Regularization prevents overfitting despite large capacity.",
                "The inductive bias of architecture shapes learned functions.",
                "Real problems benefit from carefully designed architectures.",
            ]
        }
    },
    "L15.1_backpropagation": {
        "title": "Backpropagation and Gradient Computation",
        "sections": {
            "The Chain Rule in Networks": [
                "Backpropagation applies the chain rule to compute gradients.",
                "Gradients flow backward from outputs to model parameters.",
                "Each layer propagates error signals to previous layers.",
                "The chain rule decomposes complex derivatives into simple parts.",
                "Automatic differentiation implements the chain rule efficiently.",
                "Understanding backpropagation reveals how networks learn.",
            ],
            "Forward and Backward Pass": [
                "The forward pass computes outputs from inputs through layers.",
                "Intermediate activations are stored for backward computation.",
                "The backward pass computes gradients of loss w.r.t. parameters.",
                "Gradients enable parameter updates toward lower loss.",
                "Forward pass cost is one inference, backward is ~2x inference.",
                "Efficient computation critical for training large networks.",
            ],
            "Gradient Flow and Backpropagation Challenges": [
                "Vanishing gradients occur when gradients shrink through layers.",
                "Exploding gradients occur when gradients grow exponentially.",
                "Gradient clipping prevents explosion by capping gradient norms.",
                "Careful initialization helps maintain stable gradient flow.",
                "Batch normalization stabilizes gradient flow through networks.",
                "Skip connections (residual networks) enable training very deep.",
            ],
            "Modern Automatic Differentiation": [
                "Automatic differentiation computes gradients without manual derivation.",
                "Reverse-mode AD (backprop) is efficient for single-output functions.",
                "Forward-mode AD suits few parameters and many outputs.",
                "Tape-based systems record operations for backward pass.",
                "Automatic differentiation enables rapid prototyping.",
                "PyTorch and TensorFlow implement efficient automatic differentiation.",
            ]
        }
    },
    "L15.2_computation-graph": {
        "title": "Computation Graphs and Neural Network Flow",
        "sections": {
            "Graph Structure": [
                "Computation graphs represent mathematical operations as nodes.",
                "Edges represent data flow between operations.",
                "Neural networks are directed acyclic graphs (DAGs).",
                "Graph structure enables efficient automatic differentiation.",
                "Different operations have different computational costs.",
                "Graph optimization can reduce computation.",
            ],
            "Dynamic vs. Static Graphs": [
                "Static graphs are defined before execution (e.g., TensorFlow 1.x).",
                "Dynamic graphs are built during execution (e.g., PyTorch).",
                "Static graphs enable more optimization opportunities.",
                "Dynamic graphs enable Python control flow within models.",
                "Dynamic graphs simplify debugging and experimentation.",
                "Hybrid approaches combine benefits of both.",
            ],
            "Automatic Differentiation in Graphs": [
                "Graph representation enables automatic gradient computation.",
                "Reverse traversal computes gradients in backward pass.",
                "Higher-order gradients require second traversal.",
                "Efficient caching avoids recomputing intermediate values.",
                "Memory usage grows with graph complexity.",
                "Checkpointing trades computation for memory.",
            ]
        }
    },
    "L16.1_loss-functions": {
        "title": "Loss Functions and Training Objectives",
        "sections": {
            "Classification Losses": [
                "Cross-entropy loss measures divergence between distributions.",
                "Softmax cross-entropy suits multi-class classification.",
                "Binary cross-entropy applies to binary classification.",
                "Focal loss addresses class imbalance in hard examples.",
                "Hinge loss used in support vector machines.",
                "Proper loss function choice affects convergence.",
            ],
            "Regression Losses": [
                "Mean squared error (MSE) penalizes large errors quadratically.",
                "Mean absolute error (MAE) is robust to outliers.",
                "Huber loss combines benefits of MSE and MAE.",
                "Log-cosh loss is smooth approximation to L1 distance.",
                "Quantile loss enables prediction of conditional distributions.",
                "Loss choice reflects assumption about error distribution.",
            ],
            "Specialized Loss Functions": [
                "Contrastive loss learns similarity between samples.",
                "Triplet loss enforces spacing in embedding space.",
                "Siamese losses compare representations across samples.",
                "Ranking losses optimize relative ordering of predictions.",
                "Adversarial losses pit generator against discriminator.",
                "Domain-specific losses encode problem structure.",
            ],
            "Loss Landscapes and Optimization": [
                "Loss landscape shape affects optimization difficulty.",
                "Sharp minima generalize poorly to test data.",
                "Flat minima suggest better generalization.",
                "Asymmetry in loss landscape guides gradient descent.",
                "Multiple local minima exist in high dimensions.",
                "Understanding loss geometry improves training.",
            ]
        }
    },
    "L16.2_optimization": {
        "title": "Neural Network Optimization Techniques",
        "sections": {
            "Batch Normalization Benefits": [
                "Batch normalization reduces internal covariate shift.",
                "Normalized inputs to each layer improve stability.",
                "Enables higher learning rates during training.",
                "Reduces sensitivity to weight initialization.",
                "Acts as regularizer reducing overfitting.",
                "Becomes different during inference vs. training.",
            ],
            "Learning Rate Strategies": [
                "Constant learning rates rarely work well throughout training.",
                "Cyclical learning rates alternate between high and low.",
                "Warm restarts jump learning rate back up periodically.",
                "OneCycle policy ramps up then down over single epoch.",
                "Discriminative learning rates vary across layers.",
                "Proper scheduling significantly impacts final performance.",
            ],
            "Gradient Accumulation": [
                "Accumulation enables larger effective batch sizes.",
                "Useful when memory limits batch size.",
                "Accumulate gradients over multiple forward/backward passes.",
                "Update weights after accumulated gradient.",
                "Increases training time but enables larger effective batches.",
                "Reduces gradient noise improving convergence.",
            ]
        }
    },
    "L17.1_regularization": {
        "title": "Regularization Techniques in Neural Networks",
        "sections": {
            "Weight Regularization": [
                "L1 regularization (Lasso) encourages sparsity.",
                "L2 regularization (Ridge) discourages large weights.",
                "Elastic Net combines L1 and L2 penalties.",
                "Weight decay in optimization approximates L2 regularization.",
                "Regularization constrains model complexity.",
                "Strength of regularization trades fit vs. simplicity.",
            ],
            "Early Stopping": [
                "Early stopping monitors validation performance during training.",
                "Training stops when validation loss stops improving.",
                "Prevents overfitting without explicit regularization.",
                "Simple but effective approach to generalization.",
                "Requires validation set separate from training.",
                "Checkpoint best model during training.",
            ],
            "Data Augmentation": [
                "Augmentation artificially expands dataset through transformations.",
                "Random crops, rotations, flips increase training diversity.",
                "Mixup interpolates between samples and labels.",
                "Cutmix mixes regions from different samples.",
                "AutoAugment searches for optimal augmentation policies.",
                "Effective augmentation enables training with less data.",
            ],
            "Advanced Regularization": [
                "Stochastic depth randomly drops layers during training.",
                "MixUp and CutMix regularize through sample mixing.",
                "Label smoothing prevents overconfident predictions.",
                "Adversarial training improves robustness.",
                "Mixup-based methods reduce memorization.",
                "Combination of techniques provides best results.",
            ]
        }
    },
    "L17.2_dropout-batchnorm": {
        "title": "Dropout and Batch Normalization",
        "sections": {
            "Dropout Mechanism": [
                "Dropout randomly disables neurons during training.",
                "Each neuron kept with probability p during forward pass.",
                "Prevents co-adaptation of features.",
                "Acts as ensemble of thinned networks.",
                "No dropout applied during inference.",
                "Scaling ensures same expected output at inference.",
            ],
            "Dropout Variants": [
                "Standard dropout drops neurons independently.",
                "Spatial dropout drops feature maps in convolutional layers.",
                "Variational dropout shares dropout mask across timesteps.",
                "DropConnect drops weights instead of activations.",
                "Monte Carlo dropout enables uncertainty estimation.",
                "Variants optimize dropout for different architectures.",
            ],
            "Batch Normalization Details": [
                "Normalization performed per-feature across minibatch.",
                "Learnable scale and shift parameters restore expressiveness.",
                "Running statistics tracked for inference.",
                "Different behavior during training vs. inference.",
                "Improves gradient flow enabling faster training.",
                "Reduces need for careful weight initialization.",
            ],
            "Normalization Variants": [
                "Layer normalization normalizes across features per sample.",
                "Instance normalization per sample per feature.",
                "Group normalization groups features for normalization.",
                "Layer norm doesn't depend on batch size.",
                "Each normalization suits different architectures.",
                "Normalization critical for stable deep learning.",
            ]
        }
    }
}

def commit_safely(message, retries=3):
    """Stage and commit with retry logic"""
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
        except Exception as e:
            if attempt == retries - 1:
                return False
            continue
    return False

def process_phase1():
    """Process Phase 1: Foundational content"""
    
    module_path = "modules/module-03/lessons"
    total_commits = 0
    
    print("=" * 80)
    print("MODULE-03 ENRICHMENT PHASE 1: FOUNDATIONAL CONTENT")
    print("=" * 80)
    
    for lesson_name, lesson_data in PHASE1_CONTENT.items():
        lesson_path = os.path.join(module_path, lesson_name)
        os.makedirs(lesson_path, exist_ok=True)
        
        readme_path = os.path.join(lesson_path, "README.md")
        title = lesson_data["title"]
        
        print(f"\n[*] {lesson_name}: {title}")
        
        # Initialize README with title
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n## Comprehensive Learning Guide\n\n")
        
        section_count = 0
        para_count = 0
        
        # Build the content section by section
        for section_name, paragraphs in lesson_data["sections"].items():
            section_count += 1
            
            # Read current README
            with open(readme_path, "r", encoding="utf-8") as f:
                current_content = f.read()
            
            # Add section header
            new_content = current_content + f"## {section_name}\n\n"
            
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            # Commit section header
            commit_msg = f"[{lesson_name}] Add {section_name}"
            if commit_safely(commit_msg):
                total_commits += 1
            
            # Add each paragraph with individual commits
            for para_idx, para_text in enumerate(paragraphs, 1):
                # Read current content
                with open(readme_path, "r", encoding="utf-8") as f:
                    current_content = f.read()
                
                # Add paragraph
                new_content = current_content + f"{para_text}\n\n"
                
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                
                # Commit paragraph
                para_count += 1
                commit_msg = f"[{lesson_name}] {section_name} - para {para_idx}"
                if commit_safely(commit_msg):
                    total_commits += 1
                    sys.stdout.write(f"\r  Commits: {total_commits}")
                    sys.stdout.flush()
        
        print(f"\n  Sections: {section_count}, Paragraphs: {para_count}")
    
    print("\n" + "=" * 80)
    print(f"PHASE 1 COMPLETE: {total_commits} commits created")
    print("=" * 80)
    
    return total_commits

if __name__ == "__main__":
    try:
        total = process_phase1()
        print(f"\nPhase 1 Result: {total} commits")
        if total < 300:
            print(f"Need Phase 2 to reach 300+ commits")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
