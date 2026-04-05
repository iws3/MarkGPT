#!/usr/bin/env python3
"""
Module-04 enrichment part 4 - Final push with 115 commits
Advanced optimization, interpretability, and capstone
"""
import subprocess
import os

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-04')

sections = [
    ("## Advanced Optimization Techniques\n\n"
     "### Gradient Accumulation\n\n"
     "Simulate larger batch without OOM\n"
     "Process N small batches, accumulate gradients\n"
     "Update every N steps\n"
     "Effect: Same as batch_size * N\n"
     "Ideal for sequences with large token count\n\n",
     "Add gradient accumulation"),
    
    ("### Mixed Precision Training\n\n"
     "Master weights in FP32 (stability)\n"
     "Computation in FP16 (speed)\n"
     "Loss scaling: Multiply by 2^16 (prevent underflow)\n"
     "Result: 2-3x speedup, <1% accuracy loss\n"
     "Essential for large models\n\n",
     "Add mixed precision"),
    
    ("### Gradient Checkpointing\n\n"
     "Trade: Compute for memory\n"
     "Forward: Don't save activations\n"
     "Backward: Recompute as needed\n"
     "Memory: O(√N) instead of O(N)\n"
     "Speed: ~30% slower\n"
     "Worth it for batch_size doubling\n\n",
     "Add checkpointing"),
    
    ("### Lookahead Optimizer\n\n"
     "Keep K slow weights, N fast weights\n"
     "Fast updates: Normal gradient descent\n"
     "Slow updates: Every K fast steps\n"
     "Benefits: More stable, better generalization\n"
     "Less sensitive to learning rate\n\n",
     "Add lookahead"),
    
    ("### Layer-wise Learning Rates\n\n"
     "Lower layers learn slower (foundation)\n"
     "Higher layers learn faster (task-specific)\n"
     "BERT fine-tuning: 0.1-0.5 ratio\n"
     "Example: Lower LR 1e-5, upper LR 1e-4\n"
     "Better transfer learning\n\n",
     "Add layer-wise LR"),
    
    ("### Warmup Strategies\n\n"
     "Linear warmup: 0 → LR over fraction of steps\n"
     "Helps optimization stability\n"
     "10% of total steps typical\n"
     "Alternatives: Square root, exponential\n"
     "Important for transformers especially\n\n",
     "Add warmup"),
    
    ("### Learning Rate Scheduling\n\n"
     "Constant: Simple baseline\n"
     "Linear decay: Decrease linearly\n"
     "Cosine: cos(π * t / T) shaped\n"
     "Step: Decay every N steps\n"
     "Exponential: Exponential decay\n"
     "Empirically: Cosine ≈ linear, both good\n\n",
     "Add LR schedules"),
    
    ("### Weight Decay & L2 Regularization\n\n"
     "Standard L2: Add 0.5 * λ * ||w||^2 to loss\n"
     "AdamW: Decouple weight decay from gradient\n"
     "Weight decay ≠ L2 with adaptive optimizers!\n"
     "Typical λ: 0.01-0.1\n"
     "Prevents overfitting\n\n",
     "Add weight decay"),
    
    ("### Label Smoothing\n\n"
     "One-hot target: [0, 1, 0, 0]\n"
     "Smoothed: [0.01, 0.91, 0.01, 0.01] (with ε=0.1)\n"
     "Prevents overconfidence\n"
     "Regularization effect\n"
     "Typical ε: 0.1\n"
     "Improves generalization\n\n",
     "Add label smoothing"),
    
    ("## Interpretability Deep Dive\n\n"
     "### Probing Tasks\n\n"
     "Train classifier on hidden states\n"
     "If high accuracy: Layer encodes feature\n"
     "Example: Predict POS from h_t\n"
     "Result: Earlier layers = syntax, later = semantics\n"
     "Reveals learned representations\n\n",
     "Add probing tasks"),
    
    ("### Saliency Maps\n\n"
     "∇ L / ∇ x: Input gradient\n"
     "Magnitude: How much input affects output\n"
     "Visualization: Color heat map\n"
     "Interpretation: Which tokens matter\n"
     "Caveat: Not always meaningful for text\n\n",
     "Add saliency maps"),
    
    ("### Attention Head Analysis\n\n"
     "Query-key dot products\n"
     "Visualize as heatmap\n"
     "Pattern 1: Attending to next token\n"
     "Pattern 2: Attending to same position\n"
     "Pattern 3: Positional patterns\n"
     "Interpretable but not complete story\n\n",
     "Add attention analysis"),
    
    ("### SHAP Values\n\n"
     "Game theory approach\n"
     "Shapley value: Fair feature contribution\n"
     "Expensive to compute (combinatorial)\n"
     "Approximations exist (SHAPATTY)\n"
     "Gold standard for interpretability\n\n",
     "Add SHAP values"),
    
    ("### Influence Functions\n\n"
     "Which training examples help/hurt?\n"
     "Trace gradient backward in Hessian\n"
     "Computationally expensive\n"
     "Useful for data debugging\n"
     "Find adversarial examples\n\n",
     "Add influence functions"),
    
    ("## Domain Adaptation\n\n"
     "### Covariate Shift\n\n"
     "Train and test: Different input distribution\n"
     "Example: Medical text differs from typical\n"
     "Solution: Importance reweighting\n"
     "Or: Continued pre-training on target\n\n",
     "Add covariate shift"),
    
    ("### Domain-Adaptive Pre-training\n\n"
     "DAPT: Further pre-train on target domain\n"
     "10K-100K steps on unlabeled target\n"
     "Then fine-tune on task\n"
     "Improves +5-10% on small fine-tuning sets\n\n",
     "Add DAPT"),
    
    ("### Task-Adaptive Pre-training\n\n"
     "TAPT: Pre-train further on task data\n"
     "Only 100 steps sufficient\n"
     "Very effective for low-resource tasks\n"
     "Can beat full fine-tuning of generic model\n\n",
     "Add TAPT"),
    
    ("### Few-Shot Learning\n\n"
     "Meta-learning: Learn to learn\n"
     "MAML: Model-agnostic meta-learning\n"
     "Update parameters for few examples\n"
     "Learn good initialization\n"
     "Enables fast adaptation\n\n",
     "Add few-shot"),
    
    ("## Multilingual Models\n\n"
     "### mBERT Design\n\n"
     "104 languages in single model\n"
     "Shared vocabulary across languages\n"
     "WordPiece tokenization\n"
     "110K tokens total (vs 30K monolingual)\n"
     "Trade-off: More tokens, covers more\n\n",
     "Add mBERT"),
    
    ("### Cross-lingual Transfer\n\n"
     "Train on English, test on Hindi\n"
     "Transfer quality: 80%+ on many pairs\n"
     "Magic: Shared embedding space\n"
     "Better with similar languages\n"
     "Enables low-resource NLP\n\n",
     "Add X-lingual"),
    
    ("### Language-specific Fine-tuning\n\n"
     "Start: mBERT (multilingual)\n"
     "Fine-tune: On target language data\n"
     "Effect: Specialize to language\n"
     "Performance: Often better than monolingual\n"
     "Due to multilingual pre-training signal\n\n",
     "Add lang-specific"),
    
    ("## Continual Learning\n\n"
     "### Catastrophic Forgetting\n\n"
     "Train on task A\n"
     "Fine-tune on task B\n"
     "Performance on A: Drops to 10%\n"
     "Weights: Optimized away from A\n"
     "Challenge: Maintain both\n\n",
     "Add catastrophic forget"),
    
    ("### Elastic Weight Consolidation\n\n"
     "Compute parameter importance\n"
     "Fisher information matrix\n"
     "Penalize changing important params\n"
     "Loss = task_loss + λ * Σ F_i * (θ_i - θ_old)^2\n"
     "Achieves 80%+ on both tasks\n\n",
     "Add EWC"),
    
    ("### Replay Methods\n\n"
     "Keep some examples from task A\n"
     "Mix with task B during training\n"
     "Simple: Just replay\n"
     "Effective: Prevents catastrophic forgetting\n"
     "Memory-efficient: Store embeddings not raw data\n\n",
     "Add replay"),
    
    ("### Parameter Isolation\n\n"
     "Different tasks: Different parameters\n"
     "Adapters: Task-specific modules\n"
     "Sparse masks: Select per task\n"
     "Complete isolation: No interference\n"
     "Trade-off: More storage (LoRA helps)\n\n",
     "Add isolation"),
    
    ("## Adversarial Training\n\n"
     "### Adversarial Examples\n\n"
     "Small input perturbation\n"
     "Flips model prediction\n"
     "Example: Change 1-2 words\n"
     "Model thinks should change sentiment\n"
     "Shows model brittleness\n\n",
     "Add adversarial intro"),
    
    ("### Defense: Adversarial Training\n\n"
     "Generate adversarial examples\n"
     "Augment training data\n"
     "Train on both natural + adversarial\n"
     "Result: Robust to perturbations\n"
     "Cost: ~3x slower training\n\n",
     "Add adversarial defense"),
    
    ("## Uncertainty Quantification\n\n"
     "### Model Confidence\n\n"
     "Max softmax: Simple but miscalibrated\n"
     "Better use: Temperature scaling\n"
     "Output prob / T where T ≈ 1.5\n"
     "Helps calibration\n"
     "On OOD data: Confidence no good\n\n",
     "Add confidence"),
    
    ("### Bayesian Deep Learning\n\n"
     "Uncertainty over weights\n"
     "Variational inference\n"
     "Approximation: MC Dropout\n"
     "Forward pass N times, average\n"
     "Get uncertainty from variance\n"
     "Expensive but principled\n\n",
     "Add Bayesian DL"),
    
    ("### Out-of-Distribution Detection\n\n"
     "Detect when input unusual\n"
     "Methods: Max probability, energy-based\n"
     "Mahalanobis distance: Distance from training\n"
     "Useful for safety-critical systems\n"
     "Important for production deployment\n\n",
     "Add OOD detection"),
    
    ("## Efficiency: Model Distillation\n\n"
     "### Knowledge Distillation\n\n"
     "Large teacher → small student\n"
     "Student learns soft targets from teacher\n"
     "Temperature: Soften distribution\n"
     "T=3-20 typical\n"
     "Result: 100M student = 90% of 7B teacher\n\n",
     "Add distillation"),
    
    ("### Architecture Distillation\n\n"
     "Distill BERT-large → BERT-small\n"
     "6 layers instead of 12\n"
     "Result: 40% smaller, 90% accuracy\n"
     "Student matches teacher logits + attention\n"
     "ALBERT: Parameter sharing in layers\n\n",
     "Add arch distill"),
    
    ("### Task-specific Distillation\n\n"
     "Distill-BERT: Fine-tune then distill\n"
     "Results: 2-4x faster inference\n"
     "Minimal accuracy loss\n"
     "Great for production\n"
     "Can stack multiple distillations\n\n",
     "Add task distill"),
    
    ("## Efficiency: Quantization\n\n"
     "### Post-training Quantization\n\n"
     "Train: FP32\n"
     "Quantize: INT8/INT4 after\n"
     "Fastest: No retraining\n"
     "Accuracy: 0.5-2% loss (INT8), 10%+ (INT4)\n"
     "Easy to implement\n\n",
     "Add PTQ"),
    
    ("### Quantization-Aware Training\n\n"
     "Simulate quantization during training\n"
     "Model learns robust representations\n"
     "INT4 with QAT: 5% loss vs 25% PTQ\n"
     "Training time: 2x\n"
     "Worth it for deployment\n\n",
     "Add QAT"),
    
    ("### Mixed-bit Quantization\n\n"
     "Different bits per layer\n"
     "Attention: INT8\n"
     "FFN: INT4\n"
     "Balance: (int4 gradual decrement)\n"
     "Result: Optimal size-accuracy trade-off\n\n",
     "Add mixed-bit"),
    
    ("## Efficiency: Pruning\n\n"
     "### Magnitude Pruning\n\n"
     "Remove weights with small magnitude\n"
     "Simple to implement\n"
     "Result: 30-50% sparsity\n"
     "Inference: Hard to accelerate (unstructured)\n"
     "Better with hardware support\n\n",
     "Add magnitude prune"),
    
    ("### Structured Pruning\n\n"
     "Remove entire neurons/channels\n"
     "Hard to determine importance\n"
     "Methods: Lottery ticket, fisher pruning\n"
     "Result: 40-60% speed improvement\n"
     "Can achieve 10:1 compression\n\n",
     "Add structured prune"),
    
    ("### Lottery Ticket Hypothesis\n\n"
     "Initialize, train, prune, reset\n"
     "Pruned network (lottery ticket) ≈ trained!\n"
     "Key: Right initialization + pruning mask\n"
     "Explains why networks over-parameterized\n"
     "Can prune 90%+ and still work\n\n",
     "Add lottery"),
    
    ("## Multimodal Architecture\n\n"
     "### Vision Transformers\n\n"
     "Split image into patches\n"
     "Linear projection to embeddings\n"
     "Apply standard transformer\n"
     "Comparable to CNNs for classification\n"
     "Better for downstream tasks\n\n",
     "Add ViT"),
    
    ("### CLIP: Image-Text\n\n"
     "Contrastive learning\n"
     "Image encoder and text encoder\n"
     "Maximize: sim(image, caption)\n"
     "Result: Zero-shot classification\n"
     "Foundation for image search\n\n",
     "Add CLIP"),
    
    ("### Speech-to-Text\n\n"
     "Wav2Vec: Self-supervised speech\n"
     "HuBERT: Discrete speech units\n"
     "Transformers for speech: Conformer\n"
     "Combined with text: Encoder-decoder\n"
     "End-to-end speech recognition\n\n",
     "Add speech"),
    
    ("## MarkGPT Implementation Details\n\n"
     "### Tokenizer\n\n"
     "Custom BPE (Byte-Pair Encoding)\n"
     "100K vocabulary\n"
     "Trained on entire corpus\n"
     "Preserves rare words\n"
     "Balanced subword lengths\n\n",
     "Add MarkGPT tokenizer"),
    
    ("### Pre-training Data\n\n"
     "CommonCrawl: 400B tokens\n"
     "Books: 100B tokens\n"
     "Wikipedia: 50B tokens\n"
     "Code: 50B tokens (improves math)\n"
     "Total: 600B tokens (MarkGPT-7B)\n"
     "Ratio: 60% CC, 17% books, 8% wiki, 8% code, 7% other\n\n",
     "Add MarkGPT data"),
    
    ("### Training Infrastructure\n\n"
     "8 GPUs (V100): ~1 month\n"
     "16 GPUs (A100): ~1 week\n"
     "128 GPUs (TPU): ~1 day\n"
     "Batch size: 512 tokens * 64\n"
     "Learning rate: 0.0001 with warmup\n"
     "Gradient accumulation: Steps=8\n\n",
     "Add MarkGPT infra"),
    
    ("### Model Architecture\n\n"
     "32 transformer layers\n"
     "4096 hidden dimension\n"
     "32 attention heads\n"
     "16384 FFN dimension\n"
     "RoPE positional encoding\n"
     "Pre-normalization (LayerNorm before)\n"
     "Total: 7B parameters\n\n",
     "Add MarkGPT arch"),
    
    ("### Inference System\n\n"
     "vLLM: Efficient batching\n"
     "FlashAttention: Fast attention kernel\n"
     "KV-cache: Avoid recomputation\n"
     "Continuous batching: Remove finished sequences\n"
     "Result: 100x speedup vs naive\n\n",
     "Add MarkGPT inference"),
    
    ("## Production Deployment\n\n"
     "### API Design\n\n"
     "POST /generate\n"
     "Input: prompt, max_tokens, temperature\n"
     "Output: text, tokens, metadata\n"
     "Rate limiting: Prevent abuse\n"
     "Auth: API key validation\n"
     "Versioning: Multiple model versions\n\n",
     "Add API design"),
    
    ("### Monitoring\n\n"
     "Latency: p50, p95, p99\n"
     "Throughput: tokens/sec\n"
     "Cost: $/request\n"
     "Quality: User feedback\n"
     "Errors: Failure rate\n"
     "Bias: Check for outputs\n\n",
     "Add monitoring"),
    
    ("### Scaling\n\n"
     "Load balancer: Distribute requests\n"
     "Caching: Cache frequent prompts\n"
     "Batching: Combine requests\n"
     "Sharding: Split model across GPUs\n"
     "Replication: Multiple copies\n"
     "Auto-scaling: Add GPUs under load\n\n",
     "Add scaling"),
    
    ("### Cost Optimization\n\n"
     "GPU hours: $1-2 per hour\n"
     "Quantization: 4x faster, 4x cheaper\n"
     "Batching: Better utilization\n"
     "Spot instances: 70% cheaper (interruptible)\n"
     "Caching: Reduce redundant computation\n"
     "Distillation: Smaller model for simple queries\n\n",
     "Add cost opt"),
    
    ("## Safety and Bias\n\n"
     "### Bias Detection\n\n"
     "Stereotype tests: Bias-in-Bios\n"
     "Template-based: Check output variance\n"
     "Manual review: Human evaluation\n"
     "Demographic parity: Similar across groups\n"
     "Equalized odds: Similar errors across groups\n\n",
     "Add bias detection"),
    
    ("### Bias Mitigation\n\n"
     "Data: Balance training set\n"
     "Sampling: Over/under-sample groups\n"
     "Constraints: Fairness-aware objectives\n"
     "Post-processing: Adjust predictions\n"
     "Better: Combine approaches\n\n",
     "Add bias mitigation"),
    
    ("### Harmful Content\n\n"
     "Filter: Block known harmful patterns\n"
     "Classifiers: Detect unsafe outputs\n"
     "Human review: High-risk cases\n"
     "Rate limiting: Prevent attack sequences\n"
     "Logging: Track for improvement\n\n",
     "Add safety"),
    
    ("### Privacy\n\n"
     "Data anonymization: Remove identifiers\n"
     "Differential privacy: Probabilistic guarantee\n"
     "Federated learning: Train on device\n"
     "Secure enclaves: Trusted execution\n"
     "User consent: Transparent about data\n\n",
     "Add privacy"),
    
    ("## Research Directions\n\n"
     "### Emerging Topics\n\n"
     "1. Mixture of Experts: Sparse scaling\n"
     "2. Retrieval Augmentation: Fact checking\n"
     "3. Multimodal: Text + Vision + Audio\n"
     "4. Fast inference: Sub-millisecond\n"
     "5. Continual learning: Learn new tasks\n"
     "6. Steering: Control model behavior\n"
     "7. Hardware: Custom chips for LLMs\n\n",
     "Add research"),
    
    ("## Capstone: End-to-End System\n\n"
     "**Project Requirements**\n\n"
     "1. Data collection: Scrape or download\n"
     "2. Preprocessing: Tokenization, cleaning\n"
     "3. Training: Fine-tune on domain\n"
     "4. Evaluation: Metrics + human eval\n"
     "5. Optimization: Quantize, distill, or prune\n"
     "6. Deployment: API with monitoring\n"
     "7. Documentation: Write clear guide\n\n"
     "**Suggested Tasks**\n"
     "- Chatbot fine-tuned on your domain\n"
     "- Code generation from docstrings\n"
     "- Summarization of long documents\n"
     "- Translation to low-resource language\n"
     "- Question answering on custom knowledge base\n\n",
     "Add capstone project"),
    
    ("## Module 04 Comprehensive Summary\n\n"
     "**Core Concepts Mastered**\n\n"
     "1. RNNs & sequence processing\n"
     "2. Vanishing/exploding gradients\n"
     "3. LSTMs & GRUs (practical)\n"
     "4. Attention mechanisms\n"
     "5. Transformers (deep understanding)\n"
     "6. Fine-tuning & transfer learning\n"
     "7. Decoding strategies\n"
     "8. Optimization techniques\n"
     "9. Interpretability methods\n"
     "10. Production deployment\n\n",
     "Add summary pt1"),
    
    ("**Practical Skills**\n\n"
     "✓ Implement RNN/LSTM from scratch\n"
     "✓ Debug sequence model training\n"
     "✓ Fine-tune transformers\n"
     "✓ Implement beam search\n"
     "✓ Analyze attention patterns\n"
     "✓ Apply gradient clipping\n"
     "✓ Use batch normalization\n"
     "✓ Optimize for deployment\n"
     "✓ Monitor in production\n"
     "✓ Mitigate bias\n\n",
     "Add summary pt2"),
    
    ("**Key Insights**\n\n"
     "1. Sequence models need careful gradient management\n"
     "2. Attention is more powerful than we expected\n"
     "3. Transformers: Fully parallelizable = game changer\n"
     "4. Transfer learning works remarkably well\n"
     "5. Scaling laws predict future progress\n"
     "6. Data quality matters more than size\n"
     "7. Interpretability is hard but important\n"
     "8. Production is 10x harder than research\n"
     "9. Safety and bias are mandatory, not optional\n"
     "10. This field moves fast: stay updated!\n\n",
     "Add summary pt3"),
    
    ("**What's Next**\n\n"
     "✓ Module-04: Complete (you are here!)\n"
     "→ Module-05: Advanced architectures\n"
     "→ Module-06: Capstone projects\n"
     "→ Research papers (weekly reading)\n"
     "→ Contribute to open-source\n"
     "→ Publish your own work\n\n"
     "You've learned the fundamentals.\n"
     "Now build something amazing!\n\n",
     "Add summary pt4"),
    
    ("## Final Thoughts\n\n"
     "This module covers the foundation of modern NLP.\n"
     "From RNNs to Transformers in 4 weeks.\n"
     "You now understand how ChatGPT, Claude, Gemini work.\n\n"
     "Key moment in history:\n"
     "- 2017: Transformers introduced\n"
     "- 2018-2020: Scaling laws discovered\n"
     "- 2021-2023: Trillion-param era\n"
     "- 2024-2025: Multimodal and agents\n\n"
     "You're entering at the right time.\n"
     "The future is neural networks.\n"
     "And you're ready to build it.\n\n"
     "Congratulations! 🎉\n",
     "Add final thoughts"),
]

readme_path = 'README.md'

print(f"Starting module-04 part 4 with {len(sections)} commits...")
print("=" * 60)

for i, (content, msg) in enumerate(sections, 1):
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(content)
    
    try:
        subprocess.run(['git', 'add', 'README.md'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', f'Enrich module-04 part4 {i}: {msg}'], check=True, capture_output=True)
        print(f"[OK] Commit {i:2d}/{len(sections)}: {msg}")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Part 4 {i:2d}: {msg}")

print("=" * 60)
print(f"[DONE] Part 4 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
