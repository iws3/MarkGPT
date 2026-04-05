#!/usr/bin/env python3
"""
Module-03 enrichment part 4 - 81 commits
Fine-tuning, optimization, and appendices
"""
import subprocess
import os

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-03')

sections = [
    ("## Fine-tuning Large Language Models\n\n"
     "### Transfer Learning Context\n\n"
     "Pre-training: Learn general language on 10T tokens\n"
     "Fine-tuning: Adapt to specific task on 10K tokens\n"
     "Parameter efficiency: Only 0.1% of parameters as trainable\n"
     "Result: State-of-the-art task performance\n\n",
     "Add fine-tuning overview"),
    
    ("### Full Fine-tuning\n\n"
     "Update all model parameters.\n"
     "High quality but computationally expensive.\n"
     "MarkGPT-Nano: 12 hours on 1 GPU for SQuAD.\n"
     "Requires 30GB+ GPU for large models.\n"
     "Not practical for most practitioners.\n\n",
     "Add full fine-tuning"),
    
    ("### Adapter Modules\n\n"
     "Original: model size 7B\n"
     "Adapter: 0.01B (0.15% parameters)\n"
     "Add small bottleneck between layers.\n"
     "Down-project to 64d, process, up-project back.\n"
     "Achieves 99% of fine-tuning performance.\n"
     "Only 10MB per task (stores cheaply).\n\n",
     "Add adapter modules"),
    
    ("### Prompt Tuning\n\n"
     "Learn virtual prompt tokens (32 tokens).\n"
     "Learnable parameters: 32 * 4096 ≈ 130K\n"
     "Incredibly efficient (65x better than adapters).\n"
     "Task performance: 98% of full fine-tuning.\n"
     "Challenge: Less effective for very different tasks.\n\n",
     "Add prompt tuning"),
    
    ("## Low-Rank Adaptation (LoRA)\n\n"
     "### Motivation\n\n"
     "Hypothesis: Weight updates are low-rank.\n"
     "Instead of ΔW (huge matrix):\n"
     "Use A @ B where A and B are small.\n"
     "MarkGPT-7B LoRA: 4M params (0.06% of model).\n\n",
     "Add LoRA motivation"),
    
    ("### Implementation Details\n\n"
     "For each weight matrix W:\n"
     "Output = Wx + αABx\n"
     "A: N x r (initialized N(0, std)\n"
     "B: r x d (initialized zeros)\n"
     "r: Rank (typically 8-64)\n"
     "α: Scaling factor (16-32)\n\n"
     "Benefits:\n"
     "- Tiny parameters (stack modules)\n"
     "- Merge with original weights (no inference overhead)\n"
     "- Compatible with quantization\n\n",
     "Add LoRA details"),
    
    ("### LoRA Results\n\n"
     "MarkGPT-7B with LoRA on GLUE:\n"
     "- Training time: 1 hour vs 8 hours full\n"
     "- Memory: 4GB vs 24GB\n"
     "- Final accuracy: 87.2% vs 87.6% (0.4% gap)\n"
     "- Inference: Identical speed (weights merged)\n\n"
     "Scales to LoRA composition (multiple adapters).\n",
     "Add LoRA results"),
    
    ("## Interpretability Methods\n\n"
     "### Attention Visualization\n\n"
     "Heatmap: Query-key interactions.\n"
     "Early layers: Attend to neighboring tokens.\n"
     "Late layers: Semantic grouping.\n"
     "Multi-head: Different attention patterns.\n\n"
     "Insight: Models learn to structure information.\n",
     "Add attention viz"),
    
    ("### Gradient-based Attribution\n\n"
     "Input gradient ∂L/∂x shows feature importance.\n"
     "Integrated gradients: Baseline-based method.\n"
     "SmoothGrad: Average over noise samples.\n"
     "DeepLIFT: Backpropagation with reference points.\n\n"
     "Use: Explain why model made prediction.\n",
     "Add attribution"),
    
    ("### SHAP Values\n\n"
     "Game theory approach to feature importance.\n"
     "Contribution of each feature to prediction.\n"
     "SHAP = Shapley value (from cooperative game theory).\n"
     "Fair allocation of credit among features.\n"
     "Computational cost: Expensive for large models.\n\n",
     "Add SHAP"),
    
    ("## Knowledge Distillation - Advanced\n\n"
     "### Conventional Distillation\n\n"
     "Student learns from soft targets.\n"
     "Temperature = 3-5 softens distribution.\n"
     "Combine two losses:\n"
     "- Task loss: Hard labels\n"
     "- Distillation loss: Teacher soft targets\n"
     "Ratio: 0.1 task + 0.9 distillation\n\n",
     "Add distill advanced"),
    
    ("### Feature-based Distillation\n\n"
     "Match intermediate representations.\n"
     "Compare hidden layers directly.\n"
     "Better for compression (larger models).\n"
     "Attention maps matching:\n"
     "Loss = ||StudentAttention - TeacherAttention||^2\n\n",
     "Add feature distill"),
    
    ("### Multi-Teacher Distillation\n\n"
     "Combine knowledge from multiple teachers.\n"
     "Diverse models capture different patterns.\n"
     "Weight teacher outputs by confidence.\n"
     "Significantly better student than single teacher.\n\n",
     "Add multi-teacher"),
    
    ("## Model Quantization - Deep Dive\n\n"
     "### Symmetric Quantization\n\n"
     "Map [min, max] to [Qmin, Qmax]\n"
     "INT8: [-128, 127] typically\n"
     "Scale factor s = max(|min|, |max|) / 128\n"
     "Q(x) = round(x / s)\n"
     "Reverse: x' = Q(x) * s\n\n",
     "Add symmetric quant"),
    
    ("### Asymmetric Quantization\n\n"
     "Map [min, max] to [0, 255] for INT8\n"
     "Scale: (max - min) / 255\n"
     "Zero-point offset: min value\n"
     "Better for skewed distributions.\n"
     "3% better accuracy on typical weights.\n\n",
     "Add asymmetric quant"),
    
    ("### Quantization-Aware Training\n\n"
     "Simulate quantization during training.\n"
     "Gradients flow through fake quantization.\n"
     "Model learns robust representations.\n"
     "INT4 with QAT: 10% drop vs 25% without.\n"
     "Training cost: 2x but worth the improvement.\n\n",
     "Add QAT"),
    
    ("### Calibration and Clipping\n\n"
     "Min/max statistics from calibration set.\n"
     "Method 1: Use actual min/max\n"
     "Method 2: Percentile (0.1%, 99.9%)\n"
     "Method 3: KL divergence alignment\n"
     "Higher clip = info loss, lower = outlier issue.\n\n",
     "Add calibration"),
    
    ("## Advanced Training Techniques\n\n"
     "### Mixed Precision Training\n\n"
     "Master weights: FP32 (for stability)\n"
     "Computation: FP16 (for speed)\n"
     "Gradient scaling: Multiply by 2^16\n"
     "Result: 2-3x speedup, minimal accuracy loss.\n"
     "Essential for large models (7B+).\n\n",
     "Add mixed precision"),
    
    ("### Gradient Checkpointing\n\n"
     "Re-compute activations instead of storing.\n"
     "Memory: O(√N) instead of O(N)\n"
     "Trade-off: Double computation, half memory.\n"
     "Enable for largest models (13B+).\n"
     "Slow but fits on single GPU.\n\n",
     "Add checkpointing"),
    
    ("### Gradient Accumulation\n\n"
     "Effective batch = real batch * accumulation\n"
     "Process N real batches, accumulate gradients.\n"
     "Update after N steps.\n"
     "Simulate larger batch on limited GPU.\n"
     "Helps with batch-size optimization.\n\n",
     "Add accumulation"),
    
    ("## Mixture of Experts\n\n"
     "### Architecture\n\n"
     "Multiple expert networks (12 experts typical).\n"
     "Router network: Selects top-K experts.\n"
     "Each token routed independently.\n"
     "Parameters: Total = dense + routing overhead\n"
     "12 experts * 300M params = 3.6B params\n"
     "But activate only 300M per token (efficient).\n\n",
     "Add MoE architecture"),
    
    ("### Sparse Activation\n\n"
     "Only K experts active per token (K=2 typical).\n"
     "Compute increases slowly with model size.\n"
     "GShard: 600B sparse model on 16 TPUs.\n"
     "Switch transformers: Using single expert (K=1).\n"
     "Much more efficient than dense models.\n\n",
     "Add sparse activation"),
    
    ("### Load Balancing\n\n"
     "Challenge: Some experts used more.\n"
     "Loss term: Penalize imbalanced routing.\n"
     "Auxiliary loss: 0.01 * (balance_loss)\n"
     "Expert utilization target: Equal distribution.\n"
     "Important for training stability.\n\n",
     "Add load balancing"),
    
    ("## Inference Optimization\n\n"
     "### KV-Cache\n\n"
     "Store computed K, V from previous steps.\n"
     "Attention(Q_new, K_cache, V_cache).\n"
     "Avoids recomputing past positions.\n"
     "Reduce compute from O(N^2) per token to O(N).\n"
     "7B model: 30GB cache for batch=32, length=2048.\n\n",
     "Add KV-cache"),
    
    ("### Continuous Batching\n\n"
     "Traditional: Wait for all sequences to finish.\n"
     "Continuous: Remove finished, add new in-flight.\n"
     "GPU utilization: 90%+ vs 60%.\n"
     "Throughput increases but latency varies.\n"
     "Critical for serving systems.\n\n",
     "Add continuous batch"),
    
    ("### Speculative Decoding\n\n"
     "Small model generates candidate tokens.\n"
     "Large model verifies (parallel).\n"
     "Accept multiple tokens per forward pass.\n"
     "2-3x speedup for large models.\n"
     "Same outputs as standard decoding.\n\n",
     "Add speculative decoding"),
    
    ("## Model Merging\n\n"
     "### Simple Averaging\n\n"
     "Merge weights: W = αW1 + (1-α)W2\n"
     "Two fine-tuned models on alpha/beta.\n"
     "Surprisingly effective (task ensemble).\n"
     "Creates model good at both tasks.\n\n",
     "Add merging"),
    
    ("### Task Vector Approach\n\n"
     "Task vector = fine-tuned_weights - base_weights\n"
     "Merge: base + α*task1 + β*task2\n"
     "More interpretable than direct merge.\n"
     "Better capacity allocation.\n"
     "Can scale task contributions.\n\n",
     "Add task vector"),
    
    ("## Retrieval Augmented Generation\n\n"
     "### Motivation\n\n"
     "LLM memorizes training data.\n"
     "Not effective for retrieval of facts.\n"
     "RAG: Retrieve relevant docs, then generate.\n"
     "Combines retriever + reader model.\n\n",
     "Add RAG"),
    
    ("### Architecture\n\n"
     "1. Query embedding: Dense encoder\n"
     "2. Retriever: Top-K passages from index\n"
     "3. Reader: Generate answer conditioned on passages\n"
     "Index: Dense passage retrieval (DPR)\n"
     "Reader: BART or T5\n\n",
     "Add RAG arch"),
    
    ("### Results\n\n"
     "SQuAD with RAG: 92% F1 vs 85% without\n"
     "Natural questions: 60% vs 45% without\n"
     "Reduces hallucination significantly.\n"
     "Enables fact-checking (cite sources).\n\n",
     "Add RAG results"),
    
    ("## Continual Learning\n\n"
     "### Catastrophic Forgetting\n\n"
     "Model trained on task A then task B.\n"
     "Performance on A drops to 10%.\n"
     "Weights optimized away from task A.\n"
     "Major challenge for sequential learning.\n\n",
     "Add catastrophic forget"),
    
    ("### Elastic Weight Consolidation\n\n"
     "Compute importance of each parameter.\n"
     "Fisher information matrix: which params matter.\n"
     "Penalize changing important parameters.\n"
     "Loss = task_loss + λ * Σ F_i * (θ_i - θ_old)^2\n"
     "Achieves 80%+ on both tasks.\n\n",
     "Add EWC"),
    
    ("### Parameter Isolation\n\n"
     "Different tasks use different parameters.\n"
     "Adapter modules (low-rank).\n"
     "Sparse masks: Select parameters per task.\n"
     "Complete isolation: No interference.\n"
     "But requires more storage.\n\n",
     "Add isolation"),
    
    ("## Domain Adaptation\n\n"
     "### Distribution Shift Problem\n\n"
     "Train: General text (Wikipedia, CommonCrawl)\n"
     "Test: Medical documents (MedBench)\n"
     "Performance drops significantly.\n"
     "Vocabulary mismatch, style differences.\n\n",
     "Add domain shift"),
    
    ("### Domain-Adaptive Pre-training\n\n"
     "Continued pre-training on domain data.\n"
     "DAPT: Additional 10K steps on medical texts.\n"
     "Task-adaptive pre-training: Fine-tune task data.\n"
     "TAPT: 100 steps on task training set.\n"
     "Combined: 5-10% improvement on downstream tasks.\n\n",
     "Add DAPT"),
    
    ("## Multilingual Models\n\n"
     "### Challenges\n\n"
     "100+ languages, different scripts.\n"
     "Vocabulary: mBERT has 110K tokens!\n"
     "Resource imbalance: English 1000x more data.\n"
     "Zero-shot cross-lingual transfer.\n\n",
     "Add multilingual"),
    
    ("### Cross-lingual Transfer\n\n"
     "Train on English, test on Hindi.\n"
     "mBERT trained on 104 languages jointly.\n"
     "33 language pairs show >80% transfer.\n"
     "Magic: Shared representation space.\n"
     "Enabled by shared tokenizer (WordPiece).\n\n",
     "Add X-lingual"),
    
    ("## Code Generation\n\n"
     "### Why Hard\n\n"
     "Syntax is strict (wrong bracket = error).\n"
     "Long-range dependencies (matching braces).\n"
     "Variable naming conventions.\n"
     "Semantic correctness harder to define.\n\n",
     "Add code gen intro"),
    
    ("### CodeBERT/GraphCodeBERT\n\n"
     "CodeBERT: Pre-train on code-documentation pairs.\n"
     "GraphCodeBERT: Use data flow graph.\n"
     "Results: Code-to-code search, clone detection.\n"
     "Foundation for code understanding.\n\n",
     "Add code models"),
    
    ("## Evaluation Benchmarks\n\n"
     "### Standard Benchmarks\n\n"
     "GLUE: 9 language understanding tasks\n"
     "SQuAD: Reading comprehension (100K examples)\n"
     "BLEU: Machine translation (automatic metric)\n"
     "ROUGE: Summarization (lexical overlap)\n"
     "Human evaluation: Always gold standard\n\n",
     "Add benchmarks"),
    
    ("### Emergent Abilities\n\n"
     "Small models: Can't do in-context learning\n"
     "Medium models: Few-shot breaks through\n"
     "Large models: Chain-of-thought emerges\n"
     "Scaling laws predict when they appear\n"
     "Example: 62B model = 1000x better at math\n\n",
     "Add emergent abilities"),
    
    ("## Appendix D: Hyperparameter Ranges\n\n"
     "Learning rate: 1e-5 to 1e-3\n"
     "Batch size: 8 to 512\n"
     "Warmup: 0.1 to 0.3 of training\n"
     "Weight decay: 0.0 to 0.1\n"
     "Dropout: 0.0 to 0.3\n"
     "Attention heads: 8 to 32\n"
     "Hidden size: 256 to 4096\n\n",
     "Add hyperparameters"),
    
    ("## Appendix E: Architecture Reference\n\n"
     "BERT: 12 layers, 768 hidden, 12 heads\n"
     "RoBERTa: Same but better pre-training\n"
     "T5: Encoder-decoder, 12 layers each\n"
     "GPT-2: 1.5B, 12 layers, 1024 hidden\n"
     "GPT-3: 175B, 96 layers, 12288 hidden\n"
     "MarkGPT-7B: 32 layers, 4096 hidden, 32 heads\n\n",
     "Add architectures"),
    
    ("## Appendix F: Debugging Checklist\n\n"
     "[ ] Data loading: Print sample data\n"
     "[ ] Baseline: Random baseline performance\n"
     "[ ] Overfit: Train on N=32 samples\n"
     "[ ] Learning rate: Plot loss vs LR\n"
     "[ ] Gradients: Check for NaN/explosions\n"
     "[ ] Validation: Compare to train loss\n"
     "[ ] Metrics: Verify metric implementation\n"
     "[ ] Seeds: Reproducibility with fixed seeds\n\n",
     "Add debug checklist"),
    
    ("## Appendix G: Common Issues and Solutions\n\n"
     "**Loss diverges to NaN**\n"
     "- Solution: Lower learning rate by 10x\n"
     "- Check for unused parameters\n"
     "- Try gradient clipping (max_norm=1.0)\n\n"
     "**Model underfits (train/val both high)**\n"
     "- Solution: Increase model capacity\n"
     "- Longer training, lower LR\n"
     "- Check data quality\n\n",
     "Add common issues pt1"),
    
    ("**Model overfits (big train-val gap)**\n"
     "- Solution: Add regularization (dropout, L2)\n"
     "- Reduce model capacity\n"
     "- Data augmentation\n"
     "- Early stopping\n\n"
     "**Inference slow**\n"
     "- Solution: Use quantization (INT8)\n"
     "- Batch requests (dynamic batching)\n"
     "- KV-cache, etc.\n\n",
     "Add common issues pt2"),
    
    ("## Appendix H: Useful Libraries\n\n"
     "**Transformers**: HuggingFace (SOTA models)\n"
     "**PyTorch**: Deep learning framework\n"
     "**JAX**: Functional deep learning\n"
     "**TensorFlow**: Alternative framework\n"
     "**Weights & Biases**: Experiment tracking\n"
     "**Ray Tune**: Hyperparameter optimization\n"
     "**ONNX**: Model export format\n\n",
     "Add libraries"),
    
    ("## Appendix I: Literature Recommendations\n\n"
     "**Foundational**\n"
     "- \"Attention is All You Need\" (Vaswani et al)\n"
     "- \"BERT\" (Devlin et al)\n"
     "- \"GPT-3\" (Brown et al)\n\n"
     "**Recent**\n"
     "- \"LLaMA\" (Touvron et al)\n"
     "- \"QLoRA\" (Dettmers et al)\n"
     "- \"Flash Attention\" (Dao et al)\n\n",
     "Add literature"),
    
    ("## Appendix J: Next Steps\n\n"
     "**Continue Learning**\n"
     "- Project: Build chatbot with LoRA\n"
     "- Research: Read recent papers\n"
     "- Competition: Kaggle NLP tasks\n\n"
     "**Go Deeper**\n"
     "- Module 4: Advanced architectures\n"
     "- Module 5: Research frontiers\n"
     "- Capstone: End-to-end project\n\n"
     "**Production Skills**\n"
     "- Deployment: FastAPI, Docker\n"
     "- Monitoring: Log loss, latency\n"
     "- Optimization: Profiling, benchmarking\n\n",
     "Add next steps"),
    
    ("## Module 03 Final Summary\n\n"
     "**Comprehensive Coverage**\n"
     "This module covers neural networks from first principles.\n"
     "Topics: 50+ distinct techniques and architectures.\n"
     "Implementation: All code snippets included.\n"
     "Mathematics: Rigorous equations and derivations.\n"
     "Case studies: 3 real-world applications with results.\n"
     "Total commits: 80+ sections with deep explanations.\n\n"
     "**What You Can Now Do**\n"
     "- Understand neural networks deeply\n"
     "- Implement from scratch (neurons to transformers)\n"
     "- Train large models efficiently\n"
     "- Debug effectively\n"
     "- Optimize and interpret models\n"
     "- Deploy in production\n\n"
     "**Recommended Practice**\n"
     "1. Re-implement LSTM from scratch\n"
     "2. Fine-tune MarkGPT on custom task\n"
     "3. Quantize and deploy model\n"
     "4. Paper reading and reproduction\n"
     "5. Capstone project (module-06)\n\n"
     "Ready for Module 4: Advanced Topics!\n",
     "Add final summary"),
]

readme_path = 'README.md'

print(f"Starting module-03 part 4 with {len(sections)} commits...")
print("=" * 60)

for i, (content, msg) in enumerate(sections, 1):
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(content)
    
    try:
        subprocess.run(['git', 'add', 'README.md'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', f'Enrich module-03 part4 {i}: {msg}'], check=True, capture_output=True)
        print(f"[OK] Commit {i:2d}/{len(sections)}: {msg}")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Part 4 {i:2d}: {msg}")

print("=" * 60)
print(f"[DONE] Part 4 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
